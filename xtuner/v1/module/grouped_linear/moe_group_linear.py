import torch
import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor, Shard, distribute_tensor

from xtuner.v1.float8.config import Float8Config, ScalingGranularity
from xtuner.v1.float8.float8_gmm_tile_wise import TileWiseFloat8GroupedLinear
from xtuner.v1.ops import group_gemm
from xtuner.v1.ops.moe.cuda.group_gemm import ultra_ep_group_gemm


class GroupedLinear(nn.Module):
    # TODO:Missng example docs
    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_routed_experts: int,
        moe_bias: bool = False,
        ep_mesh: DeviceMesh | None = None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_routed_experts = num_routed_experts
        weight = torch.empty(num_routed_experts * out_features, in_features)

        self.ep_mesh = ep_mesh
        if self.ep_mesh is not None and self.ep_mesh.size() > 1:
            self.weight = nn.Parameter(distribute_tensor(weight, ep_mesh, [Shard(0)]))
        else:
            self.weight = nn.Parameter(weight)

        self.moe_bias = moe_bias
        self._ultra_ep_replica_weight: torch.Tensor | None = None
        self._ultra_ep_replica_grad: torch.Tensor | None = None
        if self.moe_bias:
            bias = torch.zeros(num_routed_experts, out_features)
            if self.ep_mesh is not None and self.ep_mesh.size() > 1:
                self.bias = nn.Parameter(distribute_tensor(bias, ep_mesh, [Shard(0)]))
            else:
                self.bias = nn.Parameter(torch.zeros(num_routed_experts, out_features))

    def configure_ultra_ep_buffers(
        self,
        replica_weight: torch.Tensor,
        replica_grad: torch.Tensor,
    ) -> None:
        """Attach Manager-owned replica buffers without registering parameters."""
        if self.moe_bias:
            raise NotImplementedError("UltraEP does not currently support grouped expert bias")
        if replica_weight.ndim != 3 or replica_weight.shape[1:] != (self.out_features, self.in_features):
            raise ValueError(
                "Unexpected UltraEP replica weight shape: "
                f"expected [R, {self.out_features}, {self.in_features}], got {tuple(replica_weight.shape)}"
            )
        if replica_grad.shape != replica_weight.shape or replica_grad.dtype != torch.float32:
            raise ValueError("UltraEP replica grad must be an FP32 tensor matching replica weight shape")
        # Tensor attributes are intentionally plain references: Manager owns
        # their storage and they must stay out of state_dict()/parameters().
        self._ultra_ep_replica_weight = replica_weight
        self._ultra_ep_replica_grad = replica_grad

    def forward(self, x: torch.Tensor, tokens_per_expert: torch.Tensor, decoding: bool = False):
        weight = self.weight.to_local() if isinstance(self.weight, DTensor) else self.weight
        weight = weight.view(-1, self.out_features, self.in_features)
        if self._ultra_ep_replica_weight is None:
            out = group_gemm(x, weight, tokens_per_expert)
        else:
            assert self._ultra_ep_replica_grad is not None
            out = ultra_ep_group_gemm(
                x,
                weight,
                self._ultra_ep_replica_weight,
                self._ultra_ep_replica_grad,
                tokens_per_expert,
            )

        if self.moe_bias:
            bias = self.bias.to_local() if isinstance(self.bias, DTensor) else self.bias
            out = out + bias.repeat_interleave(tokens_per_expert, dim=0)  # TODO: 无法 compile
        return out


def build_grouped_linear(
    in_features: int,
    out_features: int,
    num_routed_experts: int,
    moe_bias: bool = False,
    ep_mesh: DeviceMesh | None = None,
    float8_cfg: Float8Config | None = None,
):
    """Build a grouped linear layer with optional float8 support."""
    if float8_cfg is None or float8_cfg.scaling_granularity_grouped_gemm is None:
        return GroupedLinear(in_features, out_features, num_routed_experts, moe_bias=moe_bias, ep_mesh=ep_mesh)
    elif float8_cfg.scaling_granularity_grouped_gemm == ScalingGranularity.TILEWISE:
        return TileWiseFloat8GroupedLinear(
            in_features, out_features, num_routed_experts, moe_bias=moe_bias, ep_mesh=ep_mesh
        )
    else:
        raise NotImplementedError(
            f"Unsupported grouped GEMM float8 scaling granularity: {float8_cfg.scaling_granularity_grouped_gemm}"
        )
