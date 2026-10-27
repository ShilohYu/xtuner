"""Explicit SP1 precision comparison against XT; retain native AM attention cores."""

import os
from contextlib import ExitStack, contextmanager
from functools import partial
from importlib import import_module
from types import MethodType
from typing import Any, Callable, Iterator
from unittest.mock import patch

import torch
import torch.nn.functional as F

from experiments.train_infra.text_attention import _decoder_with_shared_hc_post, _text_mlp, _text_rms_norm


@contextmanager
def automodel_moe_reduction_precision(text: torch.nn.Module, *, training: bool = False) -> Iterator[dict[str, Any]]:
    """Use native sorted routing and a common FP32 combine after AM's own expert GEMMs.

    Args:
        text (torch.nn.Module): AutoModel text model, router, or grouped experts.

    Yields:
        dict[str, Any]: Installed forward-only EP1 diagnostic settings.
    """
    from nemo_automodel.components.moe.experts import GroupedExperts
    from nemo_automodel.components.moe.layers import Gate

    assert os.environ.get("PERMUTE_COMPUTE_DTYPE") == "fp32", "Set FP32 combine before starting the process"
    report: dict[str, Any] = {
        "scope": "experimental training, EP1" if training else "forward/no_grad, EP1",
        "route_order": "sorted=True",
        "combine": "XT native FP32",
        "modules": {},
    }
    with ExitStack() as stack:
        for name, module in text.named_modules():
            if isinstance(module, Gate):
                assert module.n_groups == 1 and module.score_func == "sigmoid_with_bias"
                stack.enter_context(patch.object(module, "_route_scores", _ordered_route_scores(module)))
                report["modules"][name] = "native selection with sorted topk before normalization"
            elif isinstance(module, GroupedExperts):
                assert module.use_torch_mm and not module.use_mxfp8 and module.config.apply_router_weight_after_down
                stack.enter_context(patch.object(module, "forward", _native_experts_with_fp32_combine(module, training=training)))
                report["modules"][name] = "native AM GEMMs, XT FP32 combine on own expert values"
        assert report["modules"]
        yield report


def _ordered_route_scores(module: torch.nn.Module) -> Callable[..., Any]:
    native_route = module._route_scores
    native_topk = torch.topk

    def route(scores: torch.Tensor) -> Any:
        def topk(states: torch.Tensor, k: int, *args: Any, **kwargs: Any) -> Any:
            if states.shape == scores.shape and k == module.topk:
                kwargs["sorted"] = True
            return native_topk(states, k, *args, **kwargs)

        with patch.object(torch, "topk", topk):
            return native_route(scores)

    return route


def _combine_native_fp32(values: torch.Tensor, slots: torch.Tensor, weights: torch.Tensor, *, training: bool = False) -> torch.Tensor:
    from xtuner.v1.ops import unpermute

    assert os.environ.get("PERMUTE_COMPUTE_DTYPE") == "fp32"
    assert training or not torch.is_grad_enabled()
    tokens, topk = weights.shape
    row_map = torch.full((topk, tokens), -1, dtype=torch.int32, device=values.device)
    row_map[slots % topk, slots // topk] = torch.arange(len(slots), dtype=torch.int32, device=values.device)
    return unpermute(values, row_map.flatten(), probs=weights)


def _native_experts_with_fp32_combine(module: torch.nn.Module, *, training: bool = False) -> Callable[..., torch.Tensor]:
    from nemo_automodel.components.moe import experts

    native_forward = module.forward
    native_permute = experts._permute_tokens_for_grouped_mm
    native_weight = experts._apply_router_weight_fp32

    def forward(
        x: torch.Tensor, token_mask: torch.Tensor, weights: torch.Tensor, indices: torch.Tensor
    ) -> torch.Tensor:
        assert training or not torch.is_grad_enabled()
        mesh = getattr(module.gate_and_up_projs, "device_mesh", None)
        assert mesh is None or mesh.size() == 1
        captured: dict[str, torch.Tensor] = {}

        def permute(*args: Any, **kwargs: Any) -> Any:
            result = native_permute(*args, **kwargs)
            assert kwargs.get("return_slot_ids") and "slots" not in captured
            captured["slots"] = result[1]
            return result

        def weight(values: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
            assert "values" not in captured
            captured["values"] = values
            return native_weight(values, *args, **kwargs)

        with (
            patch.object(experts, "_permute_tokens_for_grouped_mm", permute),
            patch.object(experts, "_apply_router_weight_fp32", weight),
        ):
            result = native_forward(x, token_mask, weights, indices)
        if not captured["slots"].numel():
            return result
        # Diagnostic recombine: keep the native expert forward observable and change only its reduction.
        return _combine_native_fp32(captured["values"], captured["slots"], weights, training=training)

    return forward


@contextmanager
def native_dsa_precision(text: torch.nn.Module, backend: str, *, training: bool = False) -> Iterator[dict[str, Any]]:
    """Align DSA projection geometry, index scoring and key order for forward comparison.

    Args:
        text (torch.nn.Module): Model or individual attention module to adapt.
        backend (str): Either ``xtuner`` or ``automodel``.

    Yields:
        dict[str, Any]: Installed diagnostic settings. Native sparse cores are retained.
    """
    report: dict[str, Any] = {
        "scope": "experimental training, batch1, SP1/CP1" if training else "forward/no_grad, batch1, SP1/CP1",
        "backend": backend, "modules": [],
    }
    with ExitStack() as stack:
        if backend == "automodel":
            from nemo_automodel.components.models.glm5_next.layers import Glm5NextSparseAttention

            report["projections"] = "own q_a/kv_a, index wk/weights/compress: full-pack GEMM, native pools/select/core"
            for name, module in text.named_modules():
                if isinstance(module, Glm5NextSparseAttention):
                    stack.enter_context(patch.object(module, "forward", _am_dsa_full_pack(module, training=training)))
                    report["modules"].append(name)
        elif backend == "xtuner":
            from xtuner.v1.model.moe.glm53.nope_dsa_mla import NoPEDSAMultiLatentAttention
            from xtuner.v1.ops.sparse_mla.kpool import torch_kpool_topk_indices

            report.update(indexer="torch", key_order="ascending, invalid entries last")
            for name, module in text.named_modules():
                if isinstance(module, NoPEDSAMultiLatentAttention):
                    stack.enter_context(patch.object(module.indexer, "_topk_indices_fn", torch_kpool_topk_indices))
                    stack.enter_context(
                        patch.object(module, "sparse_mla_func", _sorted_sparse(module.sparse_mla_func))
                    )
                    stack.enter_context(patch.object(module, "forward", _xt_dsa_forward_guard(module, training=training)))
                    report["modules"].append(name)
        else:
            raise ValueError(backend)
        assert report["modules"], "No DSA modules selected"
        yield report


def _am_dsa_full_pack(module: torch.nn.Module, *, training: bool = False) -> Callable[..., torch.Tensor]:
    native_forward = module.forward

    def forward(hidden_states: torch.Tensor, *, packed_context: Any, **kwargs: Any) -> torch.Tensor:
        # shortcut: batch1/CP1 diagnostic; broader layouts require separate validation.
        assert (training or not torch.is_grad_enabled()) and hidden_states.shape[0] == 1
        assert not packed_context.cp_enabled
        values = {
            module.q_a_proj: module.q_a_proj(hidden_states),
            module.kv_a_proj_with_mqa: module.kv_a_proj_with_mqa(hidden_states),
            module.indexer.wk: module.indexer.wk(hidden_states),
            module.indexer.weights_proj: module.indexer.weights_proj(hidden_states),
        }
        gate_weight = module.indexer.index_kpool_compress_gate
        gate_values = F.linear(hidden_states, gate_weight)
        native_linear = F.linear

        def slice_values(full: torch.Tensor, chunk: torch.Tensor) -> torch.Tensor:
            assert chunk.untyped_storage().data_ptr() == hidden_states.untyped_storage().data_ptr()
            assert chunk.stride(-1) == 1 and chunk.stride(-2) == hidden_states.stride(1)
            offset, remainder = divmod(
                chunk.storage_offset() - hidden_states.storage_offset(), hidden_states.stride(1)
            )
            assert remainder == 0 and 0 <= offset <= hidden_states.shape[1] - chunk.shape[-2]
            result = full[:, offset : offset + chunk.shape[-2]]
            return result.squeeze(0) if chunk.ndim == 2 else result

        def projection(full: torch.Tensor) -> Callable[[torch.Tensor], torch.Tensor]:
            return lambda chunk: slice_values(full, chunk)

        def linear(states: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
            if weight is gate_weight:
                assert bias is None
                return slice_values(gate_values, states)
            return native_linear(states, weight, bias)

        with ExitStack() as geometry:
            for child, full in values.items():
                geometry.enter_context(patch.object(child, "forward", projection(full)))
            # Keep native prepare_pools, including its short-document and tail behavior.
            geometry.enter_context(patch.object(F, "linear", linear))
            return native_forward(hidden_states, packed_context=packed_context, **kwargs)

    return forward


def _sorted_sparse(native: Callable[..., Any]) -> Callable[..., Any]:
    def sparse(q: torch.Tensor, kv: torch.Tensor, ids: torch.Tensor, scale: float, **kwargs: Any) -> Any:
        sentinel = torch.iinfo(ids.dtype).max
        ordered = ids.masked_fill(ids < 0, sentinel).sort(-1).values
        ordered = ordered.masked_fill(ordered == sentinel, -1).contiguous()
        return native(q, kv, ordered, scale, **kwargs)

    return sparse


def _xt_dsa_forward_guard(module: torch.nn.Module, *, training: bool = False) -> Callable[..., Any]:
    native_forward = module.forward

    def forward(
        hidden_states: torch.Tensor, position_embeddings: Any = None, seq_ctx: Any = None, **kwargs: Any
    ) -> Any:
        assert (training or not torch.is_grad_enabled()) and hidden_states.shape[0] == 1
        assert seq_ctx is not None
        mesh = seq_ctx.sequence_parallel_mesh
        assert mesh is None or mesh.size() == 1
        return native_forward(hidden_states, position_embeddings=position_embeddings, seq_ctx=seq_ctx, **kwargs)

    return forward


@contextmanager
def automodel_native_core_precision(
    text: torch.nn.Module, *, routed_activation: bool = False, training: bool = False
) -> Iterator[dict[str, Any]]:
    """Temporarily align primitives and packed metadata without replacing parameters.

    Keep this context active through backward. This diagnostic shares mHC/gated-norm
    primitives with XT but leaves AM's DSA/KDA entrypoints and routed expert execution intact.
    Backward through the routed activation requires the explicit training opt-in.
    """
    from nemo_automodel.components.models.glm5_next import layers
    from nemo_automodel.components.models.glm5_next.cp import Glm5NextPackedContext
    from nemo_automodel.components.moe.experts import GroupedExperts
    from nemo_automodel.components.moe.layers import MLP
    from torch.distributed.fsdp import FSDPModule

    from xtuner.v1.ops.gated_deltanet.rms_norm_gated import rms_norm_gated
    from xtuner.v1.ops.kda.chunk_cumsum import chunk_cumsum

    fla_forward = import_module("fla.ops.kda.chunk_fwd")
    native_gate = layers._fused_kda_gate
    native_row = Glm5NextPackedContext.row_cu_seqlens
    native_core = layers._chunk_kda
    parameter_ids = [id(p) for p in text.parameters()]
    fsdp_managed = any(isinstance(module, FSDPModule) for module in text.modules())
    parameter_structure = {name: (tuple(p.shape), p.requires_grad) for name, p in text.named_parameters()}

    def unweighted_norm(self, states):
        return F.rms_norm(states, (states.shape[-1],), weight=None, eps=self.eps)

    def hyper_connection(self, states):
        # Reuse the flattened view in both branches so BF16 gradients merge as in XT.
        flat = states.flatten(2)
        normed = self.input_norm(flat.float())
        hc = self.hc_mult
        mixes = F.linear(normed, self.fn.float()).split([hc, hc, hc * hc], dim=-1)
        pre, post, comb = self._fp32_params(*mixes, hc, self.hc_eps, self.hc_sinkhorn_iters)
        collapsed = (pre.unsqueeze(-1) * flat.view(states.shape)).sum(-2).to(states.dtype)
        return post, comb, collapsed

    def gated_norm(self, states, gate):
        return rms_norm_gated(states, gate, self.weight, None, activation="sigmoid", eps=self.variance_epsilon)

    def fp32_gate(gate, *args, **kwargs):
        return native_gate(gate.float(), *args, **kwargs)

    def prefix(g, chunk_size, scale, cu_seqlens, chunk_indices):
        assert chunk_size == 64
        return chunk_cumsum(g, cu_seqlens, chunk_indices, scale)

    def row_int32(self, row):
        if self.cp_enabled:
            raise NotImplementedError("Native-core precision diagnostic has only been validated at CP1")
        assert self.doc_ids.shape[1] < 2**31
        device, cpu = native_row(self, row)
        return device.to(torch.int32), cpu.to(torch.int32)

    report = {
        "mode": "AM primitives aligned to XT; native attention cores retained; diagnostic SP1",
        "ordinary_rms_norm": "AM text uses XT native RMSNorm; affine before final cast; XT model unchanged",
        "gate_input": "float32",
        "packed_indices": "int32",
        "gate_prefix": "XT fixed 64-token scan",
        "routed_activation": "XT BF16 clamped SwiGLU" if routed_activation else "native AM",
        "modules": {},
        "parameter_check": "identities at install; names/shapes/trainability after FSDP execution"
        if fsdp_managed
        else "identities before and after execution",
    }
    with ExitStack() as stack:
        for name, module in text.named_modules():
            replacement = None
            if routed_activation and isinstance(module, GroupedExperts):
                assert module.config.expert_activation == "swiglu" and module.config.swiglu_limit > 0
                assert module.config.apply_router_weight_after_down and not module.use_mxfp8
                stack.enter_context(
                    patch.object(
                        module,
                        "expert_activation_grouped",
                        partial(_routed_activation, limit=module.config.swiglu_limit, training=training),
                    )
                )
                report["modules"][name + ".expert_activation_grouped"] = "XT BF16 clamped SwiGLU"
            if isinstance(module, layers.Glm5NextUnweightedRMSNorm):
                replacement = unweighted_norm
            elif isinstance(module, layers.Glm5NextHyperConnection):
                replacement = hyper_connection
            elif isinstance(module, layers._TorchRMSNormGated):
                replacement = gated_norm
            elif isinstance(module, layers.Glm5NextRMSNorm):
                replacement = _text_rms_norm
            elif isinstance(module, layers.Glm5NextDecoderLayer):
                replacement = _decoder_with_shared_hc_post
            elif isinstance(module, MLP) and module.activation == "swiglu" and module.swiglu_limit > 0:
                replacement = _text_mlp
            if replacement is not None:
                stack.enter_context(patch.object(module, "forward", MethodType(replacement, module)))
                report["modules"][name] = replacement.__name__
        assert report["modules"], "No GLM text modules selected"
        stack.enter_context(patch.object(layers, "_fused_kda_gate", fp32_gate))
        stack.enter_context(patch.object(fla_forward, "chunk_local_cumsum", prefix))
        stack.enter_context(patch.object(Glm5NextPackedContext, "row_cu_seqlens", row_int32))
        assert layers._chunk_kda is native_core
        assert parameter_ids == [id(p) for p in text.parameters()]
        yield report
        assert layers._chunk_kda is native_core
        # FSDP lazily replaces Parameter objects while unsharding/resharding.
        if fsdp_managed:
            assert parameter_structure == {
                name: (tuple(p.shape), p.requires_grad) for name, p in text.named_parameters()
            }
        else:
            assert parameter_ids == [id(p) for p in text.parameters()]


def _routed_activation(states: torch.Tensor, probs: torch.Tensor, *, limit: float, training: bool = False) -> torch.Tensor:
    from xtuner.v1.ops.act_fn import native_clamped_swiglu

    assert training or not torch.is_grad_enabled(), "Backward requires the explicit training profile"
    return (native_clamped_swiglu(states, limit=limit) * probs).to(states.dtype)
