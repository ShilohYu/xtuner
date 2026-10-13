"""Native KDA modules retain forward/backward parity with shared kernel settings."""

import pytest
import torch
from torch.distributed.pipelining import microbatch
from torch.nn.attention.flex_attention import BlockMask

microbatch.BlockMask = BlockMask
from nemo_automodel.components.models.glm5_next import layers
from nemo_automodel.components.models.glm5_next.config import Glm5NextTextConfig
from experiments.train_infra.text_attention import install_shared_text_kernels
from xtuner.v1.data_proto import SequenceContext
from xtuner.v1.module.attention.kda import KimiDeltaAttention


@pytest.mark.gpu
def test_packed_native_kda_forward_and_backward(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for the real KDA kernels")
    for cls in (layers.Glm5NextRMSNorm, layers.Glm5NextUnweightedRMSNorm, layers._TorchRMSNormGated):
        monkeypatch.setattr(cls, "forward", cls.forward)
    for name in ("_chunk_kda", "_fused_kda_gate", "_fla_causal_conv1d"):
        monkeypatch.setattr(layers, name, getattr(layers, name))
    install_shared_text_kernels(layers)
    torch.manual_seed(197)
    cfg = Glm5NextTextConfig(hidden_size=32, num_hidden_layers=1,
                            layer_types=["linear_attention"], mlp_layer_types=["dense"],
                            linear_num_heads=2, linear_head_dim=128)
    xt = KimiDeltaAttention(hidden_size=32, num_heads=2, head_dim=128).cuda().bfloat16().train()
    xt.A_log.data = xt.A_log.data.float()
    xt.dt_bias.data = xt.dt_bias.data.float()
    am = layers.Glm5NextLinearAttention(cfg, 0).cuda().bfloat16().train()
    am._fp32_params.float()
    with torch.no_grad():
        xt.A_log.zero_()
        xt.dt_bias.normal_(std=0.1)
        source = dict(xt.named_parameters())
        for name, p in am.named_parameters():
            p.copy_(source[name.removeprefix("_fp32_params.")])
    inputs = torch.randn(1, 96, 32, device="cuda", dtype=torch.bfloat16)
    x, y = (inputs.clone().requires_grad_() for _ in range(2))
    cu = torch.tensor([0, 37, 96], device="cuda", dtype=torch.int32)
    context = SequenceContext(input_ids=torch.zeros(1, 96, device="cuda", dtype=torch.long),
                              cu_seq_lens_q=cu, cu_seq_lens_k=cu,
                              max_length_q=59, max_length_k=59, device="cuda")
    xo = xt(x, seq_ctx=context)["projected_output"]
    yo = am._core(y, cu_seqlens=cu)
    torch.testing.assert_close(xo, yo, atol=0, rtol=0)
    cotangent = torch.randn_like(xo, dtype=torch.float32)
    (xo.float() * cotangent).sum().backward()
    (yo.float() * cotangent).sum().backward()
    torch.testing.assert_close(x.grad, y.grad, atol=1e-5, rtol=1e-5)
    target = {n.removeprefix("_fp32_params."): p for n, p in am.named_parameters()}
    assert source.keys() == target.keys()
    for name, p in source.items():
        assert p.grad is not None and target[name].grad is not None, name
        torch.testing.assert_close(p.grad, target[name].grad, atol=1e-5, rtol=1e-5, msg=name)
