"""Native KDA modules retain forward/backward parity with shared kernel settings."""

import pytest
import torch
from torch.distributed.pipelining import microbatch
from torch.nn.attention.flex_attention import BlockMask

microbatch.BlockMask = BlockMask
from nemo_automodel.components.models.glm5_next import layers
from nemo_automodel.components.models.glm5_next.config import Glm5NextTextConfig
from experiments.train_infra.text_attention import install_shared_text_kernels, install_text_rms_norm
from xtuner.v1.data_proto import SequenceContext
from xtuner.v1.module.attention.kda import KimiDeltaAttention


def test_text_norm_adapter_preserves_native_vision(monkeypatch):
    from nemo_automodel.components.models.glm5_next import vision
    from xtuner.v1.ops import rms_norm

    for cls in (layers.Glm5NextRMSNorm, layers.Glm5NextUnweightedRMSNorm,
                layers._TorchRMSNormGated, layers.Glm5NextDecoderLayer):
        monkeypatch.setattr(cls, "forward", cls.forward)
    for name in ("_chunk_kda", "_fused_kda_gate", "_fla_causal_conv1d"):
        monkeypatch.setattr(layers, name, getattr(layers, name))
    torch.manual_seed(23)
    visual = vision.Glm5NextRMSNorm(32, 1e-5, torch.bfloat16)
    text = layers.Glm5NextRMSNorm(32, 1e-5, torch.bfloat16)
    gated = layers._TorchRMSNormGated(32, 1e-5, torch.bfloat16)
    inputs = torch.randn(2, 37, 32, dtype=torch.bfloat16)
    with torch.no_grad():
        visual.weight.uniform_(0.5, 1.5)
        text.weight.copy_(visual.weight)
        baseline = visual(inputs)
        native_forward = layers.Glm5NextRMSNorm.forward
        install_shared_text_kernels(layers)
        assert install_text_rms_norm(torch.nn.ModuleList([text, gated])) == 1
        assert gated.forward.__func__ is layers._TorchRMSNormGated.forward
        assert layers.Glm5NextRMSNorm.forward is native_forward
        torch.testing.assert_close(visual(inputs), baseline, atol=0, rtol=0)
        torch.testing.assert_close(text(inputs), rms_norm(inputs, text.weight, 1e-5), atol=0, rtol=0)


@pytest.mark.parametrize("moe", [False, True])
def test_native_decoder_preserves_both_residual_paths(monkeypatch, moe):
    from nemo_automodel.components.models.common import BackendConfig
    from xtuner.v1.module.decoder_layer.mhc import hc_post

    for cls in (layers.Glm5NextUnweightedRMSNorm, layers._TorchRMSNormGated, layers.Glm5NextDecoderLayer):
        monkeypatch.setattr(cls, "forward", cls.forward)
    for name in ("_chunk_kda", "_fused_kda_gate", "_fla_causal_conv1d"):
        monkeypatch.setattr(layers, name, getattr(layers, name))

    class Update(torch.nn.Linear):
        def forward(self, x, padding_mask=None, **kwargs):
            assert padding_mask is mask
            return super().forward(x)

    torch.manual_seed(29)
    cfg = Glm5NextTextConfig(hidden_size=32, intermediate_size=64, num_hidden_layers=1,
                            layer_types=["linear_attention"], mlp_layer_types=["dense"],
                            linear_num_heads=2, linear_head_dim=128)
    layer = layers.Glm5NextDecoderLayer(cfg, 0, None, BackendConfig(linear="torch"))
    layer.self_attn = Update(32, 32, bias=False, dtype=torch.bfloat16)
    layer.is_moe_layer = moe
    if moe:
        layer.mlp = Update(32, 32, bias=False, dtype=torch.bfloat16)
    with torch.no_grad():
        for p in layer.parameters():
            p.normal_(std=0.1)
    parameter_ids = {n: id(p) for n, p in layer.named_parameters()}
    install_shared_text_kernels(layers)
    install_text_rms_norm(layer)
    assert parameter_ids == {n: id(p) for n, p in layer.named_parameters()}
    mask = torch.zeros(1, 7, dtype=torch.bool)
    x = torch.randn(1, 7, cfg.hc_mult, 32, dtype=torch.bfloat16, requires_grad=True)
    post, comb, collapsed = layer.attn_hc(x)
    update = layer.self_attn(layer.input_layernorm(collapsed), padding_mask=mask)
    intermediate = hc_post(update, x, post, comb)
    post, comb, collapsed = layer.ffn_hc(intermediate)
    normalized = layer.post_attention_layernorm(collapsed)
    update = layer.mlp(normalized, mask) if moe else layer.mlp(normalized)
    expected = hc_post(update, intermediate, post, comb)
    actual = layer(x, packed_context=None, padding_mask=mask)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    actual.float().square().sum().backward()
    assert torch.isfinite(x.grad).all()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in layer.parameters())


@pytest.mark.gpu
def test_packed_native_kda_forward_and_backward(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for the real KDA kernels")
    for cls in (layers.Glm5NextRMSNorm, layers.Glm5NextUnweightedRMSNorm,
                layers._TorchRMSNormGated, layers.Glm5NextDecoderLayer):
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
