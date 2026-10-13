"""The AM-only correction must match actual HF forward and every differentiable input."""
import torch
from torch.distributed.pipelining import microbatch
from torch.nn.attention.flex_attention import BlockMask
microbatch.BlockMask = BlockMask
from nemo_automodel.components.models.glm5_next.layers import _TorchRMSNormGated, Glm5NextRMSNorm
from transformers.models.glm5_next.modeling_glm5_next import Glm5NextTextRMSNormGated
from experiments.train_infra.text_attention import install_automodel_hf_gated_norm


def test_hf_gated_norm_forward_backward_and_scope():
    torch.manual_seed(93)
    am = _TorchRMSNormGated(128, 1e-5, torch.bfloat16)
    other = _TorchRMSNormGated(128, 1e-5, torch.bfloat16)
    norm = Glm5NextRMSNorm(128, 1e-5, torch.bfloat16)
    hf = Glm5NextTextRMSNormGated(128, eps=1e-5).bfloat16()
    with torch.no_grad():
        am.weight.uniform_(0.5, 1.5)
        hf.weight.copy_(am.weight)
    x = torch.randn(1, 37, 4, 128, dtype=torch.bfloat16)
    gate = torch.randn_like(x)
    assert not torch.equal(am(x, gate), hf(x, gate)), "Fixture must expose the old rounding"
    before = other(x, gate).detach().clone()
    plain = norm(x).detach().clone()
    parameter = am.weight
    assert install_automodel_hf_gated_norm(torch.nn.ModuleList([am, norm])) == ["0"]
    assert am.weight is parameter
    torch.testing.assert_close(other(x, gate), before, atol=0, rtol=0)
    torch.testing.assert_close(norm(x), plain, atol=0, rtol=0)
    a, b = (x.clone().requires_grad_() for _ in range(2))
    ga, gb = (gate.clone().requires_grad_() for _ in range(2))
    ya, yb = am(a, ga), hf(b, gb)
    torch.testing.assert_close(ya, yb, atol=0, rtol=0)
    cotangent = torch.randn_like(ya)
    ya.backward(cotangent); yb.backward(cotangent)
    for left, right in ((a.grad, b.grad), (ga.grad, gb.grad), (am.weight.grad, hf.weight.grad)):
        torch.testing.assert_close(left, right, atol=0, rtol=0)
