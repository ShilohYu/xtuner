"""Check the trainable combine against independent weighted-sum derivatives."""

import pytest
import torch
from torch.utils.checkpoint import checkpoint

from experiments.train_infra.precision_profile import _combine_native_fp32


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA native combine")
def test_combine_backward_and_checkpoint():
    torch.manual_seed(19)
    tokens, topk, hidden = 11, 8, 128
    # Missing slots model padding; shuffled slots model expert dispatch order.
    slots = torch.randperm(tokens * topk, device="cuda")[:-7]
    values = torch.randn(len(slots), hidden, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    weights = torch.rand(tokens, topk, device="cuda", requires_grad=True)
    cotangent = torch.randn(tokens, hidden, device="cuda", dtype=torch.bfloat16)
    expected_values = (cotangent[slots // topk].float() * weights.flatten()[slots, None]).bfloat16()
    expected_weights = torch.zeros_like(weights).flatten()
    expected_weights[slots] = (values.float() * cotangent[slots // topk].float()).sum(-1)
    def combine(v, w):
        return _combine_native_fp32(v, slots, w, training=True)
    direct = combine(values, weights)
    actual = torch.autograd.grad(direct, (values, weights), cotangent)
    torch.testing.assert_close(actual[0], expected_values, rtol=0, atol=0)
    torch.testing.assert_close(actual[1].flatten(), expected_weights, rtol=2e-5, atol=2e-5)
    replayed = checkpoint(combine, values, weights, use_reentrant=False)
    replayed_grad = torch.autograd.grad(replayed, (values, weights), cotangent)
    torch.testing.assert_close(replayed, direct, rtol=0, atol=0)
    for x, y in zip(replayed_grad, actual):
        torch.testing.assert_close(x, y, rtol=0, atol=0)
    assert all(torch.isfinite(g).all() and g.abs().sum() > 0 for g in actual)
