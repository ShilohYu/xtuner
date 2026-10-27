"""Packed DSA projections preserve gradients across an uneven document boundary."""

from types import SimpleNamespace

import torch

from experiments.train_infra.precision_profile import _am_dsa_full_pack


def test_kv_full_pack_and_gradient():
    class DSA(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.q_a_proj = torch.nn.Linear(4, 4, bias=False)
            self.kv_a_proj_with_mqa = torch.nn.Linear(4, 4, bias=False)
            self.indexer = torch.nn.Module()
            self.indexer.wk = torch.nn.Linear(4, 4, bias=False)
            self.indexer.weights_proj = torch.nn.Linear(4, 4, bias=False)
            self.indexer.index_kpool_compress_gate = torch.nn.Parameter(torch.randn(4, 4))

        def forward(self, states, *, packed_context):
            return torch.cat([self.kv_a_proj_with_mqa(states[:, a:b]) for a, b in ((0, 5), (5, 8))], dim=1)

    module = DSA().double()
    states = torch.randn(1, 8, 4, dtype=torch.float64, requires_grad=True)
    cotangent = torch.randn_like(states)
    expected = module.kv_a_proj_with_mqa(states)
    expected_grads = torch.autograd.grad(expected, (states, module.kv_a_proj_with_mqa.weight), cotangent)
    calls = []
    original = module.kv_a_proj_with_mqa.forward

    def recording_projection(value):
        calls.append(value.shape[1])
        return original(value)

    module.kv_a_proj_with_mqa.forward = recording_projection
    actual = _am_dsa_full_pack(module, training=True)(states, packed_context=SimpleNamespace(cp_enabled=False))
    actual_grads = torch.autograd.grad(actual, (states, module.kv_a_proj_with_mqa.weight), cotangent)
    assert calls == [8], calls
    assert torch.equal(actual, expected)
    assert all(torch.equal(a, b) for a, b in zip(actual_grads, expected_grads))
    assert module.kv_a_proj_with_mqa.forward is recording_projection
