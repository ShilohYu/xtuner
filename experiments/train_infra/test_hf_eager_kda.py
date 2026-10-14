"""Keep the independent HF oracle native and isolate documents at conv/core."""

import torch

from experiments.train_infra.hf_eager_kda import eager_function, packed_chunk, packed_conv
from transformers.models.glm5_next import modeling_glm5_next as hf


def test_native_conv_dispatch_and_document_reset(monkeypatch):
    native = eager_function("causal_conv1d_fn")

    def forbidden(*args, **kwargs):
        raise AssertionError("Accelerated wrapper was called")

    forbidden.__wrapped__ = native
    monkeypatch.setattr(hf, "causal_conv1d_fn", forbidden)
    x = torch.tensor([[[1.0, 2.0, 3.0, 4.0, 5.0]]], dtype=torch.bfloat16)
    weight = torch.tensor([[1.0, 2.0, 3.0]])
    result = packed_conv(x, weight, activation="silu", cu=[0, 2, 2, 5])
    expected = torch.nn.functional.silu(torch.tensor([[[3.0, 8.0, 9.0, 18.0, 26.0]]])).bfloat16()
    torch.testing.assert_close(result, expected, atol=0, rtol=0)


def test_native_chunk_matches_document_recurrence_and_head_partition(monkeypatch):
    native = eager_function("chunk_kimi_delta_attention")

    def forbidden(*args, **kwargs):
        raise AssertionError("FLA wrapper was called")

    forbidden.__wrapped__ = native
    monkeypatch.setattr(hf, "chunk_kimi_delta_attention", forbidden)
    recurrent = eager_function("recurrent_kimi_delta_attention")
    torch.manual_seed(731)
    q, k = [torch.randn(1, 75, 10, 4) for _ in range(2)]
    v = torch.randn(1, 75, 10, 6)
    g = -torch.rand_like(q)
    beta = torch.rand(1, 75, 10)
    cu = [0, 7, 7, 75]
    actual, state = packed_chunk(q, k, v, g, beta, cu=cu)
    assert state is None and actual.shape == v.shape
    for a, b in [(0, 7), (7, 75)]:
        args = [x[:, a:b] for x in (q, k, v, g, beta)]
        whole = native(*args, use_qk_l2norm_in_kernel=True)[0]
        expected = recurrent(*args, initial_state=None, output_final_state=False, use_qk_l2norm_in_kernel=True)[0]
        torch.testing.assert_close(actual[:, a:b], whole, atol=0, rtol=0)
        torch.testing.assert_close(actual[:, a:b], expected, atol=2e-6, rtol=2e-5)
