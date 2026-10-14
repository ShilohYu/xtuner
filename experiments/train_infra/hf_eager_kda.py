"""Actual HF Torch references, with document resets for packed diagnostics."""

import inspect
from pathlib import Path

import torch

from transformers.models.glm5_next import modeling_glm5_next as hf


def eager_function(name):
    # Installed FLA/causal-conv packages override HF even without use_kernels=True.
    fn = inspect.unwrap(getattr(hf, name))
    assert Path(fn.__code__.co_filename).resolve() == Path(hf.__file__).resolve(), name
    return fn


def packed_conv(hidden_states, weight, bias=None, activation=None, *, cu, **kwargs):
    assert cu[0] == 0 and cu[-1] == hidden_states.shape[-1]
    assert all(a <= b for a, b in zip(cu[:-1], cu[1:]))
    fn = eager_function("causal_conv1d_fn")
    out = torch.empty_like(hidden_states)
    for a, b in zip(cu[:-1], cu[1:]):
        if b > a:
            out[:, :, a:b] = fn(hidden_states[:, :, a:b], weight, bias=bias, activation=activation)
    return out


def packed_chunk(query, key, value, g, beta, *, cu, use_qk_l2norm_in_kernel=True, **kwargs):
    assert cu[0] == 0 and cu[-1] == query.shape[1]
    assert all(a <= b for a, b in zip(cu[:-1], cu[1:]))
    fn = eager_function("chunk_kimi_delta_attention")
    out = torch.empty_like(value)
    for a, b in zip(cu[:-1], cu[1:]):
        if b <= a:
            continue
        # Heads are independent; bound HF's O(T * H * chunk_size * D) temporaries.
        for h in range(0, query.shape[2], 8):
            sl = (slice(None), slice(a, b), slice(h, h + 8))
            out[sl], _ = fn(
                query[sl], key[sl], value[sl], g[sl], beta[sl], use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel
            )
    return out, None
