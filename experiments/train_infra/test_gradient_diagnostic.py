"""Gradient observation must preserve backward with either checkpoint implementation."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import CheckpointImpl, checkpoint_wrapper

from experiments.train_infra.gradient_diagnostic import BoundaryGradients


@pytest.mark.parametrize("implementation", [CheckpointImpl.REENTRANT, CheckpointImpl.NO_REENTRANT])
def test_observation_preserves_checkpoint_backward(tmp_path, implementation):
    class Layer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.input_layernorm = torch.nn.LayerNorm(4)
            self.self_attn = checkpoint_wrapper(torch.nn.Linear(4, 4), checkpoint_impl=implementation)
            self.post_attention_layernorm = torch.nn.LayerNorm(4)

        def forward(self, x):
            return self.post_attention_layernorm(self.self_attn(self.input_layernorm(x)))

    text = torch.nn.Module()
    text.layers = torch.nn.ModuleDict({"0": Layer()})
    text.norm = torch.nn.LayerNorm(4)
    model = SimpleNamespace(model=SimpleNamespace(language_model=text))
    x = torch.randn(1, 3, 4, requires_grad=True)
    target = torch.randn_like(x)
    baseline = text.norm(text.layers["0"](x))
    baseline.backward(target)
    expected = [x.grad.clone(), *[p.grad.clone() for p in text.parameters()]]
    x.grad = None
    text.zero_grad(set_to_none=True)
    with patch("torch.distributed.get_rank", return_value=0):
        observer = BoundaryGradients(model, "automodel", tmp_path, 0)
        output = text.norm(text.layers["0"](x))
        output.backward(target)
        observer.close()
    assert torch.equal(output, baseline)
    assert all(torch.equal(a, b) for a, b in zip(expected, [x.grad, *[p.grad for p in text.parameters()]]))
    assert torch.equal(observer.values["final_norm.output"], target)
    assert (tmp_path / "gradients/step_000/boundaries.safetensors").exists()
