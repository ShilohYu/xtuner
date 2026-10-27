"""Opt-in real-loss boundary gradients; retain native backward and optimizer paths."""

from pathlib import Path

import torch
import torch.distributed as dist
from safetensors.torch import save_file


class BoundaryGradients:
    def __init__(self, model, backend, out, step):
        self.target = Path(out) / "gradients" / f"step_{step:03d}"
        self.handles = []
        self.values = {}
        self.activations = {}
        text = model.language_model if backend == "xtuner" else model.model.language_model
        for name, module in [("final_norm", text.norm), *[(f"layer{i}", m) for i, m in text.layers.items()]]:
            self.add(module, name, backend == "xtuner" and name.startswith("layer"))
            if name.startswith("layer"):
                for child in ("input_layernorm", "post_attention_layernorm", "self_attn"):
                    self.add(getattr(module, child), name + "." + child, backend == "xtuner" and child == "self_attn")

    def add(self, module, name, dictionary):
        def observe(module, inputs, kwargs, output):
            if not torch.is_grad_enabled() or dist.get_rank() != 0:
                return
            value = (
                output["projected_output" if name.endswith("self_attn") else "hidden_states"] if dictionary else output
            )
            state = inputs[0] if inputs else kwargs.get("hidden_states")
            if name == "final_norm":
                self.activations.setdefault("final_norm.output", value.detach().cpu().contiguous().clone())
            for suffix, tensor in (("input", state), ("output", value)):
                if isinstance(tensor, torch.Tensor) and tensor.requires_grad:
                    key = name + "." + suffix

                    def capture(grad, key=key):
                        assert key not in self.values, f"Repeated backward capture: {key}"
                        self.values[key] = grad.detach().cpu().contiguous().clone()

                    tensor.register_hook(capture)

        self.handles.append(module.register_forward_hook(observe, with_kwargs=True))

    def close(self):
        for handle in self.handles:
            handle.remove()
        if dist.get_rank() == 0:
            assert "final_norm.output" in self.values, "No loss backward observed"
            self.target.mkdir(parents=True, exist_ok=False)
            assert all(torch.isfinite(v).all() for v in self.values.values())
            save_file(self.values, str(self.target / "boundaries.safetensors"))
            save_file(self.activations, str(self.target / "activations.safetensors"))
