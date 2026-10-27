"""Use XT Muon for AM's EP1 experiment, preserving logical expert matrices."""
from types import MethodType, SimpleNamespace

import torch
from torch.distributed.tensor import DTensor

from xtuner.v1.config import MuonConfig
from xtuner.v1.engine.train_engine import TrainEngine
from xtuner.v1.utils.grad_norm import cal_grad_norm


def expert_to_xt(value):
    # AM [E, in, out] -> XT [E*out, in], including the separate gate/up rows.
    return value.transpose(-1, -2).contiguous().flatten(0, 1)


def expert_to_am(value, shape):
    return value.reshape(shape[0], shape[2], shape[1]).transpose(-1, -2).contiguous()


class _ParameterView:
    """Expose canonical expert names/layouts to the existing XT group builder."""
    def __init__(self, model, parameters, num_experts):
        self.model = model
        self._parameters = parameters
        self.config = SimpleNamespace(n_routed_experts=num_experts)

    def named_parameters(self):
        return iter(self._parameters)

    def trainable_parameters(self):
        return [(name, p) for name, p in self._parameters if p.requires_grad]

    def modules(self):
        return self.model.modules()


@torch.no_grad()
def build_muon(model, settings):
    """Keep XT defaults/grouping; only adapt AM's transposed fused expert storage.

    The canonical expert copies belong to the optimizer, never to model forward.
    This experiment is EP1/SP1 only and does not implement optimizer checkpoint export.
    """
    config = MuonConfig(**settings)
    n_experts = model.config.text_config.n_routed_experts
    parameters, pairs, originals, names = [], [], {}, {}
    for name, p in model.named_parameters():
        clean = name.replace('_checkpoint_wrapped_module.', '').replace('_orig_mod.', '')
        canonical = p
        if p.requires_grad and clean.endswith(('experts.gate_and_up_projs', 'experts.down_projs')):
            assert isinstance(p, DTensor) and p.ndim == 3 and p.shape[0] == n_experts
            assert len(p.placements) == 1 and p.placements[0].is_shard(0), p.placements
            assert p.shape[0] % p.device_mesh.size() == 0
            canonical = torch.nn.Parameter(expert_to_xt(p.detach()), requires_grad=True)
            pairs.append((p, canonical))
            clean = clean.replace('gate_and_up_projs', 'fused_w1w3.weight').replace('down_projs', 'fused_w2.weight')
        parameters.append((clean, canonical))
        originals[id(canonical)] = p
        names[id(canonical)] = name
    optimizer = config.build(_ParameterView(model, parameters, n_experts))

    @torch.no_grad()
    def before_step(opt, args, kwargs):
        for native, canonical in pairs:
            canonical.grad = expert_to_xt(native.grad) if native.grad is not None else None

    @torch.no_grad()
    def after_step(opt, args, kwargs):
        for native, canonical in pairs:
            native.copy_(expert_to_am(canonical, native.shape))

    original_zero_grad = optimizer.zero_grad

    def zero_grad(self, set_to_none=True):
        original_zero_grad(set_to_none=set_to_none)
        for native, _ in pairs:
            if set_to_none:
                native.grad = None
            elif native.grad is not None:
                native.grad.zero_()

    optimizer.register_step_pre_hook(before_step)
    optimizer.register_step_post_hook(after_step)
    optimizer.zero_grad = MethodType(zero_grad, optimizer)
    optimizer._alignment_originals = originals
    optimizer._alignment_names = names
    return optimizer


@torch.no_grad()
def clip_muon_gradients(optimizer, model, max_norm):
    """Report all-parameter norm, clip only groups selected by XT MuonConfig."""
    params = [p for p in model.parameters() if p.requires_grad]
    assert not any(getattr(p, '_nemo_model_owned_grad_divisor', None) for p in params)
    # EP1/SP1 backward already reduced these DTensors; no EP/PP scaling is needed.
    grads = [p.grad for p in params if p.grad is not None]
    norm, grouped = cal_grad_norm(grads, dtype=torch.float32)
    selected = [optimizer._alignment_originals[id(p)] for group in optimizer.param_groups
                if group.get('clip_grad', True) for p in group['params']]
    selected_grads = [p.grad for p in selected if p.grad is not None]
    TrainEngine._clip_gradients(selected_grads, max_norm=max_norm, dtype=torch.float32,
                               precomputed_grad_info=(norm, grouped) if len(selected) == len(params) else None)
    return norm


def optimizer_manifest(optimizer):
    return [{'algorithm': g['algorithm'], 'lr': g['lr'], 'weight_decay': g['weight_decay'],
             'num_experts': g['num_experts'], 'clip_grad': g['clip_grad'],
             'parameters': [{'name': optimizer._alignment_names[id(p)], 'shape': list(p.shape)} for p in g['params']]}
            for g in optimizer.param_groups]
