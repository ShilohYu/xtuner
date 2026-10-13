"""Isolate layer1 routing sensitivity and routed-expert activation on recorded inputs."""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace
import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import load_file
from torch.distributed.pipelining import microbatch
from torch.nn.attention.flex_attention import BlockMask
microbatch.BlockMask = BlockMask
from nemo_automodel.components.models.glm5_next.model import build_glm5_next_moe_config
from nemo_automodel.components.moe.layers import Gate
from nemo_automodel.components.moe.experts import swiglu_clamped_deepep
from transformers.models.glm5_next.modeling_glm5_next import Glm5NextTextTopkRouter, Glm5NextTextExperts
from xtuner.v1.module.decoder_layer.moe_decoder_layer import MoEGate
from xtuner.v1.module.router.noaux_router import NoAuxRouterConfig
from xtuner.v1.ops.act_fn import native_clamped_swiglu
from experiments.train_infra.plot_forward import errors

ROOT = Path('/mnt/shared-storage-user/yushan/GLM-5.3-flash-integration')
CKPT = Path('/tmp/GLM-5.3-Flash-4layer')
OUT = Path('/tmp/glm53_layer1_router_probe_20261013')

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--route-replay", action="store_true")
    args = parser.parse_args()
    out = OUT.with_name(OUT.name + "_replay") if args.route_replay else OUT
    out.mkdir(exist_ok=False)
    cfg = json.loads((CKPT / 'config.json').read_text())['text_config']
    cfg['num_local_experts'] = cfg['n_routed_experts']
    cfg = SimpleNamespace(**cfg)
    mapping = json.loads((CKPT / 'model.safetensors.index.json').read_text())['weight_map']
    def weight(name):
        key = 'model.language_model.layers.1.mlp.' + name
        with safe_open(str(CKPT / mapping[key]), framework='pt') as f:
            return f.get_tensor(key).cuda()
    xt = MoEGate(hidden_size=cfg.hidden_size, n_routed_experts=cfg.n_routed_experts,
        num_experts_per_tok=cfg.num_experts_per_tok, router_config=NoAuxRouterConfig(
            scoring_func='sigmoid', router_scaling_factor=cfg.routed_scaling_factor,
            norm_topk_prob=True, n_group=1, topk_group=1)).cuda().bfloat16()
    am = Gate(build_glm5_next_moe_config(cfg, torch.bfloat16), gate_precision=torch.float32).cuda()
    hf = Glm5NextTextTopkRouter(cfg).cuda().bfloat16()
    w, bias = weight('gate.weight').bfloat16(), weight('gate.e_score_correction_bias').float()
    for gate in (xt, am, hf):
        gate.weight.copy_(w)
    xt.router.e_score_correction_bias = bias.clone()
    am.e_score_correction_bias = bias.clone()
    hf.e_score_correction_bias = bias.clone()
    valid = ~load_file(str(ROOT / 'validation/train_infra/forward_alignment_20261013/pack_rank0_step0.safetensors'))['padding_mask'][0].cuda().bool()
    report = {'scope': 'Layer1 saved full 8K inputs; common BF16 checkpoint gate weights and FP32 bias; no training, native router calls.'}
    report['head'] = subprocess.check_output(['git', '-C', str(Path(__file__).resolve().parents[2]), 'rev-parse', 'HEAD'], text=True).strip()
    report['script_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    def compare(name, a, b):
        assert torch.isfinite(a).all() and torch.isfinite(b).all(), name
        report[name] = errors(a, b)
        print(name, report[name], flush=True)
    def canonical(ids, weights):
        order = ids.argsort(-1)
        return ids.gather(-1, order), weights.gather(-1, order)
    for label, source in [('native', '/tmp/glm53_xtuner_hf_norm_vs_native_automodel_forward_20261013'), ('shared', '/tmp/glm53_xtuner_automodel_shared_mlp_v2_forward_20261013')]:
        xs = []
        for side in ('xtuner', 'automodel'):
            with safe_open(str(Path(source) / side / 'hidden_states.safetensors'), framework='pt') as f:
                xs.append(f.get_tensor('layer1.post_attention_layernorm').cuda())
        x, y = xs
        compare(label + '.input', x[:, valid], y[:, valid])
        a = xt(x)
        bw, bi, _ = am(x.flatten(0, 1), valid, None)
        _, hw, hi = hf(x)
        ai, aw = canonical(a['topk_ids'], a['topk_weights'])
        bi, bw = canonical(bi, bw)
        hi, hw = canonical(hi, hw)
        assert torch.equal(ai, bi) and torch.equal(ai, hi)
        torch.testing.assert_close(aw, bw, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(aw, hw, atol=1e-6, rtol=1e-6)
        compare(label + '.same_input_XT_AM_weights', aw[valid], bw[valid])
        compare(label + '.same_input_XT_HF_weights', aw[valid], hw[valid])
        cy = xt(y)
        ci, cw = canonical(cy['topk_ids'], cy['topk_weights'])
        changed = (ai != ci).any(-1)
        report[label + '.upstream_changed_expert_sets'] = {'real_tokens': int(valid.sum()), 'changed_tokens': int(changed[valid].sum()), 'fraction': float(changed[valid].float().mean())}
        print(label, report[label + '.upstream_changed_expert_sets'], flush=True)
        # Actual routed expert on the most-used expert's real-token subset.
        expert = int(torch.bincount(ai[valid].flatten(), minlength=cfg.n_routed_experts).argmax())
        selected = valid & (ai == expert).any(-1)
        inp = x[0, selected]
        gu = torch.cat([weight(f'experts.{expert}.gate_proj.weight'), weight(f'experts.{expert}.up_proj.weight')]).bfloat16()
        down = weight(f'experts.{expert}.down_proj.weight').bfloat16()
        projected = F.linear(inp, gu)
        xa = native_clamped_swiglu(projected, limit=cfg.swiglu_limit)
        aa = swiglu_clamped_deepep(projected, torch.ones((len(inp), 1), device='cuda'), cfg.swiglu_limit)
        ha = Glm5NextTextExperts._apply_gate(cfg, projected)
        assert torch.equal(xa, ha)
        compare(label + '.expert_activation_XT_AM', xa, aa)
        compare(label + '.expert_output_XT_AM', F.linear(xa, down), F.linear(aa, down))
        report[label + '.expert'] = {'index': expert, 'real_tokens': len(inp), 'XT_activation_equals_HF': True}
        if args.route_replay:
            # Common mathematical replay, NOT a replacement for either runtime's grouped GEMM/combine.
            scores_y = F.linear(y[0].float(), w.float()).sigmoid()
            fixed_w = scores_y.gather(1, ai)
            fixed_w = fixed_w / (fixed_w.sum(-1, keepdim=True) + 1e-20) * cfg.routed_scaling_factor
            cases = [(x[0], ai, aw, False), (y[0], ci, cw, False),
                     (y[0], ai, fixed_w, False), (x[0], ai, aw, True)]
            outputs = [torch.zeros_like(x[0], dtype=torch.float32) for _ in cases]
            for expert in range(cfg.n_routed_experts):
                gu = torch.cat([weight(f'experts.{expert}.gate_proj.weight'), weight(f'experts.{expert}.up_proj.weight')]).bfloat16()
                down = weight(f'experts.{expert}.down_proj.weight').bfloat16()
                for dst, (inp, ids, probs, am_activation) in zip(outputs, cases):
                    rows, slots = torch.where((ids == expert) & valid[:, None])
                    if not rows.numel():
                        continue
                    projected = F.linear(inp[rows], gu)
                    act = (swiglu_clamped_deepep(projected, torch.ones((len(rows), 1), device='cuda'), cfg.swiglu_limit)
                           if am_activation else native_clamped_swiglu(projected, limit=cfg.swiglu_limit))
                    result = F.linear(act, down).float() * probs[rows, slots, None]
                    dst.index_add_(0, rows, result)
                if expert % 48 == 0:
                    print(label, 'expert', expert, flush=True)
            for name, result in zip(('own_routes', 'fixed_XT_expert_sets', 'same_input_AM_activation'), outputs[1:]):
                compare(label + '.common_expert_replay.' + name, outputs[0][valid], result[valid])
            report[label + '.replay_scope'] = 'All 288 routed experts, all 7384 real tokens, BF16 linear/activation except named AM activation; FP32 weighting/accumulation, shared expert and mHC excluded. Fixed sets recompute unbiased weights from y.'
        (out / 'comparison.json').write_text(json.dumps(report, indent=2))
    (out / 'comparison.json').write_text(json.dumps(report, indent=2))

if __name__ == '__main__':
    with torch.no_grad():
        main()
