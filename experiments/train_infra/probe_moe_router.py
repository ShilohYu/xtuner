"""Isolate layer1 routing sensitivity and routed-expert activation on recorded inputs."""
# ruff: noqa: E402

import argparse
import hashlib
import json
import os
import subprocess
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from typing import Callable
from unittest.mock import patch

import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import load_file
from torch.distributed.pipelining import microbatch
from torch.nn.attention.flex_attention import BlockMask


microbatch.BlockMask = BlockMask
from nemo_automodel.components.models.glm5_next.model import build_glm5_next_moe_config
from nemo_automodel.components.moe.experts import swiglu_clamped_deepep
from nemo_automodel.components.moe.layers import Gate

from experiments.train_infra.plot_forward import errors
from experiments.train_infra.precision_profile import (
    _combine_native_fp32,
    _routed_activation,
    automodel_moe_reduction_precision,
)
from transformers.models.glm5_next.modeling_glm5_next import Glm5NextTextExperts, Glm5NextTextTopkRouter
from xtuner.v1.module.decoder_layer.moe_decoder_layer import MoEGate
from xtuner.v1.module.router.noaux_router import NoAuxRouterConfig
from xtuner.v1.ops.act_fn import native_clamped_swiglu


ROOT = Path("/mnt/shared-storage-user/yushan/GLM-5.3-flash-integration")
CKPT = Path("/tmp/GLM-5.3-Flash-4layer")
OUT = Path("/tmp/glm53_layer1_router_probe_20261013")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--route-replay", action="store_true")
    parser.add_argument("--native-kernels", action="store_true", help="Compare real-shape grouped GEMMs and combine")
    parser.add_argument("--source", type=Path, default=Path("/tmp/glm53_native_core_precision_forward_valid_20261013"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    out = args.out
    out.mkdir(exist_ok=False)
    cfg = json.loads((CKPT / "config.json").read_text())["text_config"]
    cfg["num_local_experts"] = cfg["n_routed_experts"]
    cfg = SimpleNamespace(**cfg)
    mapping = json.loads((CKPT / "model.safetensors.index.json").read_text())["weight_map"]

    def weight(name):
        key = "model.language_model.layers.1.mlp." + name
        with safe_open(str(CKPT / mapping[key]), framework="pt") as f:
            return f.get_tensor(key).cuda()

    xt = (
        MoEGate(
            hidden_size=cfg.hidden_size,
            n_routed_experts=cfg.n_routed_experts,
            num_experts_per_tok=cfg.num_experts_per_tok,
            router_config=NoAuxRouterConfig(
                scoring_func="sigmoid",
                router_scaling_factor=cfg.routed_scaling_factor,
                norm_topk_prob=True,
                n_group=1,
                topk_group=1,
            ),
        )
        .cuda()
        .bfloat16()
    )
    am = Gate(build_glm5_next_moe_config(cfg, torch.bfloat16), gate_precision=torch.float32).cuda()
    hf = Glm5NextTextTopkRouter(cfg).cuda().bfloat16()
    w, bias = weight("gate.weight").bfloat16(), weight("gate.e_score_correction_bias").float()
    for gate in (xt, am, hf):
        gate.weight.copy_(w)
    xt.router.e_score_correction_bias = bias.clone()
    am.e_score_correction_bias = bias.clone()
    hf.e_score_correction_bias = bias.clone()
    valid = (
        ~load_file(str(ROOT / "validation/train_infra/forward_alignment_20261013/pack_rank0_step0.safetensors"))[
            "padding_mask"
        ][0]
        .cuda()
        .bool()
    )
    report = {
        "scope": "Layer1 saved full 8K inputs; common BF16 checkpoint gate weights and FP32 bias; no training, native router calls."
    }
    report["source_dump"] = str(args.source)
    report["router_groups"] = {"n_group": cfg.n_group, "topk_group": cfg.topk_group}
    report["head"] = subprocess.check_output(
        ["git", "-C", str(Path(__file__).resolve().parents[2]), "rev-parse", "HEAD"], text=True
    ).strip()
    report["script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    report["permute_compute_dtype_env"] = os.environ.get("PERMUTE_COMPUTE_DTYPE")

    def compare(name, a, b):
        assert torch.isfinite(a).all() and torch.isfinite(b).all(), name
        report[name] = errors(a, b)
        print(name, report[name], flush=True)

    def canonical(ids, weights):
        order = ids.argsort(-1)
        return ids.gather(-1, order), weights.gather(-1, order)

    for label, source in [("aligned_layer0", args.source)]:
        xs = []
        for side in ("xtuner", "automodel"):
            with safe_open(str(Path(source) / side / "hidden_states.safetensors"), framework="pt") as f:
                xs.append(f.get_tensor("layer1.post_attention_layernorm").cuda())
        x, y = xs
        compare(label + ".input", x[:, valid], y[:, valid])
        a = xt(x)
        bw, bi, _ = am(x.flatten(0, 1), valid, None)
        raw_am_ids = bi.clone()
        native_topk = torch.topk

        def ordered_topk(input: torch.Tensor, k: int, *args: object, **kwargs: object) -> object:
            if input.shape == (x.shape[1], cfg.n_routed_experts) and k == cfg.num_experts_per_tok:
                kwargs["sorted"] = True
            return native_topk(input, k, *args, **kwargs)

        with patch.object(torch, "topk", ordered_topk):
            ordered_weights, ordered_ids, _ = am(x.flatten(0, 1), valid, None)
        assert torch.equal(ordered_ids, a["topk_ids"])
        if os.environ.get("PERMUTE_COMPUTE_DTYPE") == "fp32":
            with automodel_moe_reduction_precision(am):
                pw, pi, _ = am(x.flatten(0, 1), valid, None)
            assert torch.equal(pw, ordered_weights) and torch.equal(pi, ordered_ids)
            assert torch.equal(am(x.flatten(0, 1), valid, None)[1], raw_am_ids)
        compare(label + ".AM_sorted_topk_weights", a["topk_weights"][valid], ordered_weights[valid])
        report[label + ".native_route_order_equal"] = torch.equal(raw_am_ids[valid], a["topk_ids"][valid])
        _, hw, hi = hf(x)
        ai, aw = canonical(a["topk_ids"], a["topk_weights"])
        bi, bw = canonical(bi, bw)
        hi, hw = canonical(hi, hw)
        assert torch.equal(ai, bi) and torch.equal(ai, hi)
        torch.testing.assert_close(aw, bw, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(aw, hw, atol=1e-6, rtol=1e-6)
        compare(label + ".same_input_XT_AM_weights", aw[valid], bw[valid])
        compare(label + ".same_input_XT_HF_weights", aw[valid], hw[valid])
        cy = xt(y)
        ci, cw = canonical(cy["topk_ids"], cy["topk_weights"])
        changed = (ai != ci).any(-1)
        report[label + ".upstream_changed_expert_sets"] = {
            "real_tokens": int(valid.sum()),
            "changed_tokens": int(changed[valid].sum()),
            "fraction": float(changed[valid].float().mean()),
        }
        print(label, report[label + ".upstream_changed_expert_sets"], flush=True)
        # Actual routed expert on the most-used expert's real-token subset.
        expert = int(torch.bincount(ai[valid].flatten(), minlength=cfg.n_routed_experts).argmax())
        selected = valid & (ai == expert).any(-1)
        inp = x[0, selected]
        gu = torch.cat(
            [weight(f"experts.{expert}.gate_proj.weight"), weight(f"experts.{expert}.up_proj.weight")]
        ).bfloat16()
        down = weight(f"experts.{expert}.down_proj.weight").bfloat16()
        projected = F.linear(inp, gu)
        xa = native_clamped_swiglu(projected, limit=cfg.swiglu_limit)
        aa = swiglu_clamped_deepep(projected, torch.ones((len(inp), 1), device="cuda"), cfg.swiglu_limit)
        ha = Glm5NextTextExperts._apply_gate(cfg, projected)
        assert torch.equal(xa, ha)
        assert torch.equal(
            xa, _routed_activation(projected, torch.ones((len(inp), 1), device="cuda"), limit=cfg.swiglu_limit)
        )
        compare(label + ".expert_activation_XT_AM", xa, aa)
        compare(label + ".expert_output_XT_AM", F.linear(xa, down), F.linear(aa, down))
        report[label + ".expert"] = {"index": expert, "real_tokens": len(inp), "XT_activation_equals_HF": True}
        if args.native_kernels:
            report["native_kernels"] = _native_kernel_comparison(
                cfg, x[0], a["topk_ids"], a["topk_weights"], valid, weight, compare
            )
        if args.route_replay:
            # Common mathematical replay, NOT a replacement for either runtime's grouped GEMM/combine.
            scores_y = F.linear(y[0].float(), w.float()).sigmoid()
            fixed_w = scores_y.gather(1, ai)
            fixed_w = fixed_w / (fixed_w.sum(-1, keepdim=True) + 1e-20) * cfg.routed_scaling_factor
            cases = [(x[0], ai, aw, False), (y[0], ci, cw, False), (y[0], ai, fixed_w, False), (x[0], ai, aw, True)]
            outputs = [torch.zeros_like(x[0], dtype=torch.float32) for _ in cases]
            for expert in range(cfg.n_routed_experts):
                gu = torch.cat(
                    [weight(f"experts.{expert}.gate_proj.weight"), weight(f"experts.{expert}.up_proj.weight")]
                ).bfloat16()
                down = weight(f"experts.{expert}.down_proj.weight").bfloat16()
                for dst, (inp, ids, probs, am_activation) in zip(outputs, cases):
                    rows, slots = torch.where((ids == expert) & valid[:, None])
                    if not rows.numel():
                        continue
                    projected = F.linear(inp[rows], gu)
                    act = (
                        swiglu_clamped_deepep(projected, torch.ones((len(rows), 1), device="cuda"), cfg.swiglu_limit)
                        if am_activation
                        else native_clamped_swiglu(projected, limit=cfg.swiglu_limit)
                    )
                    result = F.linear(act, down).float() * probs[rows, slots, None]
                    dst.index_add_(0, rows, result)
                if expert % 48 == 0:
                    print(label, "expert", expert, flush=True)
            for name, result in zip(("own_routes", "fixed_XT_expert_sets", "same_input_AM_activation"), outputs[1:]):
                compare(label + ".common_expert_replay." + name, outputs[0][valid], result[valid])
            report[label + ".replay_scope"] = (
                "All 288 routed experts, all 7384 real tokens, BF16 linear/activation except named AM activation; FP32 weighting/accumulation, shared expert and mHC excluded. Fixed sets recompute unbiased weights from y."
            )
        (out / "comparison.json").write_text(json.dumps(report, indent=2))
    report["complete"] = True
    (out / "comparison.json").write_text(json.dumps(report, indent=2))
    print("COMPLETE", out, flush=True)


def _native_kernel_comparison(
    cfg: SimpleNamespace,
    states: torch.Tensor,
    ids: torch.Tensor,
    probs: torch.Tensor,
    valid: torch.Tensor,
    weight: Callable[[str], torch.Tensor],
    compare: Callable[[str, torch.Tensor, torch.Tensor], None],
) -> dict:
    from nemo_automodel.components.moe.experts import _permute_tokens_for_grouped_mm

    from xtuner.v1.ops import group_gemm, permute, unpermute

    counts = torch.bincount(ids.flatten(), minlength=cfg.n_routed_experts)
    offsets = counts.cumsum(0).to(torch.int32)
    packed, row_map = permute(states, ids.to(torch.int32))
    # Decode the native permutation with exact FP32 row tags; don't assume the extension's map layout.
    tags = torch.arange(len(states), device=states.device, dtype=torch.float32)[:, None].expand(-1, 64).contiguous()
    packed_tags, tag_map = permute(tags, ids.to(torch.int32))
    rows = packed_tags[:, 0].long()
    assert torch.equal(row_map, tag_map) and torch.equal(packed, states[rows])
    expert_ids = torch.arange(cfg.n_routed_experts, device=states.device).repeat_interleave(counts)
    matches = ids[rows] == expert_ids[:, None]
    assert bool((matches.sum(-1) == 1).all())
    slots = rows * ids.shape[1] + matches.long().argmax(-1)
    assert slots.unique().numel() == ids.numel()
    am_rows, am_slots, _, am_counts, _ = _permute_tokens_for_grouped_mm(
        ids, probs, torch.ones_like(valid), cfg.n_routed_experts, 0, return_slot_ids=True
    )
    assert torch.equal(counts, am_counts)
    report = {
        "scope": "Fixed XT full-pack input/routes/probs including padding; all 288 checkpoint experts; native primitives, not whole-layer replay",
        "xt_gemm": group_gemm.__module__ + "." + group_gemm.__name__,
        "xt_permute": permute.__module__ + "." + permute.__name__,
        "native_permuted_rows_equal_AM": torch.equal(rows, am_rows),
        "native_permuted_slots_equal_AM": torch.equal(slots, am_slots),
    }
    print("native_kernels", report, flush=True)
    gu = torch.empty(
        cfg.n_routed_experts,
        2 * cfg.moe_intermediate_size,
        cfg.hidden_size,
        device=states.device,
        dtype=states.dtype,
    )
    down = torch.empty(
        cfg.n_routed_experts,
        cfg.hidden_size,
        cfg.moe_intermediate_size,
        device=states.device,
        dtype=states.dtype,
    )
    for expert in range(cfg.n_routed_experts):
        gu[expert].copy_(
            torch.cat([weight(f"experts.{expert}.gate_proj.weight"), weight(f"experts.{expert}.up_proj.weight")])
        )
        down[expert].copy_(weight(f"experts.{expert}.down_proj.weight"))
    real = valid[rows]
    xt_up = group_gemm(packed, gu, counts)
    am_up = torch._grouped_mm(packed, gu.transpose(1, 2), offsets)
    compare("native_kernels.same_input.gate_up", xt_up[real], am_up[real])
    activation = native_clamped_swiglu(xt_up, limit=cfg.swiglu_limit)
    xt_down = group_gemm(activation, down, counts)
    am_down = torch._grouped_mm(activation, down.transpose(1, 2), offsets)
    compare("native_kernels.same_input.down", xt_down[real], am_down[real])

    def torch_combine(values: torch.Tensor) -> torch.Tensor:
        weighted = values.float() * probs.flatten()[slots, None]
        restored = torch.empty((ids.numel(), states.shape[-1]), dtype=torch.float32, device=states.device)
        restored[slots] = weighted
        return restored.view(len(states), ids.shape[1], -1).sum(1).to(states.dtype)

    xt_combined = unpermute(xt_down, row_map, probs=probs)
    if os.environ.get("PERMUTE_COMPUTE_DTYPE") == "fp32":
        assert torch.equal(xt_combined, _combine_native_fp32(xt_down, slots, probs))
        real_slots = slots[real]
        masked = _combine_native_fp32(xt_down[real], real_slots, probs)
        assert torch.equal(masked[valid], xt_combined[valid])
    compare("native_kernels.same_input.combine", xt_combined[valid], torch_combine(xt_down)[valid])
    terms = torch.empty((ids.numel(), states.shape[-1]), device=states.device, dtype=states.dtype)
    terms[slots] = xt_down
    terms = terms.view(len(states), ids.shape[1], -1)
    weighted = terms.float() * probs.unsqueeze(-1)
    bf16_terms = weighted.to(states.dtype)
    compare(
        "combine_rounding.bf16_products_fp32_sum",
        xt_combined[valid],
        bf16_terms.float().sum(1).to(states.dtype)[valid],
    )
    for name, products, dtype in (
        ("fp32_products_fp32_scan", weighted, torch.float32),
        ("bf16_products_bf16_scan", bf16_terms, states.dtype),
        ("bf16_probs_products_bf16_scan", terms * probs.to(states.dtype).unsqueeze(-1), states.dtype),
    ):
        total = torch.zeros_like(states, dtype=dtype)
        for slot in range(ids.shape[1]):
            total = total + products[:, slot]
        compare("combine_rounding." + name, xt_combined[valid], total.to(states.dtype)[valid])
    promoted = unpermute(xt_down.float(), row_map, probs=probs).to(states.dtype)
    compare("combine_rounding.promoted_vs_AM", promoted[valid], torch_combine(xt_down)[valid])
    fma_total = weighted[:, 0].clone()
    for slot in range(1, ids.shape[1]):
        fma_total = torch.addcmul(fma_total, terms[:, slot].float(), probs[:, slot, None])
    compare("combine_rounding.promoted_vs_fp32_fma_scan", promoted[valid], fma_total.to(states.dtype)[valid])
    del terms, weighted, bf16_terms, products, total, promoted, fma_total
    cumulative_down = torch._grouped_mm(
        native_clamped_swiglu(am_up, limit=cfg.swiglu_limit), down.transpose(1, 2), offsets
    )
    compare("native_kernels.cumulative_GEMMs_and_combine", xt_combined[valid], torch_combine(cumulative_down)[valid])
    del am_up, am_down, xt_up, activation, cumulative_down
    # Padding changes GEMM row geometry even when real token values/routes remain fixed.
    masked_counts = torch.bincount(expert_ids[real], minlength=cfg.n_routed_experts)
    masked_offsets = masked_counts.cumsum(0).to(torch.int32)
    masked_up = torch._grouped_mm(packed[real], gu.transpose(1, 2), masked_offsets)
    masked_down = torch._grouped_mm(
        native_clamped_swiglu(masked_up, limit=cfg.swiglu_limit), down.transpose(1, 2), masked_offsets
    )
    full_masked = torch.zeros_like(xt_down)
    full_masked[real] = masked_down
    compare("native_kernels.AM_mask_padding", xt_combined[valid], torch_combine(full_masked)[valid])
    if os.environ.get("PERMUTE_COMPUTE_DTYPE") == "fp32":
        from nemo_automodel.components.models.common import BackendConfig
        from nemo_automodel.components.moe.experts import GroupedExperts

        with torch.device("meta"):
            experts = GroupedExperts(build_glm5_next_moe_config(cfg, states.dtype), BackendConfig(experts="torch_mm"))
        experts.gate_and_up_projs = torch.nn.Parameter(gu.transpose(1, 2), requires_grad=False)
        experts.down_projs = torch.nn.Parameter(down.transpose(1, 2), requires_grad=False)
        experts.expert_activation_grouped = partial(_routed_activation, limit=cfg.swiglu_limit)
        original = experts(states, valid, probs, ids)
        compare("native_kernels.native_AM_module_replay", torch_combine(full_masked)[valid], original[valid])
        assert torch.equal(torch_combine(full_masked)[valid], original[valid])
        with automodel_moe_reduction_precision(experts):
            aligned = experts(states, valid, probs, ids)
        compare("native_kernels.reusable_AM_combine_profile", xt_combined[valid], aligned[valid])
        assert torch.equal(xt_combined[valid], aligned[valid])
        assert torch.equal(experts(states, valid, probs, ids), original)
        report["native_AM_module_profile_restored"] = True
    return report


if __name__ == "__main__":
    with torch.no_grad():
        main()
