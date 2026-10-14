"""Replay native layer0, locate first divergence, and ablate one precision boundary at a time."""
# ruff: noqa: E402

import argparse
import inspect
import json
import subprocess
from contextlib import ExitStack
from functools import partial
from pathlib import Path
from types import MethodType
from unittest.mock import patch

import torch
from safetensors import safe_open
from safetensors.torch import load_file
from torch.distributed.pipelining import microbatch
from torch.nn.attention.flex_attention import BlockMask


# AutoModel needs this PyTorch 2.9 alias before its imports.
microbatch.BlockMask = BlockMask
from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.glm5_next import layers as am_layers
from nemo_automodel.components.models.glm5_next.config import Glm5NextTextConfig
from nemo_automodel.components.models.glm5_next.model import _packed_context_from_inputs, build_glm5_next_moe_config

from experiments.train_infra.text_attention import install_xtuner_hf_rms_norm
from transformers.models.glm5_next.modeling_glm5_next import (
    Glm5NextTextHyperConnection,
)
from xtuner.v1.data_proto import SequenceContext
from xtuner.v1.model.moe.glm53 import Glm53TextMoEConfig
from xtuner.v1.model.moe.glm53 import decoder_layer as xt_layers


ROOT = Path("/mnt/shared-storage-user/yushan/GLM-5.3-flash-integration")
SOURCE = Path("/tmp/glm53_forward_resumed_20261013")
OUT = Path("/tmp/glm53_layer0_modules_resumed_20261013")


def main():
    global SOURCE, OUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--checkpoint", type=Path, default=Path("/tmp/GLM-5.3-Flash-4layer"))
    parser.add_argument(
        "--profile-backward",
        action="store_true",
        help="Compare full layer0 VJPs under the cumulative precision profile",
    )
    parser.add_argument("--backward-seed", type=int, default=20261013)
    parser.add_argument("--xt-native-profile", action="store_true")
    parser.add_argument(
        "--fixture",
        type=Path,
        default=ROOT / "validation/train_infra/forward_alignment_20261013/pack_rank0_step0.safetensors",
    )
    args = parser.parse_args()
    SOURCE, OUT, checkpoint = args.source, args.out, args.checkpoint
    for path in [
        checkpoint / "config.json",
        args.fixture,
        SOURCE / "xtuner/hidden_states.safetensors",
        SOURCE / "automodel/hidden_states.safetensors",
    ]:
        if not path.is_file():
            raise FileNotFoundError(path)
    OUT.mkdir(exist_ok=False)
    config = json.loads((checkpoint / "config.json").read_text())["text_config"]
    cfg = Glm5NextTextConfig(**config)
    xcfg = Glm53TextMoEConfig.from_hf(checkpoint)
    mapping = json.loads((checkpoint / "model.safetensors.index.json").read_text())["weight_map"]

    def weight(name):
        key = "model.language_model.layers.0." + name
        with safe_open(str(checkpoint / mapping[key]), framework="pt") as f:
            return f.get_tensor(key).cuda()

    def capture(side, name):
        with safe_open(str(SOURCE / side / "hidden_states.safetensors"), framework="pt") as f:
            return f.get_tensor("layer0" + name).cuda()

    xt = (
        xt_layers.Glm53DenseDecoderLayer(
            hidden_size=xcfg.hidden_size,
            intermediate_size=xcfg.intermediate_size,
            hidden_act=xcfg.hidden_act,
            swiglu_limit=xcfg.swiglu_limit,
            rms_norm_eps=xcfg.rms_norm_eps,
            attention_config=xcfg.linear_attention,
            layer_idx=0,
            mhc_cfg=xcfg.mhc,
        )
        .cuda()
        .bfloat16()
    )
    am = am_layers.Glm5NextDecoderLayer(
        cfg, 0, build_glm5_next_moe_config(cfg, torch.bfloat16), BackendConfig(attn="cudnn", linear="torch")
    ).cuda()
    for side, model in [("xtuner", xt), ("automodel", am)]:
        for name, p in model.named_parameters():
            key = name.replace("self_attn._fp32_params.", "self_attn.")
            for site in ("attn", "ffn"):
                key = key.replace(site + "_hc._fp32_params.", "hc_" + site + "_").replace(
                    site + "_hc.", "hc_" + site + "_"
                )
            p.data = weight(key).to(
                torch.float32
                if any(v in key for v in ["_base", "_scale", "A_log", "dt_bias", "conv1d.weight"])
                else torch.bfloat16
            )
        model.train()
    native_norms = {n: xt.get_submodule(n).forward for n in ("input_layernorm", "post_attention_layernorm")}
    install_xtuner_hf_rms_norm(xt)
    fixture = {k: v.cuda() for k, v in load_file(str(args.fixture)).items()}
    valid = ~fixture["padding_mask"][0].bool()
    context = SequenceContext(
        input_ids=fixture["input_ids"],
        cu_seq_lens_q=fixture["cu_seqlens"],
        cu_seq_lens_k=fixture["cu_seqlens"],
        device="cuda",
        max_length_q=int(fixture["cu_seqlens"].diff().max()),
        max_length_k=int(fixture["cu_seqlens"].diff().max()),
    )
    packed = _packed_context_from_inputs(
        fixture["input_ids"], None, fixture["padding_mask"], fixture["cu_seqlens"], None
    )
    x = capture("xtuner", ".input")
    assert torch.equal(x, capture("automodel", ".input"))
    import ast
    import hashlib
    import textwrap

    from safetensors.torch import save_file

    from experiments.train_infra.text_attention import _hf_gated_rms_norm
    from transformers.models.glm5_next import modeling_glm5_next as hf_code
    from transformers.models.glm5_next.configuration_glm5_next import Glm5NextTextConfig as HFConfig
    from xtuner.v1.module.attention import kda as xt_kda

    hf = hf_code.Glm5NextTextDecoderLayer(HFConfig(**config), 0).cuda().bfloat16()
    for name, p in hf.named_parameters():
        key = name.replace("self_attn.forget_gate.", "self_attn.")
        for site in ("attn", "ffn"):
            key = key.replace(site + "_hc.", "hc_" + site + "_")
        if key == "self_attn.conv1d.weight":
            w = torch.cat([weight("self_attn." + c + "_conv1d.weight") for c in "qkv"])
        else:
            w = weight(key)
        p.data = w.to(
            torch.float32
            if any(v in key for v in ["_base", "_scale", "A_log", "dt_bias", "conv1d.weight"])
            else torch.bfloat16
        )
    hf.train()
    cu = fixture["cu_seqlens"].flatten().tolist()
    report = {
        "head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "upstream_main": subprocess.check_output(["git", "rev-parse", "upstream/main"], text=True).strip(),
        "upstream_f6": subprocess.check_output(
            ["git", "rev-parse", "upstream/feat/glm53flash-f6-text-moe"], text=True
        ).strip(),
        "source_dump": str(SOURCE),
        "hf_file": inspect.getfile(hf_code),
        "am_file": inspect.getfile(am_layers),
        "cu_seqlens": cu,
        "real_tokens": int(valid.sum()),
        "scope": "layer0; shared checkpoint; train/no_grad; native kernels; SP1; XT ordinary norms use explicit HF adapter",
        "rows": [],
        "layer_interventions": {},
        "input_hashes": {},
        "weights_verified": True,
        "metric_reduction_dtype": "float64",
        "projection_blocks": {
            "kda_gate": xt.self_attn.gate_projection_block_size,
            "mhc": xt.mhc_cfg.projection_block_size,
        },
        "xt_baseline_norm": "HF adapter; native norms reported separately",
    }
    # Verify loaded weights, including the split vs concatenated conv mapping.
    for name, p in hf.named_parameters():
        key = name.replace("self_attn.forget_gate.", "self_attn.")
        for site in ("attn", "ffn"):
            key = key.replace(site + "_hc.", "hc_" + site + "_")
        expected = (
            torch.cat([weight("self_attn." + c + "_conv1d.weight") for c in "qkv"])
            if key == "self_attn.conv1d.weight"
            else weight(key)
        )
        assert torch.equal(p, expected.to(p.dtype)), name

    def metric(a, b):
        assert a.shape == b.shape, (a.shape, b.shape)
        same_dtype = a.dtype == b.dtype
        a, b = a[:, valid].float(), b[:, valid].float()
        assert torch.isfinite(a).all() and torch.isfinite(b).all()
        delta = b - a
        return {
            "relative_l2": float(delta.norm(dtype=torch.float64) / a.norm(dtype=torch.float64).clamp_min(1e-30)),
            "max_abs": float(delta.abs().max()),
            "equal_fraction": float((a == b).float().mean()),
            "bitwise_equal": same_dtype and torch.equal(a, b),
        }

    def compare(name, xtv, amv, hfv, anchor="XT"):
        r = {
            "module": name,
            "anchor": anchor,
            "shape": list(hfv.shape),
            "dtypes": [str(t.dtype) for t in [xtv, amv, hfv]],
            "XT_vs_HF": metric(hfv, xtv),
            "AM_vs_HF": metric(hfv, amv),
            "XT_vs_AM": metric(xtv, amv),
        }
        report["rows"].append(r)
        print(json.dumps(r), flush=True)
        (OUT / "comparison.json").write_text(json.dumps(report, indent=2))

    def persist(name, values):
        tensors = {k: v.detach().cpu().contiguous() for k, v in values.items()}
        save_file(tensors, str(OUT / (name + ".safetensors")))
        report["input_hashes"][name] = {
            k: hashlib.sha256(v.view(torch.uint8).numpy().tobytes()).hexdigest() for k, v in tensors.items()
        }

    # Hook real XT/AM paths on the same recorded layer input; capture kwargs too.
    records = {}
    handles = []
    modules = [
        "input_layernorm",
        "post_attention_layernorm",
        "self_attn",
        "mlp",
        "self_attn.q_proj",
        "self_attn.k_proj",
        "self_attn.v_proj",
        "self_attn.b_proj",
        "self_attn.q_conv1d",
        "self_attn.k_conv1d",
        "self_attn.v_conv1d",
        "self_attn.o_norm",
        "self_attn.o_proj",
        "mlp.gate_proj",
        "mlp.up_proj",
        "mlp.down_proj",
    ]

    def hook(side, name):
        def f(mod, args, kwargs, out):
            inp = args[0] if args else kwargs.get("hidden_states", kwargs.get("x"))
            value = out["projected_output"] if isinstance(out, dict) else out[0] if isinstance(out, tuple) else out
            records[side + "." + name] = {"input": inp.detach(), "output": value.detach()}
            if name == "self_attn.o_norm":
                records[side + "." + name]["gate"] = args[1].detach()

        return f

    for side, model in [("XT", xt), ("AM", am)]:
        for name in modules:
            handles.append(model.get_submodule(name).register_forward_hook(hook(side, name), with_kwargs=True))
    hc_records = []
    original_pre, original_post = xt_layers.hc_pre, xt_layers.hc_post

    def trace_pre(*args, **kwargs):
        out = original_pre(*args, **kwargs)
        hc_records.append({"input": args[0], "collapsed": out[0], "post": out[1], "comb": out[2]})
        return out

    def trace_post(update, residual, post, comb):
        hc_records[-1]["update"] = update
        out = original_post(update, residual, post, comb)
        hc_records[-1]["output"] = out
        return out

    xt_layers.hc_pre, xt_layers.hc_post = trace_pre, trace_post
    xo = xt._forward(x, position_embeddings=None, seq_ctx=context)
    xt_layers.hc_pre, xt_layers.hc_post = original_pre, original_post
    ao = am(x, packed_context=packed, padding_mask=fixture["padding_mask"])
    for h in handles:
        h.remove()
    persist("layer_input", {"input": x, "cu_seqlens": fixture["cu_seqlens"], "valid": valid})
    for name in modules:
        persist(name, {k: v for k, v in records["XT." + name].items() if k != "output"})
    for i, rec in enumerate(hc_records):
        persist("mhc_" + str(i), rec)
    report["standalone_vs_recorded_layer_output"] = {
        s: metric(capture(old, ""), v) for s, old, v in [("XT", "xtuner", xo), ("AM", "automodel", ao)]
    }
    # HF reference functions ignore cu_seqlens: split ONLY their sequence-dependent
    # conv/core calls by original document boundaries, leaving projections full-pack.
    from experiments.train_infra.hf_eager_kda import eager_function
    from experiments.train_infra.hf_eager_kda import packed_chunk as native_packed_chunk
    from experiments.train_infra.hf_eager_kda import packed_conv as native_packed_conv

    hf_conv = eager_function("causal_conv1d_fn")
    hf_chunk = eager_function("chunk_kimi_delta_attention")
    report["hf_eager_functions"] = {
        fn.__name__: {"file": fn.__code__.co_filename, "line": fn.__code__.co_firstlineno}
        for fn in (hf_conv, hf_chunk)
    }
    packed_conv = partial(native_packed_conv, cu=cu)
    packed_chunk = partial(native_packed_chunk, cu=cu)
    packed_conv.__wrapped__ = hf_conv
    packed_chunk.__wrapped__ = hf_chunk
    hf_code.causal_conv1d_fn = packed_conv
    hf_code.chunk_kimi_delta_attention = packed_chunk
    report["HF_execution"] = (
        "Actual Transformers eager functions; conv/core split at original documents; core evaluated in independent groups of 8 heads to bound temporary memory. No FLA substitution."
    )

    # mHC post expressions extracted from actual native decoder source, not a handwritten oracle.
    def post_expression(cls):
        tree = ast.parse(textwrap.dedent(inspect.getsource(cls.forward)))
        for node in ast.walk(tree):
            value = getattr(node, "value", None)
            if isinstance(value, ast.BinOp) and isinstance(value.op, ast.Add) and "post.to" in ast.unparse(value):
                code = compile(ast.Expression(value), inspect.getfile(cls), "eval")
                return lambda update, residual, post, comb: eval(
                    code,
                    {"torch": torch},
                    {
                        "dtype": update.dtype,
                        "update": update,
                        "hidden_states": update,
                        "residual": residual,
                        "post": post,
                        "comb": comb,
                    },
                )
        raise AssertionError("Native post expression not found")

    am_post = post_expression(am_layers.Glm5NextDecoderLayer)
    hf_post = post_expression(hf_code.Glm5NextTextDecoderLayer)
    for i, site in enumerate(["attn", "ffn"]):
        rec = hc_records[i]
        z = rec["input"]
        xp = original_pre(
            z,
            getattr(xt, "hc_" + site + "_fn"),
            getattr(xt, "hc_" + site + "_scale"),
            getattr(xt, "hc_" + site + "_base"),
            cfg.hc_mult,
            cfg.hc_sinkhorn_iters,
            cfg.hc_eps,
            norm_eps=cfg.rms_norm_eps,
            projection_block_size=xt.mhc_cfg.projection_block_size,
        )
        ap = getattr(am, site + "_hc")(z)
        hp = getattr(hf, site + "_hc")(z)
        for key, j, k in [("collapsed", 0, 2), ("post", 1, 0), ("comb", 2, 1)]:
            compare("mhc_" + site + "_pre." + key, xp[j], ap[k], hp[k])
        post_args = [rec[k] for k in ["update", "input", "post", "comb"]]
        compare("mhc_" + site + "_post", original_post(*post_args), am_post(*post_args), hf_post(*post_args))
    for name in [
        "input_layernorm",
        "post_attention_layernorm",
        "mlp",
        "mlp.gate_proj",
        "mlp.up_proj",
        "mlp.down_proj",
    ]:
        z = records["XT." + name]["input"]
        compare(name, xt.get_submodule(name)(z), am.get_submodule(name)(z), hf.get_submodule(name)(z))
        if name in native_norms:
            compare(name + ".XT_native", native_norms[name](z), am.get_submodule(name)(z), hf.get_submodule(name)(z))
    z = records["XT.self_attn"]["input"]
    for c in ["q", "k", "v", "b"]:
        name = c + "_proj"
        compare(
            "KDA." + name,
            getattr(xt.self_attn, name)(z),
            getattr(am.self_attn, name)(z),
            getattr(hf.self_attn, name)(z),
        )
    for idx, c in enumerate("qkv"):
        zc = records["XT.self_attn." + c + "_conv1d"]["input"]
        hc = packed_conv(
            zc.transpose(1, 2), hf.self_attn.conv1d.weight.squeeze(1).chunk(3)[idx], activation="silu"
        ).transpose(1, 2)
        compare(
            "KDA." + c + "_conv1d",
            getattr(xt.self_attn, c + "_conv1d")(zc, cu_seqlens=fixture["cu_seqlens"])[0],
            getattr(am.self_attn, c + "_conv1d")(zc, cu_seqlens=fixture["cu_seqlens"])[0],
            hc,
        )
    xg, xb = xt.self_attn._compute_gate_and_beta(z)
    ag = am.self_attn._fp32_params(
        am.self_attn.f_b_proj(am.self_attn.f_a_proj(z)), cfg.linear_head_dim, cfg.linear_lower_bound
    )
    ab = am.self_attn.b_proj(z).float().sigmoid()
    hg = hf.self_attn.forget_gate(z)
    hb = hf.self_attn.b_proj(z).sigmoid()
    compare("KDA.forget_gate", xg, ag, hg)
    compare("KDA.beta_sigmoid", xb, ab, hb)
    xgate = xt.self_attn._gate_output(z).view_as(records["XT.self_attn.o_norm"]["gate"])
    agate = am.self_attn.g_b_proj(am.self_attn.g_a_proj(z)).view_as(xgate)
    hgate = hf.self_attn.g_b_proj(hf.self_attn.g_a_proj(z)).view_as(xgate)
    compare("KDA.output_gate_projection", xgate, agate, hgate)
    # Capture latest native XT core inputs. All three kernels receive these same tensors.
    core = {}
    orig_chunk = xt_kda.chunk_kda

    def trace_core(**kwargs):
        core.update({k: v for k, v in kwargs.items() if isinstance(v, torch.Tensor)})
        return orig_chunk(**kwargs)

    xt_kda.chunk_kda = trace_core
    xx = xt.self_attn(z, seq_ctx=context)["projected_output"]
    xt_kda.chunk_kda = orig_chunk
    persist("KDA_core_inputs", core)
    core_args = {k: core[k] for k in ["q", "k", "v", "g", "beta"]}
    xc = orig_chunk(
        **core_args,
        cu_seqlens=fixture["cu_seqlens"],
        use_qk_l2norm_in_kernel=True,
        transpose_state_layout=True,
        safe_gate=cfg.linear_lower_bound is not None,
    )[0]
    ac = am_layers._chunk_kda(
        **core_args,
        cu_seqlens=fixture["cu_seqlens"],
        use_qk_l2norm_in_kernel=True,
        transpose_state_layout=True,
        safe_gate=cfg.linear_lower_bound is not None,
    )[0]
    hc = packed_chunk(core["q"], core["k"], core["v"], core["g"], core["beta"])[0]
    compare("KDA.core", xc, ac, hc)
    # Check the HF head partition on the first real 128-token prefix against the unsplit native function.
    small = [core[k][:, :128] for k in ["q", "k", "v", "g", "beta"]]
    whole = hf_chunk(*small, use_qk_l2norm_in_kernel=True)[0]
    split = torch.cat(
        [
            hf_chunk(*[t[:, :, h : h + 8] for t in small], use_qk_l2norm_in_kernel=True)[0]
            for h in range(0, core["q"].shape[2], 8)
        ],
        dim=2,
    )
    assert torch.equal(whole, split), "HF head splitting changed prefix results"
    report["hf_head_split_prefix128_bitwise"] = True
    n = records["XT.self_attn.o_norm"]
    nz, gate = n["input"], n["gate"]
    compare(
        "KDA.gated_norm", xt.self_attn.o_norm(nz, gate), am.self_attn.o_norm(nz, gate), hf.self_attn.o_norm(nz, gate)
    )
    compare(
        "KDA.gated_norm.AM_fix",
        xt.self_attn.o_norm(nz, gate),
        _hf_gated_rms_norm(am.self_attn.o_norm, nz, gate),
        hf.self_attn.o_norm(nz, gate),
    )
    pz = records["XT.self_attn.o_proj"]["input"]
    compare("KDA.o_proj", xt.self_attn.o_proj(pz), am.self_attn.o_proj(pz), hf.self_attn.o_proj(pz))
    print("Starting native HF attention/full-layer replay", flush=True)
    ha = hf.self_attn(z)
    aa = am.self_attn(z, packed_context=packed, padding_mask=fixture["padding_mask"])
    compare("KDA.complete_same_input", xx, aa, ha)
    ho = hf(x)[0]
    compare("layer0.complete_same_input", xo, ao, ho, anchor="shared real layer input")
    # Single-module interventions: one replacement per run, fixed layer input.
    for target in ["input_layernorm", "post_attention_layernorm", "mlp", "self_attn.o_norm"]:
        parent_name, _, leaf = target.rpartition(".")
        parent = am.get_submodule(parent_name) if parent_name else am
        old = getattr(parent, leaf)
        setattr(parent, leaf, hf.get_submodule(target))
        val = am(x, packed_context=packed, padding_mask=fixture["padding_mask"])
        report["layer_interventions"]["AM_HF_" + target] = {"vs_XT": metric(xo, val), "vs_HF": metric(ho, val)}
        setattr(parent, leaf, old)

    def xt_intervention(name):
        val = xt._forward(x, position_embeddings=None, seq_ctx=context)
        report["layer_interventions"][name] = {
            "vs_AM": metric(ao, val),
            "vs_HF": metric(ho, val),
            "vs_XT_baseline": metric(xo, val),
        }

    # Show the production baseline separately: xo above retains the historical HF norm adapter.
    with (
        patch.object(xt.input_layernorm, "forward", native_norms["input_layernorm"]),
        patch.object(xt.post_attention_layernorm, "forward", native_norms["post_attention_layernorm"]),
    ):
        xt_intervention("XT_native_norms")
    reference_hc = Glm5NextTextHyperConnection(HFConfig(**config)).cuda()

    def reference_pre(states, fn, scale, base, *args, **kwargs):
        reference_hc.fn = torch.nn.Parameter(fn)
        reference_hc.scale = torch.nn.Parameter(scale)
        reference_hc.base = torch.nn.Parameter(base)
        post, comb, collapsed = reference_hc(states)
        return collapsed, post, comb

    with patch.object(xt_layers, "hc_pre", reference_pre):
        xt_intervention("XT_HF_mhc_pre")
    with patch.object(xt_layers, "hc_post", hf_post):
        xt_intervention("XT_HF_mhc_post")
    # Upstream gate FP32 blocking is opt-in (default zero), not automatically enabled.
    with patch.object(xt.self_attn, "gate_projection_block_size", 64):
        xt_intervention("XT_opt_in_FP32_gate_projection_block64")

    # Cumulative precision-boundary profile: keep both native KDA entrypoints and
    # XT's SP-stable scan; alter AM's scan primitive, never the whole core/module.
    import importlib

    from xtuner.v1.module.decoder_layer.mhc import hc_split_sinkhorn
    from xtuner.v1.ops.kda.chunk_cumsum import chunk_cumsum as fixed_scan

    fla_forward = importlib.import_module("fla.ops.kda.chunk_fwd")
    xt_kda = importlib.import_module("xtuner.v1.module.attention.kda")
    native_entries = (xt_kda.chunk_kda, am_layers._chunk_kda)
    parameter_ids = [[id(p) for p in model.parameters()] for model in (xt, am)]

    def precision_norm(self, states, gate):
        eps = self.variance_epsilon if hasattr(self, "variance_epsilon") else self.eps
        normalized = states.float() * torch.rsqrt(states.float().square().mean(-1, keepdim=True) + eps)
        return (self.weight.float() * normalized * torch.sigmoid(gate.float())).to(states.dtype)

    def precision_mlp(self, states):
        gate = self.gate_proj(states).clamp(max=self.swiglu_limit)
        up = self.up_proj(states).clamp(min=-self.swiglu_limit, max=self.swiglu_limit)
        return self.down_proj(torch.nn.functional.silu(gate) * up)

    def precision_pre(states, fn, scale, base, hc_mult, iters, eps, norm_eps=1e-6, projection_block_size=0):
        assert projection_block_size == 0
        flat = states.flatten(2).float()
        normed = flat * torch.rsqrt(flat.square().mean(-1, keepdim=True) + norm_eps)
        mixes = torch.nn.functional.linear(normed, fn.float())
        pre, post, comb = hc_split_sinkhorn(mixes, scale, base, hc_mult, iters, eps)
        return (pre.unsqueeze(-1) * states).sum(-2).to(states.dtype), post, comb

    def precision_scan(g, chunk_size, scale, cu_seqlens, chunk_indices):
        assert chunk_size == 64
        return fixed_scan(g, cu_seqlens, chunk_indices, scale)

    cumulative = {}

    def record_cumulative(name):
        left = xt._forward(x, position_embeddings=None, seq_ctx=context)
        right = am(x, packed_context=packed, padding_mask=fixture["padding_mask"])
        cumulative[name] = {
            "XT_vs_AM": metric(left, right),
            "XT_vs_HF": metric(ho, left),
            "AM_vs_HF": metric(ho, right),
        }
        print("CUMULATIVE", name, cumulative[name]["XT_vs_AM"], flush=True)
        return left, right

    with ExitStack() as profile:
        profile.enter_context(
            patch.object(am.self_attn.o_norm, "forward", MethodType(precision_norm, am.self_attn.o_norm))
        )
        record_cumulative("AM_HF_gated_norm")
        profile.enter_context(patch.object(am.mlp, "forward", MethodType(precision_mlp, am.mlp)))
        record_cumulative("plus_AM_HF_mlp")
        profile.enter_context(patch.object(xt_layers, "hc_pre", precision_pre))
        record_cumulative("plus_XT_HF_mhc_pre")
        profile.enter_context(patch.object(xt_layers, "hc_post", hf_post))
        record_cumulative("plus_XT_HF_mhc_post")
        profile.enter_context(
            patch.object(xt.self_attn.o_norm, "forward", MethodType(precision_norm, xt.self_attn.o_norm))
        )
        record_cumulative("plus_XT_HF_gated_norm")
        profile.enter_context(patch.object(fla_forward, "chunk_local_cumsum", precision_scan))
        left, right = record_cumulative("plus_AM_fixed_scan_native_cores_retained")
        assert native_entries == (xt_kda.chunk_kda, am_layers._chunk_kda)
        assert parameter_ids == [[id(p) for p in model.parameters()] for model in (xt, am)]
        assert torch.equal(left[:, valid], right[:, valid])
        if args.profile_backward:

            def canonical_name(name):
                name = name.replace("self_attn._fp32_params.", "self_attn.")
                for site in ("attn", "ffn"):
                    name = name.replace(site + "_hc._fp32_params.", "hc_" + site + "_")
                    name = name.replace(site + "_hc.", "hc_" + site + "_")
                return name

            parameters = [
                {canonical_name(name): p for name, p in model.named_parameters() if p.requires_grad}
                for model in (xt, am)
            ]
            assert parameters[0].keys() == parameters[1].keys()
            generator = torch.Generator(device=x.device).manual_seed(args.backward_seed)
            upstream = torch.randn(left.shape, dtype=left.dtype, device=x.device, generator=generator) * 1e-3
            upstream[:, ~valid] = 0
            native_am_gate = am_layers._fused_kda_gate

            def fp32_gate_input(gate, *gate_args, **gate_kwargs):
                return native_am_gate(gate.float(), *gate_args, **gate_kwargs)

            xt_core = importlib.import_module("xtuner.v1.ops.kda.chunk_kda")
            fla_core = importlib.import_module("fla.ops.kda.chunk")
            native_core_backward = fla_core.chunk_kda_bwd
            assert native_core_backward is xt_core._fla_chunk_kda_bwd
            backward_stages = {}
            beta_reductions = {}
            fla_intra = importlib.import_module("fla.ops.kda.chunk_intra")
            native_intra_kernel = fla_intra.chunk_kda_bwd_kernel_intra

            class TraceIntraKernel:
                def __getitem__(self, grid):
                    def launch(**kwargs):
                        result = native_intra_kernel[grid](**kwargs)
                        beta_reductions.setdefault(side, {})["db2"] = kwargs["db"].detach().clone()
                        report.setdefault("intra_backward_launch", {})[side] = {
                            "configuration": str(getattr(native_intra_kernel.fn, "best_config", None)),
                            "arguments": {
                                name: {
                                    "dtype": str(value.dtype),
                                    "shape": list(value.shape),
                                    "stride": list(value.stride()),
                                }
                                if isinstance(value, torch.Tensor)
                                else value
                                for name, value in kwargs.items()
                            },
                            "cu_seqlens": kwargs["cu_seqlens"].tolist(),
                        }
                        for name in ("dAqk", "dAkk"):
                            backward_stages[side]["intra_input." + name] = (
                                kwargs[name].detach().masked_fill(~lower_triangle, 0)
                            )
                        return result

                    return launch

            fla_backward = importlib.import_module("fla.ops.kda.chunk_bwd")
            native_intra_backward = fla_backward.chunk_kda_bwd_intra

            def trace_intra_backward(**kwargs):
                beta_reductions.setdefault(side, {})["db_base"] = kwargs["db"].detach().clone()
                result = native_intra_backward(**kwargs)
                beta_reductions[side]["db_result"] = result[2].detach().clone()
                return result

            local_rows = torch.cat([torch.arange(stop - start, device=x.device) for start, stop in zip(cu, cu[1:])])
            lower_triangle = (
                torch.arange(64, device=x.device)[None, None, None, :] <= (local_rows % 64)[None, :, None, None]
            )

            def trace_core_backward(**kwargs):
                trace = backward_stages.setdefault(side, {})
                for name in ("q", "k", "v", "g", "beta", "Aqk", "Akk", "do"):
                    value = kwargs[name].detach()
                    if name in ("Aqk", "Akk"):
                        # The native kernels do not initialize unused upper-triangle storage.
                        value = value.masked_fill(~lower_triangle, 0)
                    trace["core_input." + name] = value
                dht = kwargs.get("dht")
                report.setdefault("backward_final_state_gradient", {})[side] = {
                    "present": dht is not None,
                    "nonzero": int(torch.count_nonzero(dht)) if dht is not None else 0,
                }
                if side == "AM_FP32_gate_no_dht":
                    assert dht is not None and torch.count_nonzero(dht) == 0
                    kwargs["dht"] = None
                if side == "AM_FP32_gate_int32_indices":
                    for name in ("cu_seqlens", "chunk_indices"):
                        original = kwargs[name]
                        kwargs[name] = original.to(torch.int32)
                        assert torch.equal(kwargs[name].to(original.dtype), original)
                result = native_core_backward(**kwargs)
                for name, value in zip(("dq", "dk", "dv", "db", "dg"), result):
                    trace["core_output." + name] = value.detach()
                return result

            def trace_beta_projection(module, inputs, output):
                trace = backward_stages.setdefault(side, {})
                trace["beta_projection.input"] = inputs[0].detach()
                trace["beta_projection.output"] = output.detach()

                def record_gradient(gradient):
                    trace["beta_projection.output_grad"] = gradient.detach()

                output.register_hook(record_gradient)

            gradients = []
            for side, model, params in (
                ("XT", xt, parameters[0]),
                ("AM", am, parameters[1]),
                ("AM_FP32_gate_input", am, parameters[1]),
                ("AM_FP32_gate_no_dht", am, parameters[1]),
                ("AM_FP32_gate_int32_indices", am, parameters[1]),
            ):
                print("BACKWARD starting", side, flush=True)
                with torch.enable_grad(), ExitStack() as backward_profile:
                    backward_profile.enter_context(patch.object(xt_core, "_fla_chunk_kda_bwd", trace_core_backward))
                    backward_profile.enter_context(patch.object(fla_core, "chunk_kda_bwd", trace_core_backward))
                    backward_profile.enter_context(
                        patch.object(fla_intra, "chunk_kda_bwd_kernel_intra", TraceIntraKernel())
                    )
                    backward_profile.enter_context(
                        patch.object(fla_backward, "chunk_kda_bwd_intra", trace_intra_backward)
                    )
                    beta_hook = model.self_attn.b_proj.register_forward_hook(trace_beta_projection)
                    backward_profile.callback(beta_hook.remove)
                    if side.startswith("AM_FP32_gate"):
                        backward_profile.enter_context(patch.object(am_layers, "_fused_kda_gate", fp32_gate_input))
                    inp = x.detach().requires_grad_(True)
                    val = (
                        model._forward(inp, position_embeddings=None, seq_ctx=context)
                        if side == "XT"
                        else model(inp, packed_context=packed, padding_mask=fixture["padding_mask"])
                    )
                    assert torch.equal(val[:, valid], left[:, valid]), side
                    grads = torch.autograd.grad(val, (inp, *params.values()), grad_outputs=upstream)
                gradients.append(dict(zip(("input", *params.keys()), grads)))
                del inp, val, grads
                print("BACKWARD complete", side, flush=True)

            for report_key, actual_gradients in zip(
                (
                    "cumulative_backward",
                    "cumulative_backward_AM_FP32_gate_input",
                    "cumulative_backward_AM_FP32_gate_no_dht",
                    "cumulative_backward_AM_FP32_gate_int32_indices",
                ),
                gradients[1:],
            ):
                gradient_rows = {}
                for name, reference in gradients[0].items():
                    actual = actual_gradients[name]
                    assert reference.shape == actual.shape
                    assert torch.isfinite(reference).all() and torch.isfinite(actual).all(), name
                    delta = actual.float() - reference.float()
                    gradient_rows[name] = {
                        "relative_l2": float(
                            delta.norm(dtype=torch.float64) / reference.norm(dtype=torch.float64).clamp_min(1e-30)
                        ),
                        "max_abs": float(delta.abs().max()),
                        "bitwise_equal": reference.dtype == actual.dtype and torch.equal(reference, actual),
                        "shape": list(reference.shape),
                    }
                report[report_key] = {
                    "scope": "Full real layer0, shared deterministic BF16 output cotangent with padding zero; VJP only, no optimizer step",
                    "seed": args.backward_seed,
                    "upstream_scale": 1e-3,
                    "parameters_compared": len(parameters[0]),
                    "all_finite": True,
                    "all_bitwise": all(row["bitwise_equal"] for row in gradient_rows.values()),
                    "rows": gradient_rows,
                }
                print("BACKWARD SUMMARY", report[report_key], flush=True)
            report["backward_stage_comparison"] = {
                side: {
                    name: metric(reference, backward_stages[side][name])
                    for name, reference in backward_stages["XT"].items()
                }
                for side in ("AM", "AM_FP32_gate_input", "AM_FP32_gate_no_dht", "AM_FP32_gate_int32_indices")
            }
            print("BACKWARD STAGES", report["backward_stage_comparison"], flush=True)
            report["beta_reduction_comparison"] = {}
            for side, values in beta_reductions.items():
                if side == "XT":
                    continue
                report["beta_reduction_comparison"][side] = {
                    name: metric(reference.reshape(1, x.shape[1], -1), values[name].reshape(1, x.shape[1], -1))
                    if name != "db2"
                    else metric(reference.permute(1, 2, 0, 3), values[name].permute(1, 2, 0, 3))
                    for name, reference in beta_reductions["XT"].items()
                }
            for side, values in beta_reductions.items():
                persist("backward_beta_" + side, values)
            print("BETA REDUCTION", report["beta_reduction_comparison"], flush=True)
            del gradients, backward_stages, beta_reductions
    if args.xt_native_profile:
        from experiments.train_infra.precision_profile import automodel_native_core_precision

        with automodel_native_core_precision(am) as profile_info:
            native_left = xt._forward(x, position_embeddings=None, seq_ctx=context)
            native_right = am(x, packed_context=packed, padding_mask=fixture["padding_mask"])
            assert torch.equal(native_left[:, valid], xo[:, valid])
            profile_result = {"configuration": profile_info, "forward": metric(native_left, native_right)}
            if args.profile_backward:
                native_gradients = []
                for side, model, params in (("XT", xt, parameters[0]), ("AM", am, parameters[1])):
                    with torch.enable_grad():
                        inp = x.detach().requires_grad_(True)
                        val = (
                            model._forward(inp, position_embeddings=None, seq_ctx=context)
                            if side == "XT"
                            else model(inp, packed_context=packed, padding_mask=fixture["padding_mask"])
                        )
                        expected = native_left if side == "XT" else native_right
                        assert torch.equal(val[:, valid], expected[:, valid])
                        grads = torch.autograd.grad(val, (inp, *params.values()), grad_outputs=upstream)
                    native_gradients.append(dict(zip(("input", *params.keys()), grads)))
                    del inp, val, grads
                rows = {}
                for name, reference in native_gradients[0].items():
                    actual = native_gradients[1][name]
                    assert torch.isfinite(reference).all() and torch.isfinite(actual).all()
                    rows[name] = {
                        "relative_l2": float(
                            (actual.float() - reference.float()).norm(dtype=torch.float64)
                            / reference.norm(dtype=torch.float64).clamp_min(1e-30)
                        ),
                        "bitwise_equal": reference.dtype == actual.dtype and torch.equal(reference, actual),
                    }
                profile_result["backward"] = {
                    "seed": args.backward_seed,
                    "rows": rows,
                    "all_bitwise": all(row["bitwise_equal"] for row in rows.values()),
                }
                del native_gradients
            report["xt_native_precision_profile"] = profile_result
            print("XT NATIVE PRECISION PROFILE", profile_result, flush=True)
        assert torch.equal(am(x, packed_context=packed, padding_mask=fixture["padding_mask"])[:, valid], ao[:, valid])
        report["xt_native_profile_restored_AM"] = True
    report["layer_cumulative"] = cumulative
    report["cumulative_native_kda_entrypoints_retained"] = True
    report["cumulative_parameter_identities_retained"] = True
    report["cumulative_scope"] = (
        "Diagnostic precision profile; SP1; neither KDA module/core replaced; not a native default or training acceptance"
    )
    report["baseline_layer"] = {
        "XT_HF_norm_vs_HF": metric(ho, xo),
        "AM_vs_HF": metric(ho, ao),
        "XT_HF_norm_vs_AM": metric(xo, ao),
    }
    report["complete"] = True
    report["worktree_diff"] = subprocess.check_output(["git", "diff", "HEAD"], text=True)
    report["probe_source"] = Path(__file__).read_text()
    (OUT / "comparison.json").write_text(json.dumps(report, indent=2))
    print("COMPLETE", OUT, flush=True)


if __name__ == "__main__":
    with torch.no_grad():
        main()
