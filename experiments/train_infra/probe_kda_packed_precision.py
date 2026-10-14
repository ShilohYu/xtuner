"""Full-pack KDA precision interventions with document resets and independent FP64 references."""

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from fla.modules.l2norm import l2norm_fwd
from fla.ops.common.chunk_delta_h import chunk_gated_delta_rule_fwd_h
from fla.ops.gla.chunk import chunk_gla_fwd_o_gk
from fla.ops.kda import chunk_kda as am_chunk_kda
from fla.ops.kda.wy_fast import recompute_w_u_fwd
from safetensors.torch import load_file

from experiments.train_infra.hf_eager_kda import packed_chunk
from experiments.train_infra.probe_kda_intra_precision import chunks, recurrence, reference_factors, tokens
from experiments.train_infra.probe_kda_reference_boundary import error
from xtuner.v1.ops.kda.chunk_kda import RCP_LN2, _chunk_indices, chunk_cumsum, chunk_kda, chunk_kda_fwd_intra


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    assert not args.out.exists()
    data = load_file(str(args.source / "KDA_core_inputs.safetensors"), device="cuda")
    valid = load_file(str(args.source / "layer_input.safetensors"), device="cuda")["valid"]
    q, k, v, g, beta = [data[name] for name in ("q", "k", "v", "g", "beta")]
    cu = data["cu_seqlens"]
    bounds = cu.tolist()
    ci = _chunk_indices(cu)
    qn, kn = l2norm_fwd(q)[0], l2norm_fwd(k)[0]
    gc = chunk_cumsum(g, cu, ci, RCP_LN2)
    scale = q.shape[-1] ** -0.5
    w, u, _, kg, aqk, akk = chunk_kda_fwd_intra(
        q=qn, k=kn, v=v, gk=gc, beta=beta, scale=scale, cu_seqlens=cu, chunk_indices=ci, safe_gate=True
    )
    common = dict(cu_seqlens=cu, chunk_indices=ci, use_exp2=True, transpose_state_layout=True)

    def output(wi=w, ui=u, ki=kg, ai=aqk):
        h, vn, _ = chunk_gated_delta_rule_fwd_h(k=ki, w=wi, u=ui, gk=gc, **common)
        return chunk_gla_fwd_o_gk(q=qn, v=vn, g=gc, A=ai, h=h, scale=scale, **common)

    baseline = output()
    am = am_chunk_kda(
        q, k, v, g, beta, cu_seqlens=cu, use_qk_l2norm_in_kernel=True, safe_gate=True, transpose_state_layout=True
    )[0]
    public = chunk_kda(
        q, k, v, g, beta, cu_seqlens=cu, use_qk_l2norm_in_kernel=True, safe_gate=True, transpose_state_layout=True
    )[0]
    assert torch.equal(baseline, public)
    del public
    print("native full-pack stages bitwise", flush=True)
    hf = packed_chunk(q, k, v, g, beta, cu=bounds)[0]
    hf_rounded = packed_chunk(qn, kn, v, g, beta, cu=bounds, use_qk_l2norm_in_kernel=False)[0]
    print("native HF references complete", flush=True)
    native = {"u": u, "w": w, "kg": kg, "Akk": akk, "Aqk": aqk}
    factors = {name: torch.empty_like(value) for name, value in native.items()}
    for name in ["u_precise", "u_emulated"]:
        factors[name] = torch.empty_like(u)
    for name in ["w_precise", "w_emulated"]:
        factors[name] = torch.empty_like(w)
    factors["torch_reference"] = torch.empty_like(v, dtype=torch.float32)
    ref, ref_rounded = [torch.empty_like(v, dtype=torch.float64) for _ in range(2)]
    sums = {name: [0.0, 0.0] for name in native}

    def pad(x, a, b, head):
        x = x[:, a:b, head : head + 8]
        return F.pad(x, (0, 0, 0, 0, 0, (-x.shape[1]) % 64)).contiguous()

    for a, b in zip(bounds[:-1], bounds[1:]):
        if b <= a:
            continue
        qr, kr = q[:, a:b].double(), k[:, a:b].double()
        qr = qr / (qr.square().sum(-1, keepdim=True) + 1e-6).sqrt()
        kr = kr / (kr.square().sum(-1, keepdim=True) + 1e-6).sqrt()
        ref[:, a:b] = recurrence(qr, kr, v[:, a:b], g[:, a:b], beta[:, a:b])
        ref_rounded[:, a:b] = recurrence(qn[:, a:b], kn[:, a:b], v[:, a:b], g[:, a:b], beta[:, a:b])
        del qr, kr
        for head in range(0, q.shape[2], 8):
            pp = [pad(x, a, b, head) for x in (qn, kn, v, beta.unsqueeze(-1), gc, akk)]
            # A partial last chunk ends at the actual last prefix, not a zero padded prefix.
            pp[4][:, b - a :] = pp[4][:, b - a - 1 : b - a]
            ff = reference_factors(pp[0], pp[1], pp[2], pp[3].squeeze(-1), pp[4], pp[5])
            sl = (slice(None), slice(a, b), slice(head, head + 8))
            for name, value in ff.items():
                factors[name][sl] = value[:, : b - a].to(factors[name].dtype)
            for name, value in native.items():
                target = ff[name][:, : b - a][:, valid[a:b]].double()
                original = pad(value, a, b, head)
                if name == "Aqk":
                    original = tokens(chunks(original).tril(), value.dtype)
                original = original[:, : b - a][:, valid[a:b]].double()
                sums[name][0] += float((target - original).square().sum())
                sums[name][1] += float(target.square().sum())
            del pp, ff
        print("document", a, b, "real", int(valid[a:b].sum()), flush=True)
    factor_error = error(ref_rounded[:, valid], factors["torch_reference"][:, valid])
    assert factor_error < 1e-4, factor_error
    aw, au, _, akg = recompute_w_u_fwd(k=kn, v=v, beta=beta, A=factors["Akk"], gk=gc, cu_seqlens=cu, chunk_indices=ci)
    result = {
        "scope": "Full real 8192-token pack, all 7 documents, all 64 heads; forward only",
        "real_tokens": int(valid.sum()),
        "cu_seqlens": bounds,
        "native_stages_bitwise": True,
        "tf32": torch.backends.cuda.matmul.allow_tf32,
        "FP32_factorization_vs_FP64_rounded_qk": factor_error,
        "metric": "relative L2 fractions; FP64 reductions; padding excluded",
        "stages_vs_FP32_factors": {name: (num / den) ** 0.5 for name, (num, den) in sums.items()},
        "rows": {},
    }

    def record(name, value):
        assert torch.isfinite(value).all(), name
        result["rows"][name] = {
            label: error(target[:, valid], value[:, valid])
            for label, target in [
                ("vs_HF_native", hf),
                ("vs_HF_rounded_qk", hf_rounded),
                ("vs_FP64_native_qk", ref),
                ("vs_XT_native", baseline),
                ("vs_AM_native", am),
            ]
        }
        args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        print(name, result["rows"][name], flush=True)

    record("XT_native", baseline)
    record("AM_FLA_native", am)
    record("HF_native", hf)
    record("HF_native_with_FLA_normalized_qk", hf_rounded)
    record("wu_emulated_native_product_cast", output(wi=factors["w_emulated"], ui=factors["u_emulated"]))
    record("u_remove_vbeta_cast", output(ui=factors["u_precise"]))
    record("w_remove_kbeta_cast", output(wi=factors["w_precise"]))
    record("wu_remove_product_cast", output(wi=factors["w_precise"], ui=factors["u_precise"]))
    record("Akk_FP32_solve_then_BF16", output(wi=aw, ui=au, ki=akg))
    record("Aqk_FP32_reference_then_BF16", output(ai=factors["Aqk"]))
    record("all_intra_FP32_then_BF16", output(wi=factors["w"], ui=factors["u"], ki=factors["kg"], ai=factors["Aqk"]))
    record("Torch_FP32_factorization", factors["torch_reference"].to(v.dtype))
    result["complete"] = True
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print("COMPLETE", flush=True)


if __name__ == "__main__":
    main()
