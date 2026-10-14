"""Measure KDA intra-chunk rounding interventions on a real 128-token prefix."""

import argparse
import json
from pathlib import Path

import torch
from fla.modules.l2norm import l2norm_fwd
from fla.ops.common.chunk_delta_h import chunk_gated_delta_rule_fwd_h
from fla.ops.gla.chunk import chunk_gla_fwd_o_gk
from fla.ops.kda.chunk_intra import chunk_kda_fwd_intra
from fla.ops.kda.wy_fast import recompute_w_u_fwd
from fla.ops.utils.constant import RCP_LN2
from fla.ops.utils.index import prepare_chunk_indices
from safetensors.torch import load_file

from experiments.train_infra.hf_eager_kda import eager_function
from experiments.train_infra.probe_kda_reference_boundary import error
from xtuner.v1.ops.kda.chunk_cumsum import chunk_cumsum
from xtuner.v1.ops.kda.chunk_kda import chunk_kda


def chunks(x):
    b, t, h, d = x.shape
    assert t % 64 == 0
    return x.transpose(1, 2).reshape(b, h, t // 64, 64, d).float()


def tokens(x, dtype):
    b, h, n, c, d = x.shape
    return x.reshape(b, h, n * c, d).transpose(1, 2).to(dtype).contiguous()


def recurrence(q, k, v, g, beta):
    q, k, v, g, beta = [x.double() for x in (q, k, v, g, beta)]
    state = q.new_zeros(q.shape[0], q.shape[2], q.shape[3], v.shape[3])
    out = []
    for i in range(q.shape[1]):
        state *= g[:, i].exp().unsqueeze(-1)
        delta = (v[:, i] - (state * k[:, i].unsqueeze(-1)).sum(-2)) * beta[:, i].unsqueeze(-1)
        state += k[:, i].unsqueeze(-1) * delta.unsqueeze(-2)
        out.append((state * (q[:, i] * q.shape[-1] ** -0.5).unsqueeze(-1)).sum(-2))
    return torch.stack(out, dim=1)


def reference_factors(qn, kn, v, beta, gc, akk):
    scale = qn.shape[-1] ** -0.5
    # Keep native q/k normalization AND prefix fixed; compute ideal intra factors in FP32.
    qc, kc, vc, gg = [chunks(x) for x in (qn, kn, v, gc)]
    bc = chunks(beta.unsqueeze(-1))
    decay = (
        (gg.unsqueeze(-2) - gg.unsqueeze(-3))
        .masked_fill(torch.ones(64, 64, device=qn.device, dtype=torch.bool).triu(1)[..., None], -torch.inf)
        .exp2()
    )
    lower = (kc.unsqueeze(-2) * kc.unsqueeze(-3) * decay).sum(-1) * bc
    eye = torch.eye(64, device=qn.device)
    inverse = torch.linalg.solve_triangular(eye + lower.tril(-1), eye.expand_as(lower), upper=False)
    ar = (qc.unsqueeze(-2) * kc.unsqueeze(-3) * decay).sum(-1).tril() * scale
    vbeta = vc * bc
    kbeta = kc * bc * gg.exp2()
    ur, wr = inverse @ vbeta, inverse @ kbeta
    kgr = kc * (gg[..., -1:, :] - gg).exp2()
    del decay, lower
    # Independently verify the factorization against an FP64 sequential recurrence.
    state = torch.zeros(qn.shape[0], qn.shape[2], qn.shape[3], v.shape[3], device=qn.device)
    torch_outputs = []
    for i in range(qn.shape[1] // 64):
        vn = ur[:, :, i] - wr[:, :, i] @ state
        torch_outputs.append((qc[:, :, i] * gg[:, :, i].exp2() * scale) @ state + ar[:, :, i] @ vn)
        state = state * gg[:, :, i, -1].exp2().unsqueeze(-1) + kgr[:, :, i].transpose(-1, -2) @ vn
    torch_reference = tokens(torch.stack(torch_outputs, dim=2), torch.float32)

    native_a = chunks(akk)
    # Remove only premultiplication's BF16 boundary; retain native inverse and final BF16 storage.
    u_precise = tokens(native_a @ vbeta, v.dtype)
    w_precise = tokens(native_a @ kbeta, kn.dtype)
    # Reproduce product rounding too, measuring the Torch/Triton backend residual separately.
    u_emulated = tokens(native_a @ vbeta.to(v.dtype).float(), v.dtype)
    w_emulated = tokens(native_a @ kbeta.to(kn.dtype).float(), kn.dtype)
    return {
        "u_precise": u_precise,
        "w_precise": w_precise,
        "u_emulated": u_emulated,
        "w_emulated": w_emulated,
        "Akk": tokens(inverse, torch.float32),
        "Aqk": tokens(ar, torch.float32),
        "u": tokens(ur, torch.float32),
        "w": tokens(wr, torch.float32),
        "kg": tokens(kgr, torch.float32),
        "torch_reference": torch_reference,
    }


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    assert not args.out.exists()
    data = load_file(str(args.source / "KDA_core_inputs.safetensors"), device="cuda")
    assert int(data["cu_seqlens"][1]) >= 128
    q, k, v, g, beta = [data[name][:, :128].contiguous() for name in ("q", "k", "v", "g", "beta")]
    del data
    cu = torch.tensor([0, 128], device=q.device, dtype=torch.int32)
    ci = prepare_chunk_indices(cu, 64)
    qn, kn = l2norm_fwd(q)[0], l2norm_fwd(k)[0]
    gc = chunk_cumsum(g, cu, ci, RCP_LN2)
    scale = q.shape[-1] ** -0.5
    w, u, _, kg, aqk, akk = chunk_kda_fwd_intra(
        q=qn,
        k=kn,
        v=v,
        gk=gc,
        beta=beta,
        scale=scale,
        cu_seqlens=cu,
        chunk_indices=ci,
        safe_gate=True,
    )
    common = dict(cu_seqlens=cu, chunk_indices=ci, use_exp2=True, transpose_state_layout=True)

    def output(wi=w, ui=u, ki=kg, ai=aqk):
        h, vn, _ = chunk_gated_delta_rule_fwd_h(k=ki, w=wi, u=ui, gk=gc, **common)
        return chunk_gla_fwd_o_gk(q=qn, v=vn, g=gc, A=ai, h=h, scale=scale, **common)

    baseline = output()
    public = chunk_kda(
        q, k, v, g, beta, cu_seqlens=cu, use_qk_l2norm_in_kernel=True, safe_gate=True, transpose_state_layout=True
    )[0]
    assert torch.equal(baseline, public), "Direct stages do not reproduce native XT"
    native_hf = eager_function("chunk_kimi_delta_attention")
    hf = native_hf(q, k, v, g, beta, use_qk_l2norm_in_kernel=True)[0]
    hf_rounded = native_hf(qn, kn, v, g, beta)[0]
    ref_rounded = recurrence(qn, kn, v, g, beta)
    q64, k64 = q.double(), k.double()
    q64 /= (q64.square().sum(-1, keepdim=True) + 1e-6).sqrt()
    k64 /= (k64.square().sum(-1, keepdim=True) + 1e-6).sqrt()
    ref = recurrence(q64, k64, v, g, beta)

    factors = reference_factors(qn, kn, v, beta, gc, akk)
    torch_reference = factors["torch_reference"]
    factor_error = error(ref_rounded, torch_reference)
    assert factor_error < 1e-4, factor_error
    u_precise, w_precise = factors["u_precise"], factors["w_precise"]
    u_emulated, w_emulated = factors["u_emulated"], factors["w_emulated"]
    improved_a = factors["Akk"].to(akk.dtype)
    aw, au, _, akg = recompute_w_u_fwd(k=kn, v=v, beta=beta, A=improved_a, gk=gc, cu_seqlens=cu, chunk_indices=ci)
    outputs = {
        "XT_native": baseline,
        "HF_native": hf,
        "HF_native_with_FLA_normalized_qk": hf_rounded,
        "wu_emulated_native_product_cast": output(wi=w_emulated, ui=u_emulated),
        "u_remove_vbeta_cast": output(ui=u_precise),
        "w_remove_kbeta_cast": output(wi=w_precise),
        "wu_remove_product_cast": output(wi=w_precise, ui=u_precise),
        "Akk_FP32_solve_then_BF16": output(wi=aw, ui=au, ki=akg),
        "Aqk_FP32_reference_then_BF16": output(ai=factors["Aqk"].to(aqk.dtype)),
        "all_intra_FP32_then_BF16": output(
            wi=factors["w"].to(w.dtype),
            ui=factors["u"].to(u.dtype),
            ki=factors["kg"].to(kg.dtype),
            ai=factors["Aqk"].to(aqk.dtype),
        ),
        "Torch_FP32_factorization": torch_reference.to(v.dtype),
    }
    result = {
        "scope": "First 128 tokens of real layer0 input, all heads, forward only; NOT full-pack/training acceptance",
        "shape": list(q.shape),
        "metric": "relative L2 fractions, FP64 reductions",
        "tf32": torch.backends.cuda.matmul.allow_tf32,
        "intervention_limit": "Rebuilt factors use Torch FP32 math; product-cast emulation quantifies backend residual",
        "native_stages_bitwise": True,
        "FP32_factorization_vs_FP64_rounded_qk": factor_error,
        "product_cast_isolation": {
            "u_emulation_vs_native": error(u, u_emulated),
            "w_emulation_vs_native": error(w, w_emulated),
            "u_bitwise": torch.equal(u, u_emulated),
            "w_bitwise": torch.equal(w, w_emulated),
        },
        "rows": {
            name: {
                "vs_HF_native": error(hf, value),
                "vs_HF_rounded_qk": error(hf_rounded, value),
                "vs_FP64_native_qk": error(ref, value),
                "vs_XT_native": error(baseline, value),
            }
            for name, value in outputs.items()
        },
        "stages_vs_FP32_factors": {
            name: error(reference, native)
            for name, reference, native in [
                ("Akk", factors["Akk"], akk),
                # Aqk's upper triangle is uninitialized and ignored by the native output kernel.
                ("Aqk", factors["Aqk"], tokens(chunks(aqk).tril(), aqk.dtype)),
                ("u", factors["u"], u),
                ("w", factors["w"], w),
                ("kg", factors["kg"], kg),
            ]
        },
    }
    assert all(torch.isfinite(x).all() for x in outputs.values())
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
