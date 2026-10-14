"""Isolate q/k normalization rounding using recorded real KDA core inputs."""

import argparse
import json
from pathlib import Path

import torch
from fla.modules.l2norm import l2norm_fwd
from fla.ops.kda import chunk_kda as fla_chunk
from safetensors.torch import load_file

from experiments.train_infra.hf_eager_kda import packed_chunk
from xtuner.v1.ops.kda.chunk_kda import chunk_kda as xt_chunk


def error(reference, actual):
    ref, val = reference.double(), actual.double()
    return float((ref - val).norm() / ref.norm())


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    data = load_file(str(args.source / "KDA_core_inputs.safetensors"), device="cuda")
    valid = load_file(str(args.source / "layer_input.safetensors"), device="cuda")["valid"]
    q, k, v, g, beta = [data[name] for name in ("q", "k", "v", "g", "beta")]
    cu = data["cu_seqlens"]
    common = dict(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        cu_seqlens=cu,
        use_qk_l2norm_in_kernel=True,
        transpose_state_layout=True,
        safe_gate=True,
    )
    xt, am = xt_chunk(**common)[0], fla_chunk(**common)[0]
    hf = packed_chunk(q, k, v, g, beta, cu=cu.tolist())[0]
    qn, kn = l2norm_fwd(q)[0], l2norm_fwd(k)[0]
    assert qn.dtype == q.dtype and kn.dtype == k.dtype
    hf_rounded = packed_chunk(qn, kn, v, g, beta, cu=cu.tolist(), use_qk_l2norm_in_kernel=False)[0]
    result = {
        "scope": "Real shared KDA inputs; HF native Torch, FLA and XT; FP64 error reductions",
        "rows": {
            name: {
                "vs_HF_native": error(hf[:, valid], value[:, valid]),
                "vs_HF_FLA_normalized_qk": error(hf_rounded[:, valid], value[:, valid]),
            }
            for name, value in [("XT", xt), ("AM_FLA", am)]
        },
    }
    # An independent FP64 recurrence on the first document, retaining unrounded q/k norms.
    end = min(128, int(cu[1]))
    assert end > 0
    qr, kr, vr, gr, br = [t[:, :end].double() for t in (q, k, v, g, beta)]
    qr = qr / (qr.square().sum(-1, keepdim=True) + 1e-6).sqrt()
    kr = kr / (kr.square().sum(-1, keepdim=True) + 1e-6).sqrt()
    state = torch.zeros(q.shape[0], q.shape[2], q.shape[3], v.shape[3], device=q.device, dtype=torch.float64)
    outputs = []
    for i in range(end):
        state = state * gr[:, i].exp().unsqueeze(-1)
        delta = (vr[:, i] - (state * kr[:, i].unsqueeze(-1)).sum(-2)) * br[:, i].unsqueeze(-1)
        state = state + kr[:, i].unsqueeze(-1) * delta.unsqueeze(-2)
        outputs.append((state * (qr[:, i] * q.shape[-1] ** -0.5).unsqueeze(-1)).sum(-2))
    reference = torch.stack(outputs, dim=1)
    result["fp64_recurrence_prefix"] = {
        "tokens": end,
        "errors": {
            name: error(reference, value[:, :end])
            for name, value in [("HF_native", hf), ("HF_FLA_normalized_qk", hf_rounded), ("XT", xt), ("AM_FLA", am)]
        },
    }
    assert torch.isfinite(reference).all()
    assert result["fp64_recurrence_prefix"]["errors"]["HF_native"] < 0.01
    args.out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
