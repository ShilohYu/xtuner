"""Three-way real-input DSA replay, with indexer and sparse-kernel interventions."""
# ruff: noqa: E402

import argparse
import json
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from torch.distributed.pipelining import microbatch
from torch.nn.attention.flex_attention import BlockMask


microbatch.BlockMask = BlockMask
from nemo_automodel.components.models.common import BackendConfig, cudnn_sparse_attention
from nemo_automodel.components.models.glm5_next import layers as am_code
from nemo_automodel.components.models.glm5_next.config import Glm5NextTextConfig
from nemo_automodel.components.models.glm5_next.model import _packed_context_from_inputs

from experiments.train_infra.plot_forward import errors
from experiments.train_infra.precision_profile import native_dsa_precision
from experiments.train_infra.run_automodel import register_flash_mla_compat
from transformers.models.glm5_next import modeling_glm5_next as hf_code
from transformers.models.glm5_next.configuration_glm5_next import Glm5NextTextConfig as HFConfig
from xtuner.v1.data_proto import SequenceContext
from xtuner.v1.model.moe.glm53 import Glm53TextMoEConfig
from xtuner.v1.model.moe.glm53.glm53 import _glm53_rms_norm
from xtuner.v1.module.rms_norm import RMSNorm
from xtuner.v1.ops.sparse_mla.kpool import torch_kpool_topk_indices


def canonical_ids(ids, width, offset=0):
    ids = ids.reshape(ids.shape[-3] if ids.shape[1] == 1 else ids.shape[1], -1)
    ids = torch.where(ids >= 0, ids + offset, -1)
    ids = torch.nn.functional.pad(ids, (0, width - ids.shape[-1]), value=-1)
    return ids.sort(-1).values


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("/tmp/glm53_forward_resumed_20261013"))
    parser.add_argument("--checkpoint", type=Path, default=Path("/tmp/GLM-5.3-Flash-4layer"))
    parser.add_argument(
        "--fixture", type=Path, default=Path("/tmp/forward_alignment_20261013/pack_rank0_step0.safetensors")
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(exist_ok=False)
    cfg_dict = json.loads((args.checkpoint / "config.json").read_text())["text_config"]
    cfg, hcfg = Glm5NextTextConfig(**cfg_dict), HFConfig(**cfg_dict)
    hcfg._attn_implementation = "eager"
    xcfg = Glm53TextMoEConfig.from_hf(args.checkpoint)
    mapping = json.loads((args.checkpoint / "model.safetensors.index.json").read_text())["weight_map"]
    xt = xcfg.attention.build(hidden_size=cfg.hidden_size, layer_idx=1).cuda().bfloat16()
    am = am_code.Glm5NextSparseAttention(cfg, 1, BackendConfig(attn="cudnn", linear="torch")).cuda()
    hf = hf_code.Glm5NextTextAttention(hcfg, 1).cuda().bfloat16()
    register_flash_mla_compat(cudnn_sparse_attention)
    for model in (xt, am, hf):
        for name, p in model.named_parameters():
            key = "model.language_model.layers.1.self_attn." + name
            with safe_open(str(args.checkpoint / mapping[key]), framework="pt") as shard:
                weight = shard.get_tensor(key)
            p.copy_(weight)
            assert torch.equal(p.cpu(), weight.to(p.dtype)), name
        model.train()
    for norm in xt.modules():
        if isinstance(norm, RMSNorm):
            norm.rms_norm_fn = _glm53_rms_norm
    fixture = load_file(str(args.fixture), device="cuda")
    valid = ~fixture["padding_mask"][0]
    cu = fixture["cu_seqlens"].tolist()
    context = SequenceContext(
        input_ids=fixture["input_ids"],
        cu_seq_lens_q=fixture["cu_seqlens"],
        cu_seq_lens_k=fixture["cu_seqlens"],
        max_length_q=max(b - a for a, b in zip(cu[:-1], cu[1:])),
        max_length_k=max(b - a for a, b in zip(cu[:-1], cu[1:])),
        device="cuda",
    )
    packed = _packed_context_from_inputs(
        fixture["input_ids"], None, fixture["padding_mask"], fixture["cu_seqlens"], None
    )
    # AM computes positive document IDs, including padding documents, before masking output.
    computed = packed.doc_ids[0] > 0
    assert bool(computed[valid].all())
    docs = [(a, b) for a, b in zip(cu[:-1], cu[1:]) if b > a and bool(valid[a:b].any())]
    report = {
        "scope": "layer1 DSA; same real input and checkpoint; train/no_grad/SP1",
        "cu_seqlens": cu,
        "computed_tokens": int(computed.sum()),
        "real_tokens": int(valid.sum()),
        "metric": "relative_l2 = norm(right-left) / norm(left); FP64 reductions; valid tokens only",
        "torch": torch.__version__,
        "tf32": torch.backends.cuda.matmul.allow_tf32,
        "comparisons": {},
        "selections": {},
    }

    def capture(side, name):
        with safe_open(str(args.source / side / "hidden_states.safetensors"), framework="pt", device="cuda") as f:
            return f.get_tensor("layer1." + name)

    def compare(name, left, right):
        report["comparisons"][name] = errors(left[:, valid], right[:, valid])
        print(name, report["comparisons"][name]["relative_l2"], flush=True)
        (args.out / "comparison.json").write_text(json.dumps(report, indent=2))

    width = xt.alignment * ((cfg.index_topk + cfg.index_kpool - 2 + xt.alignment) // xt.alignment)

    def selection(name, left, right):
        changed = (left != right).any(-1)[valid]
        report["selections"][name] = {
            "changed_rows": int(changed.sum()),
            "real_rows": int(valid.sum()),
            "fraction": float(changed.float().mean()),
        }
        print(name, report["selections"][name], flush=True)

    native_sparse = xt.sparse_mla_func
    native_am_sparse = am_code.cudnn_sparse_attention
    raw = {}

    def xt_sparse(q, kv, ids, scale, **kwargs):
        out = native_sparse(q, kv, ids, scale, **kwargs)
        raw.update(q=q, kv=kv, ids=ids, output=out.raw_output)
        return out

    for side in ("xtuner", "automodel"):
        z = capture(side, "input_layernorm")
        print("ANCHOR", side, flush=True)
        with patch.object(xt, "sparse_mla_func", xt_sparse):
            xo = xt(z, seq_ctx=context)["projected_output"]
        am_stages, am_sparse_calls = {}, []
        names = [
            "q_a_proj",
            "q_a_layernorm",
            "q_b_proj",
            "kv_a_proj_with_mqa",
            "kv_a_layernorm",
            "indexer.wk",
            "indexer.k_norm",
            "indexer.wq_b",
            "indexer.weights_proj",
        ]

        def stage_hook(name):
            def hook(module, inputs, output):
                am_stages.setdefault(name, []).append(output.detach())

            return hook

        handles = [am.get_submodule(name).register_forward_hook(stage_hook(name)) for name in names]

        def am_sparse(q, kv, ids, scale, **kwargs):
            am_sparse_calls.append((q, kv, ids))
            return native_am_sparse(q, kv, ids, scale, **kwargs)

        with patch.object(am_code, "cudnn_sparse_attention", am_sparse):
            ao = am(z, packed_context=packed, padding_mask=fixture["padding_mask"])
        for handle in handles:
            handle.remove()
        compare(side + ".XT_vs_AM", xo, ao)
        native_replay = xo if side == "xtuner" else ao
        compare(side + ".native_replay", capture(side, "self_attn"), native_replay)
        assert torch.equal(capture(side, "self_attn")[:, valid], native_replay[:, valid]), side
        if side != "xtuner":
            continue
        fixed = {key: value.clone() for key, value in raw.items()}
        save_file({k: v.cpu().contiguous() for k, v in fixed.items()}, str(args.out / "xt_sparse_inputs.safetensors"))
        q_a = xt.q_a_proj(z)
        q_resid = xt.q_a_layernorm(q_a)
        kv_a = xt.kv_a_proj_with_mqa(z)
        index_k = xt.indexer.wk(z)
        stages = {
            "q_a_proj": q_a,
            "q_a_layernorm": q_resid,
            "q_b_proj": xt.q_b_proj(q_resid),
            "kv_a_proj_with_mqa": kv_a,
            "kv_a_layernorm": xt.kv_a_layernorm(kv_a),
            "indexer.wk": index_k,
            "indexer.k_norm": xt.indexer.k_norm(index_k),
            "indexer.wq_b": xt.indexer.wq_b(q_resid),
            "indexer.weights_proj": xt.indexer.weights_proj(z),
        }
        for name, value in stages.items():
            joined = torch.cat(am_stages[name], dim=1)
            assert joined.shape[1] == int(computed.sum()), (name, joined.shape, int(computed.sum()))
            aligned = torch.zeros_like(value)
            aligned[:, computed] = joined
            if name == "q_a_proj":
                am_q_a = aligned
            compare("projection_geometry." + name, value, aligned)
        for key, slot in [("q", 0), ("kv", 1)]:
            joined = torch.cat([call[slot] for call in am_sparse_calls], dim=0)
            aligned = torch.zeros_like(fixed[key])
            assert joined.shape[0] == int(computed.sum()), (key, joined.shape)
            aligned[computed] = joined
            compare("projection_geometry.sparse_" + key, fixed[key].unsqueeze(0), aligned.unsqueeze(0))
        save_file(
            {key: value.cpu().contiguous() for key, value in stages.items()},
            str(args.out / "xt_indexer_features.safetensors"),
        )
        # Freeze each module input too, separating GEMM geometry from upstream error.
        for name, input_value in [
            ("q_a_proj", z),
            ("q_b_proj", q_resid),
            ("indexer.wq_b", q_resid),
            ("indexer.weights_proj", z),
        ]:
            chunked = torch.zeros_like(stages[name])
            for a, b in docs:
                for start in range(a, b, am.query_chunk_size):
                    end = min(start + am.query_chunk_size, b)
                    chunked[:, start:end] = am.get_submodule(name)(input_value[:, start:end])
            compare("same_module_input.chunk32." + name, stages[name], chunked)
        query = stages["q_b_proj"].view(1, z.shape[1], cfg.num_attention_heads, cfg.qk_nope_head_dim)
        w_kc, w_vc = xt._absorb_weights()
        absorbed = torch.zeros_like(fixed["q"])
        for a, b in docs:
            for start in range(a, b, am.query_chunk_size):
                end = min(start + am.query_chunk_size, b)
                absorbed[start:end] = torch.einsum("bqhd,hdc->bqhc", query[:, start:end], w_kc).squeeze(0)
        compare("same_module_input.chunk32.query_absorption", fixed["q"].unsqueeze(0), absorbed.unsqueeze(0))
        del stages
        am_stages.clear()
        am_sparse_calls.clear()
        # AM computes its own q_a over the full pack; retain native per-document/chunk attention.
        full_q_a = am.q_a_proj(z)
        assert torch.equal(full_q_a, q_a)

        def full_pack_query_projection(chunk):
            assert chunk.untyped_storage().data_ptr() == z.untyped_storage().data_ptr()
            offset, remainder = divmod(chunk.storage_offset() - z.storage_offset(), z.stride(1))
            assert remainder == 0 and 0 <= offset <= z.shape[1] - chunk.shape[1]
            assert torch.equal(chunk, z[:, offset : offset + chunk.shape[1]])
            return full_q_a[:, offset : offset + chunk.shape[1]]

        with patch.object(am.q_a_proj, "forward", full_pack_query_projection):
            am_full_q_a = am(z, packed_context=packed, padding_mask=fixture["padding_mask"])
        compare("AM_own_full_pack_q_a_vs_XT", xo, am_full_q_a)

        def sorted_xt_sparse(q, kv, ids, scale, **kwargs):
            ordered = torch.where(ids >= 0, ids, z.shape[1]).sort(-1).values
            ordered = ordered.masked_fill(ordered == z.shape[1], -1).contiguous()
            return native_sparse(q, kv, ordered, scale, **kwargs)

        with patch.object(xt, "sparse_mla_func", sorted_xt_sparse):
            xt_sorted = xt(z, seq_ctx=context)["projected_output"]
            compare("own_projections.XT_sorted_vs_AM_full_q_a", xt_sorted, am_full_q_a)
            with patch.object(xt.indexer, "_topk_indices_fn", torch_kpool_topk_indices):
                xt_torch_sorted = xt(z, seq_ctx=context)["projected_output"]
            compare("own_projections.XT_Torch_sorted_vs_AM_full_q_a", xt_torch_sorted, am_full_q_a)

        # Extend full-pack GEMM geometry to the indexer's own key/weight/compression projections.
        def sliced_projection(values):
            def project(chunk):
                assert chunk.untyped_storage().data_ptr() == z.untyped_storage().data_ptr()
                offset, remainder = divmod(chunk.storage_offset() - z.storage_offset(), z.stride(1))
                assert remainder == 0 and 0 <= offset <= z.shape[1] - chunk.shape[1]
                return values[:, offset : offset + chunk.shape[1]]

            return project

        full_index_k = am.indexer.wk(z)
        full_index_weights = am.indexer.weights_proj(z)
        full_compress = torch.nn.functional.linear(z, am.indexer.index_kpool_compress_gate)

        def full_pack_pools(doc):
            keys = am.indexer.k_norm(sliced_projection(full_index_k)(doc)).squeeze(0)
            gates = sliced_projection(full_compress)(doc).squeeze(0)
            length = keys.shape[0]
            count = length // am.indexer.index_kpool
            if count:
                width = count * am.indexer.index_kpool
                grouped_keys = keys[:width].view(count, am.indexer.index_kpool, am.indexer.head_dim)
                grouped_gates = gates[:width].view_as(grouped_keys)
                logits = grouped_gates.float() + am.indexer.index_kpool_compress_ape.float().unsqueeze(0)
                pool_keys = (logits.softmax(1).to(keys.dtype) * grouped_keys).sum(1)
                indices = torch.arange(width, device=z.device).view(count, am.indexer.index_kpool)
                return pool_keys, indices
            return keys.new_empty((0, am.indexer.head_dim)), torch.empty(
                (0, am.indexer.index_kpool), device=z.device, dtype=torch.long
            )

        with ExitStack() as geometry:
            geometry.enter_context(patch.object(am.q_a_proj, "forward", full_pack_query_projection))
            geometry.enter_context(
                patch.object(am.indexer.weights_proj, "forward", sliced_projection(full_index_weights))
            )
            geometry.enter_context(patch.object(am.indexer, "prepare_pools", full_pack_pools))
            am_full_projections = am(z, packed_context=packed, padding_mask=fixture["padding_mask"])
        compare("own_projections.all_full_pack_vs_XT", xo, am_full_projections)
        compare("own_projections.all_full_pack_vs_XT_sorted", xt_sorted, am_full_projections)
        compare("own_projections.all_full_pack_vs_XT_Torch_sorted", xt_torch_sorted, am_full_projections)
        with native_dsa_precision(am, "automodel") as am_profile, native_dsa_precision(xt, "xtuner") as xt_profile:
            reusable_am = am(z, packed_context=packed, padding_mask=fixture["padding_mask"])
            reusable_xt = xt(z, seq_ctx=context)["projected_output"]
        compare("reusable_profile.XT_vs_AM", reusable_xt, reusable_am)
        assert torch.equal(reusable_xt[:, valid], reusable_am[:, valid])
        assert torch.equal(reusable_am, am_full_projections)
        assert torch.equal(reusable_xt, xt_torch_sorted)
        assert torch.equal(am(z, packed_context=packed, padding_mask=fixture["padding_mask"]), ao)
        assert torch.equal(xt(z, seq_ctx=context)["projected_output"], xo)
        report["reusable_profile"] = {"automodel": am_profile, "xtuner": xt_profile, "restored": True}
        # One variable: AM query geometry becomes a whole document instead of 32 tokens.
        with patch.object(am, "query_chunk_size", z.shape[1]):
            am_whole = am(z, packed_context=packed, padding_mask=fixture["padding_mask"])
        compare("AM_whole_document_queries_vs_XT", xo, am_whole)
        xt_ids = canonical_ids(fixed["ids"], width)
        hf_ids, am_ids = torch.full_like(xt_ids, -1), torch.full_like(xt_ids, -1)
        ho = torch.zeros_like(z)
        hf_indices = []
        handle = hf.indexer.register_forward_hook(lambda module, inputs, output: hf_indices.append(output))
        for a, b in docs:
            ho[:, a:b] = hf(z[:, a:b], attention_mask=torch.ones(1, b - a, device=z.device, dtype=torch.bool))[0]
            hf_ids[a:b] = canonical_ids(hf_indices.pop(), width, a)
            doc = z[:, a:b]
            pools, indices = am.indexer.prepare_pools(doc)
            for start in range(0, b - a, am.query_chunk_size):
                end = min(start + am.query_chunk_size, b - a)
                qh = doc[:, start:end]
                qr = am.q_a_layernorm(am.q_a_proj(qh))
                ids = am.indexer.select(qh, qr, torch.arange(start, end, device=z.device), pools, indices, b - a)
                am_ids[a + start : a + end] = canonical_ids(ids, width, a)
        handle.remove()
        compare("XT_vs_HF", ho, xo)
        compare("AM_vs_HF", ho, ao)
        selection("native_XT_vs_AM", xt_ids, am_ids)
        selection("native_XT_vs_HF", xt_ids, hf_ids)
        selection("native_AM_vs_HF", am_ids, hf_ids)
        # Change only XT's indexer scoring backend, keeping its projected inputs fixed.
        with (
            patch.object(xt.indexer, "_topk_indices_fn", torch_kpool_topk_indices),
            patch.object(xt, "sparse_mla_func", xt_sparse),
        ):
            torch_output = xt(z, seq_ctx=context)["projected_output"]
        selection("XT_TileLang_vs_Torch", xt_ids, canonical_ids(raw["ids"], width))
        selection("XT_Torch_vs_HF", canonical_ids(raw["ids"], width), hf_ids)
        compare("XT_Torch_indexer_vs_AM", ao, torch_output)
        # Fixed key sets: retain each framework's own projections and attention math.
        with patch.object(xt.indexer, "forward", lambda *a, **kw: am_ids.unsqueeze(1)):
            changed = xt(z, seq_ctx=context)["projected_output"]
        compare("XT_AM_keys_vs_AM", ao, changed)
        with patch.object(xt.indexer, "forward", lambda *a, **kw: hf_ids.unsqueeze(1)):
            changed = xt(z, seq_ctx=context)["projected_output"]
        compare("XT_HF_keys_vs_HF", ho, changed)
        # Diagnostic interventions, not native alignment: reuse just AM's q_a output,
        # then additionally its selected keys and order; every downstream XT op stays native.
        compare("intervention.native_XT_vs_AM", ao, xo)
        with patch.object(xt.q_a_proj, "forward", lambda inputs: am_q_a):
            changed = xt(z, seq_ctx=context)["projected_output"]
            compare("intervention.XT_AM_q_a_vs_AM", ao, changed)
            ordered_am_ids = torch.where(am_ids >= 0, am_ids, z.shape[1]).sort(-1).values
            ordered_am_ids = ordered_am_ids.masked_fill(ordered_am_ids == z.shape[1], -1).unsqueeze(1).contiguous()
            with patch.object(xt.indexer, "forward", lambda *a, **kw: ordered_am_ids):
                changed = xt(z, seq_ctx=context)["projected_output"]
            compare("intervention.XT_AM_q_a_keys_order_vs_AM", ao, changed)
            assert torch.equal(ao[:, valid], changed[:, valid])
        # All three sparse inputs fixed; isolates wrapper/index ordering differences.
        core_am = native_am_sparse(fixed["q"], fixed["kv"], fixed["ids"], xt.softmax_scale)
        compare("same_sparse_inputs.XT_vs_AM", fixed["output"].unsqueeze(0), core_am.unsqueeze(0))
        sorted_ids = torch.where(fixed["ids"] >= 0, fixed["ids"], z.shape[1]).sort(-1).values
        sorted_ids = sorted_ids.masked_fill(sorted_ids == z.shape[1], -1).contiguous()
        sorted_output = native_sparse(
            fixed["q"], fixed["kv"], sorted_ids, xt.softmax_scale, value_dim=cfg.kv_lora_rank
        ).raw_output
        compare("same_sparse_inputs.XT_sorted_vs_AM", sorted_output.unsqueeze(0), core_am.unsqueeze(0))
        assert torch.equal(sorted_output[valid], core_am[valid])
        # Fixed q/kv/key sets, but AM's native per-document sparse launch geometry.
        doc_core = torch.zeros_like(core_am)
        for a, b in docs:
            ids = fixed["ids"][a:b]
            assert bool(((ids < 0) | ((ids >= a) & (ids < b))).all())
            ids = torch.where(ids >= 0, ids - a, -1)
            doc_core[a:b] = native_am_sparse(fixed["q"][a:b], fixed["kv"][a:b], ids, xt.softmax_scale)
        compare("same_sparse_inputs.AM_pack_vs_documents", core_am.unsqueeze(0), doc_core.unsqueeze(0))
        unfolded = torch.einsum("shm,hdm->shd", core_am, w_vc)
        doc_unfolded = torch.zeros_like(unfolded)
        for a, b in docs:
            doc_unfolded[a:b] = torch.einsum("qhc,hvc->qhv", core_am[a:b], w_vc)
        compare("same_module_input.value_unfold", unfolded.unsqueeze(0), doc_unfolded.unsqueeze(0))
        unfolded = unfolded.reshape(1, z.shape[1], -1)
        projected = xt.o_proj(unfolded)
        doc_projected = torch.zeros_like(projected)
        for a, b in docs:
            doc_projected[:, a:b] = am.o_proj(unfolded[:, a:b])
        compare("same_module_input.o_proj", projected, doc_projected)
        save_file(
            {"xt_ids": xt_ids.cpu(), "am_ids": am_ids.cpu(), "hf_ids": hf_ids.cpu()},
            str(args.out / "indices.safetensors"),
        )
    report["complete"] = True
    (args.out / "comparison.json").write_text(json.dumps(report, indent=2))
    print("COMPLETE", flush=True)


if __name__ == "__main__":
    main()
