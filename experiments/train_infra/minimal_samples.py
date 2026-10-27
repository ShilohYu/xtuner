"""Three intact samples on the preview tokenizer, replayed independently on every rank."""

import argparse
import hashlib
import json
import os
from pathlib import Path

import torch
from safetensors.torch import save_file

from experiments.train_infra.preflight import PROCESSOR, SOURCE, TOKEN_BUDGET, check_item
from experiments.train_infra.record import recording_collator, tensor_hash
from experiments.train_infra.replay import square_weights
from transformers import AutoTokenizer
from xtuner.v1.datasets.mllm_tokenize_fn.glm53_vl_tokenize_fn import Glm53VLTokenizeFunction


def fixed_dataloader(self, *, tokenizer, dp_mesh, global_batch_size, micro_batch_size, seed, total_step):
    """Use the native collator without the full dataset scan or additional packing."""
    assert global_batch_size == 8 and micro_batch_size == 1 and dp_mesh.size() == 8
    assert total_step in (2, 5), "Two-step localization or five-step acceptance only"
    sample_path = Path(os.environ["TRAIN_INFRA_FIXED_SAMPLE"])
    item = torch.load(sample_path, weights_only=True)
    meta = json.loads(sample_path.with_suffix(".json").read_text())
    assert self.pack_max_length == meta["pack_length"]
    assert self.pack_max_length >= item["num_tokens"]
    batches = []
    for _ in range(total_step):
        batch = recording_collator(
            [[item]],
            pack_max_length=self.pack_max_length,
            padding_token_idx=tokenizer.pad_token_id or 0,
            **self.collator_kwargs,
        )[0]
        ctx = batch["seq_ctx"]
        assert tensor_hash(ctx.input_ids) == meta["fixture_hashes"]["input_ids"]
        assert batch["_source_offsets"] == [meta["source_byte_offset"]]
        batches.append([batch])
    return batches


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    tokenizer = AutoTokenizer.from_pretrained(PROCESSOR, local_files_only=True)
    tokenize = Glm53VLTokenizeFunction(
        tokenizer,
        PROCESSOR,
        SOURCE.name,
        max_length=None,
        max_frames=24,
        max_pixels=TOKEN_BUDGET,
        llm_pack_weight=-3.2,
        visual_pack_weight=5.0,
        debug=True,
    )
    tokenize.set_state("runtime")
    reports = []
    for spec in json.loads(args.selection.read_text())["samples"]:
        with SOURCE.open() as stream:
            stream.seek(spec["source_byte_offset"])
            row = json.loads(stream.readline())
        row_hash = hashlib.sha256(json.dumps(row, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
        assert row_hash == spec["row_sha256"]
        item = tokenize(row, media_root=str(SOURCE.parent))
        item["_source_offset"] = spec["source_byte_offset"]
        count = check_item(item, tokenize.processor.image_token_id, spec["media_count"], tokenize.merge_unit)
        assert count == spec["historical_supervised_tokens"]
        assert item["num_tokens"] == spec["historical_runtime_tokens"]
        assert (
            hashlib.sha256(torch.as_tensor(item["input_ids"]).numpy().tobytes()).hexdigest()
            == spec["historical_input_ids_sha256"]
        )
        # Keep the complete unshifted sample and round only to the 64-token kernel tile.
        length = (item["num_tokens"] + 63) // 64 * 64
        batch = recording_collator(
            [[item]],
            pack_max_length=length,
            padding_token_idx=tokenizer.pad_token_id or 0,
            image_token_id=tokenize.processor.image_token_id,
            merge_unit=tokenize.merge_unit,
        )[0]
        ctx = batch["seq_ctx"]
        tensors = {"input_ids": ctx.input_ids, "labels": batch["shifted_labels"], "cu_seqlens": ctx.cu_seq_lens_q}
        for key in ("pixel_values", "image_grid_thw", "mm_token_type_ids"):
            value = getattr(ctx, key, None)
            if value is not None:
                tensors[key] = value
        padding = torch.zeros_like(tensors["input_ids"], dtype=torch.bool)
        if ctx.num_padding:
            padding[:, -ctx.num_padding :] = True
        tensors["padding_mask"] = padding
        tensors["loss_weights"] = square_weights(tensors["labels"], tensors["cu_seqlens"])
        assert int((tensors["labels"] != -100).sum()) == count
        assert length - ctx.num_padding == item["num_tokens"] - 1
        modality = spec["modality"]
        torch.save(item, args.out / f"{modality}.pt")
        save_file({k: v.contiguous() for k, v in tensors.items()}, str(args.out / f"{modality}.safetensors"))
        report = {
            **spec,
            "pack_length": length,
            "real_tokens": length - ctx.num_padding,
            "num_padding": ctx.num_padding,
            "supervised_tokens": count,
            "fixture_hashes": {k: tensor_hash(v) for k, v in tensors.items()},
            "new_baseline_tokenization_status": "verified; complete inputs match historical token hashes",
        }
        (args.out / f"{modality}.json").write_text(json.dumps(report, indent=2))
        reports.append(report)
        print(modality, length, ctx.num_padding, count, flush=True)
    (args.out / "manifest.json").write_text(json.dumps({"samples": reports}, indent=2))


if __name__ == "__main__":
    main()
