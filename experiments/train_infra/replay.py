"""Rebuild the exact XTuner packs from source offsets, with tensor-hash checks."""

import json
from pathlib import Path

import torch
from transformers import AutoTokenizer

from experiments.train_infra.preflight import PROCESSOR, SOURCE, TOKEN_BUDGET
from experiments.train_infra.record import tensor_hash
from xtuner.v1.datasets.collator import glm53_vl_sft_collator
from xtuner.v1.datasets.mllm_tokenize_fn.glm53_vl_tokenize_fn import Glm53VLTokenizeFunction


def square_weights(labels, cu_seqlens):
    weights = torch.zeros_like(labels, dtype=torch.float32)
    for begin, end in zip(cu_seqlens[:-1].tolist(), cu_seqlens[1:].tolist()):
        valid = labels[0, begin:end] != -100
        count = int(valid.sum())
        if count:
            weights[0, begin:end] = valid.float() / count**0.5
    return weights


class Replay:
    def __init__(self, reference_dir, rank):
        reference_dir = Path(reference_dir)
        cfg = json.loads((reference_dir / "resolved_config.json").read_text())
        self.length = cfg["dataloader_cfg"]["pack_max_length"]
        self.entries = [json.loads(raw) for raw in (reference_dir / f"inputs_rank{rank}.jsonl").open()]
        tokenizer = AutoTokenizer.from_pretrained(PROCESSOR, local_files_only=True)
        self.tokenize = Glm53VLTokenizeFunction(
            tokenizer, PROCESSOR, SOURCE.name, max_length=self.length,
            max_frames=24, max_pixels=TOKEN_BUDGET,
            llm_pack_weight=-3.2, visual_pack_weight=5.0, debug=True,
        )
        self.tokenize.set_state("runtime")

    def batches(self, entry):
        batches = []
        for contract in entry["batches"]:
            samples = []
            with SOURCE.open() as stream:
                for offset in contract["source_offsets"]:
                    stream.seek(offset)
                    samples.append(self.tokenize(json.loads(stream.readline()), media_root=str(SOURCE.parent)))
            batch = glm53_vl_sft_collator(
                [samples], pack_max_length=self.length,
                padding_token_idx=self.tokenize.tokenizer.pad_token_id or 0,
                image_token_id=self.tokenize.processor.image_token_id,
                merge_unit=self.tokenize.merge_unit,
            )[0]
            ctx = batch["seq_ctx"]
            tensors = {"input_ids": ctx.input_ids, "labels": batch["shifted_labels"],
                       "cu_seqlens": ctx.cu_seq_lens_q}
            for key in ("pixel_values", "image_grid_thw", "mm_token_type_ids"):
                value = getattr(ctx, key, None)
                if value is not None:
                    tensors[key] = value
            for key, expected in contract["hashes"].items():
                actual = tensor_hash(tensors[key])
                assert actual == expected, f"Replay tensor mismatch: {key}, {actual} != {expected}"
            padding = torch.zeros_like(tensors["input_ids"], dtype=torch.bool)
            if contract["num_padding"]:
                padding[:, -contract["num_padding"]:] = True
            tensors["padding_mask"] = padding
            tensors["loss_weights"] = square_weights(tensors["labels"], tensors["cu_seqlens"])
            batches.append(tensors)
        return batches
