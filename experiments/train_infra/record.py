"""Experiment-only input provenance and metrics; native model/optimizer remain unchanged."""

import hashlib
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist

from xtuner.v1.datasets.collator import glm53_vl_sft_collator as native_collator


def tensor_hash(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def strict_getitem(self, index):
    # Fail on real input errors instead of silently replacing the example with fake_data.
    offset = int(self.offsets[index])
    with open(self.path) as stream:
        stream.seek(offset)
        raw = json.loads(stream.readline())
    item = self.tokenize_fn(raw, media_root=self.media_root)
    item["_source_offset"] = offset
    return item


def recording_collator(instances, **kwargs):
    batches = native_collator(instances, **kwargs)
    for samples, batch in zip(instances, batches):
        ctx = batch["seq_ctx"]
        kept_length = ctx.input_ids.shape[-1] - ctx.num_padding + 1
        offsets, total = [], 0
        for sample in samples:
            if total >= kept_length:
                break
            offsets.append(sample["_source_offset"])
            total += sample["num_tokens"]
        batch["_source_offsets"] = offsets
    return batches


def install():
    from xtuner.v1.datasets import config as data_config
    from xtuner.v1.datasets.vlm_jsonl import VLMJsonlDataset
    from xtuner.v1.engine.train_engine import TrainEngine
    from xtuner.v1.train.trainer import Trainer

    VLMJsonlDataset.__getitem__ = strict_getitem
    data_config.glm53_vl_sft_collator = recording_collator
    out = Path(os.environ["WORK_DIR"])
    prepare = Trainer._prepare_model_input
    clip = TrainEngine.clip_grad_norm
    train = TrainEngine.train_step
    optimize = TrainEngine.step_optimizer

    def prepare_record(self, batches):
        step, rank = self.cur_step, dist.get_rank()
        records = []
        for batch in batches:
            ctx = batch["seq_ctx"]
            tensors = {"input_ids": ctx.input_ids, "labels": batch["shifted_labels"],
                       "cu_seqlens": ctx.cu_seq_lens_q}
            for key in ("pixel_values", "image_grid_thw", "mm_token_type_ids"):
                value = getattr(ctx, key, None)
                if value is not None:
                    tensors[key] = value
            records.append({"source_offsets": batch["_source_offsets"],
                            "num_padding": ctx.num_padding,
                            "hashes": {key: tensor_hash(value) for key, value in tensors.items()},
                            "shapes": {key: list(value.shape) for key, value in tensors.items()}})
        with (out / f"inputs_rank{rank}.jsonl").open("a") as stream:
            stream.write(json.dumps({"step": step, "batches": records}) + "\n")
        print(f"INPUT_RECORDED rank={rank} step={step}", flush=True)
        return prepare(self, batches)

    def train_record(self, batches):
        self._experiment_step = getattr(self, "_experiment_step", 0)
        print(f"STAGE rank={dist.get_rank()} step={self._experiment_step} forward_backward", flush=True)
        result = train(self, batches)
        loss = torch.tensor(result["total_loss"], device="cuda", dtype=torch.float64)
        dist.all_reduce(loss)
        self._experiment_loss = loss.item()
        return result

    def clip_record(self, *args, **kwargs):
        print(f"STAGE rank={dist.get_rank()} clip_reduce", flush=True)
        return clip(self, *args, **kwargs)

    def optimize_record(self, norm):
        # A small trainable FP32 parameter provides exact nonzero-update evidence.
        probe_name, probe = next((name, p) for name, p in self.model.trainable_parameters()
                                if p.dtype == torch.float32 and p.numel() <= 16384)
        local = probe.to_local() if hasattr(probe, "to_local") else probe
        before = local.detach().clone()
        lr = self.optimizer.param_groups[0]["lr"]
        print(f"STAGE rank={dist.get_rank()} step={self._experiment_step} optimizer lr={lr}", flush=True)
        result = optimize(self, norm)
        record = {"step": self._experiment_step, "loss": self._experiment_loss,
                  "grad_norm": float(norm), "lr": lr, "probe": probe_name,
                  "update_max_abs": float((local.detach() - before).abs().max()),
                  "peak_memory_bytes": torch.cuda.max_memory_allocated()}
        with (out / f"metrics_rank{dist.get_rank()}.jsonl").open("a") as stream:
            stream.write(json.dumps(record) + "\n")
        self._experiment_step += 1
        return result

    Trainer._prepare_model_input = prepare_record
    TrainEngine.train_step = train_record
    TrainEngine.clip_grad_norm = clip_record
    TrainEngine.step_optimizer = optimize_record


if __name__ == "__main__":
    install()
    from xtuner.v1.train.cli.sft import app

    app(exit_on_error=False)
