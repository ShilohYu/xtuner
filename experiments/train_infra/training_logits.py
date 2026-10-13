"""Fixed-input logits before/after training; observation preserves RNG and router state."""

import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
from safetensors.torch import load_file, save_file

from experiments.train_infra.record import tensor_hash


@torch.no_grad()
def capture_logits(model, backend, out, completed_updates):
    fixture_path = os.environ["TRAIN_INFRA_LOGITS_FIXTURE"]
    fixture = load_file(fixture_path)
    batch = {key: value.cuda() for key, value in fixture.items()}
    positions = (batch["labels"][0] != -100).nonzero().flatten()
    assert positions.numel()
    buffers = [(v, v.detach().clone()) for v in model.buffers()]
    loads = [(m, m._cumulative_expert_load) for m in model.modules() if hasattr(m, "_cumulative_expert_load")]
    print(f"STAGE rank={dist.get_rank()} logits completed_updates={completed_updates}", flush=True)
    try:
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
            if backend == "xtuner":
                from xtuner.v1.data_proto import SequenceContext

                cu = batch["cu_seqlens"]
                maximum = int((cu[1:] - cu[:-1]).max())
                ctx = SequenceContext(
                    input_ids=batch["input_ids"], cu_seq_lens_q=cu, cu_seq_lens_k=cu,
                    max_length_q=maximum, max_length_k=maximum, device="cuda",
                    num_padding=int(batch["padding_mask"].sum()),
                    pixel_values=batch["pixel_values"].bfloat16(), image_grid_thw=batch["image_grid_thw"],
                    mm_token_type_ids=batch["mm_token_type_ids"],
                )
                output = model(seq_ctx=ctx, loss_ctx=None)
                logits = output.logits[:, positions]
            else:
                output = model(
                    input_ids=batch["input_ids"], pixel_values=batch["pixel_values"].bfloat16(),
                    image_grid_thw=batch["image_grid_thw"], cu_seqlens=batch["cu_seqlens"],
                    padding_mask=batch["padding_mask"], logits_to_keep=positions,
                )
                logits = output.logits
            assert torch.isfinite(logits).all(), "Nonfinite fixed-input logits"
            if dist.get_rank() == 0:
                target = Path(out) / "logits" / f"updates_{completed_updates:03d}"
                target.mkdir(parents=True, exist_ok=False)
                save_file({"logits": logits.detach().cpu().contiguous(), "positions": positions.cpu(),
                           "labels": batch["labels"][:, positions].cpu()}, str(target / "logits.safetensors"))
                (target / "metadata.json").write_text(json.dumps({
                    "backend": backend, "completed_updates": completed_updates, "fixture": fixture_path,
                    "input_hashes": {k: tensor_hash(v) for k, v in fixture.items()},
                    "training": model.training, "logits_dtype": str(logits.dtype),
                    "native_automodel_text": os.environ.get("TRAIN_INFRA_NATIVE_TEXT") == "1",
                    "xtuner_hf_norm": os.environ.get("TRAIN_INFRA_XTUNER_HF_NORM") == "1",
                }, indent=2))
    finally:
        for value, before in buffers:
            value.copy_(before)
        for module, before in loads:
            module._cumulative_expert_load = before
    dist.barrier()
