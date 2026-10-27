"""Fixed-input logits at every update; observation preserves RNG and router state."""

import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
from safetensors.torch import load_file, save_file

from experiments.train_infra.record import tensor_hash
from xtuner.v1.utils.profile import timer


@torch.no_grad()
def capture_logits(model, backend, out, completed_updates):
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    timing = {}
    with timer("logits_observation_seconds", timing):
        fixture_path = os.environ["TRAIN_INFRA_LOGITS_FIXTURE"]
        fixture = load_file(fixture_path)
        batch = {key: value.cuda() for key, value in fixture.items()}
        media = {key: batch[key] for key in ("pixel_values", "image_grid_thw") if key in batch}
        if "pixel_values" in media:
            media["pixel_values"] = media["pixel_values"].bfloat16()
        positions = (batch["labels"][0] != -100).nonzero().flatten()
        assert positions.numel()
        buffers = [(v, v.detach().clone()) for v in model.buffers()]
        loads = [(m, m._cumulative_expert_load) for m in model.modules() if hasattr(m, "_cumulative_expert_load")]
        hidden_states, handles = {}, []
        if completed_updates == 0 and os.environ.get("TRAIN_INFRA_FIXED_SAMPLE"):
            text = model.language_model if backend == "xtuner" else model.model.language_model
            def hook(name, key=None):
                def capture(module, inputs, output):
                    if dist.get_rank() == 0:
                        value = output[key] if key else output
                        hidden_states[name] = value.detach().cpu().contiguous()
                return capture
            for i, layer in text.layers.items():
                handles.append(layer.register_forward_hook(hook(f"layer{i}", "hidden_states" if backend == "xtuner" else None)))
            handles.append(text.norm.register_forward_hook(hook("final_norm")))
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
                        mm_token_type_ids=batch["mm_token_type_ids"], **media,
                    )
                    output = model(seq_ctx=ctx, loss_ctx=None)
                    logits = output.logits[:, positions]
                else:
                    output = model(
                        input_ids=batch["input_ids"], **media, cu_seqlens=batch["cu_seqlens"],
                        padding_mask=batch["padding_mask"], logits_to_keep=positions,
                    )
                    logits = output.logits
                assert torch.isfinite(logits).all(), "Nonfinite fixed-input logits"
                if dist.get_rank() == 0:
                    target = Path(out) / "logits" / f"updates_{completed_updates:03d}"
                    target.mkdir(parents=True, exist_ok=False)
                    if hidden_states:
                        save_file(hidden_states, str(Path(out) / "initial_hidden_states.safetensors"))
                    biases = {name: value.detach().cpu().contiguous() for name, value in model.named_buffers()
                              if name.endswith("e_score_correction_bias")}
                    if biases:
                        save_file(biases, str(target / "router_bias.safetensors"))
                    save_file({"logits": logits.detach().cpu().contiguous(), "positions": positions.cpu(),
                               "labels": batch["labels"][:, positions].cpu()}, str(target / "logits.safetensors"))
                    (target / "metadata.json").write_text(json.dumps({
                        "backend": backend, "completed_updates": completed_updates, "fixture": fixture_path,
                        "input_hashes": {k: tensor_hash(v) for k, v in fixture.items()},
                        "training": model.training, "logits_dtype": str(logits.dtype),
                        "native_automodel_text": os.environ.get("TRAIN_INFRA_NATIVE_TEXT") == "1" and os.environ.get("TRAIN_INFRA_ALIGNED_PRECISION") != "1",
                        "aligned_precision": os.environ.get("TRAIN_INFRA_ALIGNED_PRECISION") == "1",
                        "permute_compute_dtype": os.environ.get("PERMUTE_COMPUTE_DTYPE"),
                        "xtuner_hf_norm": os.environ.get("TRAIN_INFRA_XTUNER_HF_NORM") == "1",
                    }, indent=2))
        finally:
            for handle in handles:
                handle.remove()
            for value, before in buffers:
                value.copy_(before)
            for module, before in loads:
                module._cumulative_expert_load = before
        dist.barrier()
        torch.cuda.synchronize()
    record = {"completed_updates": completed_updates, **timing,
              "peak_memory_bytes": torch.cuda.max_memory_allocated(),
              "peak_reserved_memory_bytes": torch.cuda.max_memory_reserved(),
              "timing_scope": "extra_logits_forward_cpu_copy_dump_restore_and_rank_barrier"}
    with (Path(out) / f"observation_metrics_rank{dist.get_rank()}.jsonl").open("a") as stream:
        stream.write(json.dumps(record) + "\n")
