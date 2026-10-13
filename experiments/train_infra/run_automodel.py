"""Native NeMo GLM training on hash-verified XTuner packs, with matched losses."""

import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F

from experiments.train_infra.replay import Replay


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    rank = int(os.environ["RANK"])
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl")
    assert dist.get_world_size() == 8
    torch.manual_seed(42)
    args.out.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((args.reference / "resolved_config.json").read_text())
    assert cfg["sp_size"] == 1 and cfg["fsdp_cfg"]["ep_size"] == 1
    replay = Replay(args.reference, rank)
    assert replay.entries, "Reference never reached input construction"

    # Compatibility for the installed torch 2.9 pipeline API; no environment changes.
    from torch.distributed.pipelining import microbatch
    from torch.nn.attention.flex_attention import BlockMask

    if not hasattr(microbatch, "BlockMask"):
        microbatch.BlockMask = BlockMask
    from nemo_automodel import NeMoAutoModelForImageTextToText
    from nemo_automodel.components.distributed.config import DistributedSetup, FSDP2Config
    from nemo_automodel.components.distributed.mesh import ParallelismSizes
    from nemo_automodel.components.loss.chunked_ce import ChunkedCrossEntropy
    from nemo_automodel.components.models.common import BackendConfig
    from nemo_automodel.components.models.glm5_next.model import Glm5NextForConditionalGeneration
    from nemo_automodel.components.training.utils import (
        prepare_for_final_backward, prepare_for_grad_accumulation, scale_grads_and_clip_grad_norm,
    )
    from xtuner.v1.loss.moe_loss import BalancingLossConfig

    from nemo_automodel.components.distributed import activation_checkpointing, parallelizer

    native_checkpoint_wrapper = activation_checkpointing.checkpoint_wrapper

    def matching_checkpoint_wrapper(*args, **kwargs):
        kwargs["preserve_rng_state"] = cfg["fsdp_cfg"]["checkpoint_preserve_rng_state"]
        return native_checkpoint_wrapper(*args, **kwargs)

    activation_checkpointing.checkpoint_wrapper = matching_checkpoint_wrapper
    parallelizer.checkpoint_wrapper = matching_checkpoint_wrapper
    native_layer_specs = parallelizer._get_model_layer_group_specs

    def matching_layer_specs():
        specs = native_layer_specs()
        glm_groups = {"language": ["model.language_model.layers"], "vision": ["model.visual.blocks"]}
        specs[Glm5NextForConditionalGeneration] = glm_groups
        specs["FSDPGlm5NextForConditionalGeneration"] = glm_groups
        return specs

    # The checkout's generic heuristic selects only the 24-layer vision tower.
    parallelizer._get_model_layer_group_specs = matching_layer_specs

    strategy = FSDP2Config(
        enable_compile=cfg["fsdp_cfg"]["torch_compile"], activation_checkpointing=True,
        defer_fsdp_grad_sync=False, reshard_after_forward=True,
        mp_policy=torch.distributed.fsdp.MixedPrecisionPolicy(
            param_dtype=torch.bfloat16, reduce_dtype=torch.bfloat16, output_dtype=torch.bfloat16,
        ),
    )
    setup = DistributedSetup.build(strategy=strategy, parallelism_sizes=ParallelismSizes(ep_size=1),
                                   activation_checkpointing=True)
    backend = BackendConfig(attn="cudnn", linear="torch", rms_norm="torch_fp32",
                            experts="torch_mm", dispatcher="torch", rope_fusion=False,
                            gate_precision="float32", fake_balanced_gate=False,
                            enable_hf_state_dict_adapter=True, enable_fsdp_optimizations=True)
    print(f"STAGE rank={rank} load_model", flush=True)
    model = NeMoAutoModelForImageTextToText.from_pretrained(
        "/tmp/GLM-5.3-Flash-4layer-with-processor", dtype=torch.float32,
        _restore_loaded_dtype=False, force_hf=False, backend=backend,
        distributed_setup=setup, attn_implementation="sdpa",
        use_liger_kernel=False, use_sdpa_patching=False, has_packed_sequence=True,
        freeze_config={"freeze_modules": [{"glob": "*.self_attn.indexer"}]},
        text_config={"num_nextn_predict_layers": 0, "output_hidden_states": False},
    ).train()
    assert isinstance(model, Glm5NextForConditionalGeneration), type(model)
    assert not any(p.is_meta for p in model.parameters())
    groups = parallelizer.get_model_layer_groups(model)
    assert len(groups.get("language", [])) == 4 and len(groups.get("vision", [])) == 24, groups
    parameters = [p for p in model.parameters() if p.requires_grad]
    assert all(p.dtype == torch.float32 for p in parameters), "FP32 master weights required"
    optimizer = torch.optim.AdamW(parameters, lr=2e-5, betas=(0.9, 0.95), eps=1e-8,
                                 weight_decay=0.05, foreach=False)
    total = cfg["total_step"]
    assert total is not None, "This gate uses the explicit reference step schedule"
    warmup = int(total * 0.1)
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        [torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: step / warmup if step < warmup else 1),
         torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total - warmup, eta_min=1e-6)],
        [warmup],
    )
    ce = ChunkedCrossEntropy(chunk_len=1024, reduction="sum")
    balance_cfg = BalancingLossConfig(**cfg["model_cfg"]["text_config"]["balancing_loss_cfg"])
    capture = {"enabled": False, "context": None, "loads": []}

    def gate_hook(gate, inputs, kwargs, output):
        # Match XTuner's normalized sigmoid + correction-bias distribution.
        # Execute the same tensor operations during AC replay, but record only the main forward.
        hidden = inputs[0]
        logits = F.linear(hidden.float(), gate.weight.float(), bias=None)
        scores = logits.sigmoid() + gate.e_score_correction_bias.float()
        scores = scores / scores.sum(-1, keepdim=True)
        mask = kwargs.get("token_mask", inputs[1] if len(inputs) > 1 else None)
        scores = scores.reshape(-1, scores.shape[-1])
        indices = output[1].reshape(-1, output[1].shape[-1])
        if mask is not None:
            valid = mask.reshape(-1).bool()
            scores, indices = scores[valid], indices[valid]
        if capture["enabled"]:
            capture["context"].accumulate(router_weights=scores)
            capture["loads"].append(torch.bincount(indices.reshape(-1), minlength=scores.shape[-1]))

    for name, module in model.named_modules():
        if name.endswith(".mlp.gate"):
            module.register_forward_hook(gate_hook, with_kwargs=True)
    manifest = {"reference": str(args.reference), "backend": asdict(backend),
                "strategy": asdict(strategy), "torch": torch.__version__,
                "model_class": type(model).__name__, "auxiliary_loss": balance_cfg.model_dump(),
                "master_dtype": "float32", "reduce_dtype": "bfloat16",
                "checkpoint_preserve_rng_state": cfg["fsdp_cfg"]["checkpoint_preserve_rng_state"],
                "layer_groups": {name: len(layers) for name, layers in groups.items()}}
    (args.out / f"environment_rank{rank}.json").write_text(json.dumps(manifest, default=str, indent=2))
    for entry in replay.entries:
        torch.cuda.reset_peak_memory_stats()
        print(f"STAGE rank={rank} step={entry['step']} replay_inputs", flush=True)
        batches = replay.batches(entry)
        denominator = sum(batch["loss_weights"].sum() for batch in batches).cuda()
        dist.all_reduce(denominator)
        prepare_for_grad_accumulation([model])
        local_ce = torch.zeros((), device="cuda")
        local_aux = torch.zeros((), device="cuda")
        for index, batch in enumerate(batches):
            if index == len(batches) - 1:
                prepare_for_final_backward([model])
            batch = {key: tensor.cuda() for key, tensor in batch.items()}
            labels, weights = batch.pop("labels"), batch.pop("loss_weights") / (denominator + 1e-12)
            batch.pop("mm_token_type_ids", None)
            positions = (labels[0] != -100).nonzero().flatten()
            context = balance_cfg.build()
            context._batch_size = len(batches)
            capture.update(enabled=True, context=context, loads=[])
            print(f"STAGE rank={rank} step={entry['step']} forward", flush=True)
            output = model(logits_to_keep=positions, **batch)
            capture["enabled"] = False
            value = ce(output.logits, labels[:, positions], loss_weights=weights[:, positions])
            assert capture["loads"], "Native MoE gate capture did not run"
            loads = torch.stack(capture["loads"])
            global_loads = loads.clone()
            dist.all_reduce(global_loads)
            aux = context.finalize(tokens_per_expert_local=loads, tokens_per_expert_global=global_loads,
                                   n_routed_experts=288, num_experts_per_tok=8,
                                   non_pad_token=int((~batch["padding_mask"]).sum()))
            print(f"STAGE rank={rank} step={entry['step']} backward", flush=True)
            (value * dist.get_world_size() + aux).backward()
            local_ce += value.detach()
            local_aux += aux.detach()
        dist.all_reduce(local_ce)
        norm = scale_grads_and_clip_grad_norm(1.0, [model], device_mesh=setup.mesh_context.device_mesh,
                                              foreach=False, dp_group_size=8)
        assert torch.isfinite(norm) and torch.isfinite(local_ce)
        name, probe = next((name, p) for name, p in model.named_parameters()
                           if p.requires_grad and p.dtype == torch.float32 and p.numel() <= 16384)
        local = probe.to_local() if hasattr(probe, "to_local") else probe
        before = local.detach().clone()
        lr = optimizer.param_groups[0]["lr"]
        print(f"STAGE rank={rank} step={entry['step']} optimizer lr={lr}", flush=True)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        model.update_moe_gate_bias()
        scheduler.step()
        row = {"step": entry["step"], "loss": float(local_ce + local_aux),
               "ce_loss": float(local_ce), "balancing_loss": float(local_aux),
               "grad_norm": float(norm), "lr": lr, "probe": name,
               "update_max_abs": float((local.detach() - before).abs().max()),
               "peak_memory_bytes": torch.cuda.max_memory_allocated()}
        with (args.out / f"metrics_rank{rank}.jsonl").open("a") as stream:
            stream.write(json.dumps(row) + "\n")
        del batches, batch, output, context
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
