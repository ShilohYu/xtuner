"""Native NeMo GLM training on hash-verified XTuner packs, with matched losses."""

import argparse
import inspect
import importlib.metadata
import json
import os
from contextlib import ExitStack
from dataclasses import asdict
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F

from experiments.train_infra.replay import Replay
from xtuner.v1.utils.profile import timer


def register_glm_layers(parallelizer, model_class, *, conv_fp32=False):
    native_layer_specs = parallelizer._get_model_layer_group_specs

    def matching_layer_specs():
        specs = native_layer_specs()
        glm_groups = {"language": ["model.language_model.layers"], "vision": ["model.visual.blocks"]}
        specs[model_class] = glm_groups
        specs[model_class.__name__] = glm_groups
        specs[f"FSDP{model_class.__name__}"] = glm_groups
        return specs

    # The checkout's generic heuristic selects only the 24-layer vision tower.
    parallelizer._get_model_layer_group_specs = matching_layer_specs

    class GlmPrecisionStrategy(parallelizer.DefaultParallelizationStrategy):
        # EP1's generic path otherwise casts the native FP32 holders to BF16.
        _fp32_compute_module_names = tuple(model_class._keep_in_fp32_modules_strict)
        if conv_fp32:
            _fp32_compute_module_names += ("q_conv1d", "k_conv1d", "v_conv1d")
        _apply_fsdp_sharding = parallelizer.Qwen3_5ParallelizationStrategy._apply_fsdp_sharding

    for name in (model_class.__name__, f"FSDP{model_class.__name__}"):
        parallelizer.PARALLELIZATION_STRATEGIES[name] = GlmPrecisionStrategy()


def register_flash_mla_compat(sparse_attention):
    native = sparse_attention._FLASH_MLA_SPARSE_FWD
    if "indexer_topk" in inspect.signature(native).parameters:
        return False

    def legacy_forward(*args, indexer_topk=0, **kwargs):
        # The installed pre-indexer API already consumes explicit sparse indices and lengths.
        assert indexer_topk == 0, "Legacy FlashMLA cannot implement a nonzero indexer_topk"
        return native(*args, **kwargs)

    sparse_attention._FLASH_MLA_SPARSE_FWD = legacy_forward
    return True


def main(precision_contexts):
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
    from nemo_automodel.components.models.common import cudnn_sparse_attention
    from nemo_automodel.components.models.glm5_next.model import Glm5NextForConditionalGeneration
    from nemo_automodel.components.models.glm5_next import vision
    from nemo_automodel.components.models.glm5_next import layers
    from flash_attn import flash_attn_varlen_func
    from experiments.train_infra.vision_attention import install_flash_attention_2
    from experiments.train_infra.text_attention import install_shared_text_kernels, install_text_rms_norm, install_text_mlp
    from nemo_automodel.components.training.utils import (
        prepare_for_final_backward, prepare_for_grad_accumulation,
    )
    from xtuner.v1.loss.moe_loss import BalancingLossConfig

    from nemo_automodel.components.distributed import activation_checkpointing, parallelizer

    native_checkpoint_wrapper = activation_checkpointing.checkpoint_wrapper

    def matching_checkpoint_wrapper(*args, **kwargs):
        kwargs["preserve_rng_state"] = cfg["fsdp_cfg"]["checkpoint_preserve_rng_state"]
        return native_checkpoint_wrapper(*args, **kwargs)

    activation_checkpointing.checkpoint_wrapper = matching_checkpoint_wrapper
    parallelizer.checkpoint_wrapper = matching_checkpoint_wrapper
    conv_fp32 = os.environ.get("TRAIN_INFRA_ALIGNED_PRECISION") == "1"
    register_glm_layers(parallelizer, Glm5NextForConditionalGeneration, conv_fp32=conv_fp32)
    conv_precision = {}
    if conv_fp32:
        from experiments.train_infra.conv_precision import automodel_conv_precision
        conv_precision = precision_contexts.enter_context(automodel_conv_precision())
    legacy_flash_mla = register_flash_mla_compat(cudnn_sparse_attention)
    assert os.environ.get("XTUNER_USE_FA3", "0") == "0", "Reference ViT uses FlashAttention 2"
    assert cfg["model_cfg"].get("freeze_vision") or os.environ.get("XTUNER_DETERMINISTIC", "false") != "true", (
        "Fixed Triton configuration currently requires frozen vision; the AM vision adapter uses nondeterministic FA2 backward"
    )
    install_flash_attention_2(vision, flash_attn_varlen_func)
    native_text = os.environ.get("TRAIN_INFRA_NATIVE_TEXT") == "1"
    shared_text_kernels = {"mode": "native AutoModel"} if native_text else install_shared_text_kernels(layers)

    strategy = FSDP2Config(
        enable_compile=cfg["fsdp_cfg"]["torch_compile"], activation_checkpointing=True,
        defer_fsdp_grad_sync=False, reshard_after_forward=True,
        mp_policy=torch.distributed.fsdp.MixedPrecisionPolicy(
            param_dtype=torch.bfloat16, reduce_dtype=torch.bfloat16, output_dtype=torch.bfloat16,
            cast_forward_inputs=False,
        ),
    )
    setup = DistributedSetup.build(strategy=strategy, parallelism_sizes=ParallelismSizes(ep_size=1),
                                   activation_checkpointing=True)
    backend = BackendConfig(attn="cudnn", linear="torch", rms_norm="torch_fp32",
                            experts="torch_mm", dispatcher="torch", rope_fusion=False,
                            gate_precision="float32", fake_balanced_gate=False,
                            enable_hf_state_dict_adapter=True, enable_fsdp_optimizations=True)
    print(f"STAGE rank={rank} load_model", flush=True)
    frozen_modules = [{"glob": "*.self_attn.indexer"}]
    if cfg["model_cfg"].get("freeze_vision"):
        frozen_modules.append({"glob": "model.visual"})
    model = NeMoAutoModelForImageTextToText.from_pretrained(
        "/tmp/GLM-5.3-Flash-4layer-with-processor", dtype=torch.float32,
        _restore_loaded_dtype=False, force_hf=False, backend=backend,
        distributed_setup=setup, attn_implementation="sdpa",
        use_liger_kernel=False, use_sdpa_patching=False, has_packed_sequence=True,
        freeze_config={"freeze_modules": frozen_modules},
        text_config={"num_nextn_predict_layers": 0, "output_hidden_states": False},
    ).train()
    assert isinstance(model, Glm5NextForConditionalGeneration), type(model)
    if not native_text:
        shared_text_kernels["rms_norm_modules"] = install_text_rms_norm(model.model.language_model)
        shared_text_kernels["bf16_clamped_mlp_modules"] = install_text_mlp(model.model.language_model)
    aligned_precision = {}
    if os.environ.get("TRAIN_INFRA_ALIGNED_PRECISION") == "1":
        assert native_text and os.environ.get("TRAIN_INFRA_AM_HF_GATED_NORM") != "1"
        from experiments.train_infra.precision_profile import (
            automodel_native_core_precision, native_dsa_precision, automodel_moe_reduction_precision,
        )
        text = model.model.language_model
        aligned_precision["core"] = precision_contexts.enter_context(
            automodel_native_core_precision(text, routed_activation=True, training=True))
        aligned_precision["dsa"] = precision_contexts.enter_context(native_dsa_precision(text, "automodel", training=True))
        aligned_precision["moe"] = precision_contexts.enter_context(automodel_moe_reduction_precision(text, training=True))
    gated_norm_fix = []
    if os.environ.get("TRAIN_INFRA_AM_HF_GATED_NORM") == "1":
        assert native_text, "The single-variable gated-norm experiment requires native AM text"
        from experiments.train_infra.text_attention import install_automodel_hf_gated_norm
        gated_norm_fix = install_automodel_hf_gated_norm(model.model.language_model)
        assert len(gated_norm_fix) == 2, gated_norm_fix
    if cfg["model_cfg"].get("freeze_vision"):
        assert not any(p.requires_grad for p in model.model.visual.parameters())
    assert not any(p.is_meta for p in model.parameters())
    groups = parallelizer.get_model_layer_groups(model)
    assert len(groups.get("language", [])) == 4 and len(groups.get("vision", [])) == 24, groups
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import CheckpointWrapper

    checkpoint_counts = {name: sum(isinstance(module, CheckpointWrapper) for layer in layers
                                   for module in layer.modules()) for name, layers in groups.items()}
    assert checkpoint_counts.get("language", 0) >= 8, checkpoint_counts
    if not cfg["model_cfg"].get("freeze_vision"):
        assert checkpoint_counts.get("vision", 0) >= 48, checkpoint_counts
    parameters = [p for p in model.parameters() if p.requires_grad]
    assert all(p.dtype == torch.float32 for p in parameters), "FP32 master weights required"
    from experiments.train_infra.muon_alignment import build_muon, clip_muon_gradients, optimizer_manifest

    optimizer = build_muon(model, cfg["optim_cfg"])
    (args.out / f"optimizer_groups_rank{rank}.json").write_text(json.dumps(optimizer_manifest(optimizer), indent=2))
    total = cfg["total_step"]
    assert total is not None, "This gate uses the explicit reference step schedule"
    warmup_setting = cfg["lr_cfg"]["warmup_ratio"]
    warmup = int(total * warmup_setting) if warmup_setting < 1 else int(warmup_setting)
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        [torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: step / warmup if step < warmup else 1),
         torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total - warmup, eta_min=cfg["lr_cfg"]["lr_min"])],
        [warmup],
    )
    ce = ChunkedCrossEntropy(chunk_len=1024, reduction="sum")
    balance_settings = cfg["model_cfg"]["text_config"]["balancing_loss_cfg"]
    balance_cfg = BalancingLossConfig(**balance_settings) if balance_settings is not None else None
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
            if capture["context"] is not None:
                capture["context"].accumulate(router_weights=scores)
            capture["loads"].append(torch.bincount(indices.reshape(-1), minlength=scores.shape[-1]))

    gates = []
    for name, module in model.named_modules():
        if name.endswith(".gate") and hasattr(module, "e_score_correction_bias"):
            module.register_forward_hook(gate_hook, with_kwargs=True)
            gates.append(name)
    assert len(gates) == 3, gates
    manifest = {"reference": str(args.reference), "backend": asdict(backend),
                "strategy": asdict(strategy), "torch": torch.__version__,
                "model_class": type(model).__name__, "auxiliary_loss": balance_cfg.model_dump() if balance_cfg else None,
                "master_dtype": "float32", "reduce_dtype": "bfloat16",
                "shared_text_kernels": shared_text_kernels,
                "hf_gated_norm_modules": gated_norm_fix,
                "aligned_precision": aligned_precision,
                "conv_precision": conv_precision,
                "checkpoint_preserve_rng_state": cfg["fsdp_cfg"]["checkpoint_preserve_rng_state"],
                "legacy_flash_mla_zero_indexer_adapter": legacy_flash_mla,
                "vision_attention": {"implementation": "flash_attention_2_varlen",
                                     "version": importlib.metadata.version("flash-attn"),
                                     "causal": False, "deterministic": False,
                                     "dropout": model.config.vision_config.attention_dropout},
                "layer_groups": {name: len(layers) for name, layers in groups.items()},
                "checkpoint_wrappers": checkpoint_counts, "auxiliary_gates": gates}
    (args.out / f"environment_rank{rank}.json").write_text(json.dumps(manifest, default=str, indent=2))
    if os.environ.get("TRAIN_INFRA_LOGITS_FIXTURE"):
        from experiments.train_infra.training_logits import capture_logits
        capture_logits(model, "automodel", args.out, 0)
    for entry in replay.entries:
        gradients = None
        if os.environ.get("TRAIN_INFRA_GRAD_DIAGNOSTIC") == "1":
            from experiments.train_infra.gradient_diagnostic import BoundaryGradients
            gradients = BoundaryGradients(model, "automodel", args.out, entry["step"])
        print(f"STAGE rank={rank} step={entry['step']} replay_inputs", flush=True)
        batches = replay.batches(entry)
        train_tokens = sum(int((~batch["padding_mask"]).sum()) for batch in batches)
        train_padded_tokens = sum(batch["input_ids"].numel() for batch in batches)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        timing = {}
        with timer("train_step_seconds", timing):
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
                if gradients is not None and rank == 0:
                    from safetensors.torch import save_file
                    assert len(batches) == 1, "Single-sample diagnostic only"
                    target = args.out / "loss_inputs"
                    target.mkdir(exist_ok=True)
                    save_file({"labels": labels.cpu().contiguous(), "weights": weights.cpu().contiguous()},
                              str(target / f"step_{entry['step']:03d}.safetensors"))
                batch.pop("mm_token_type_ids", None)
                positions = (labels[0] != -100).nonzero().flatten()
                context = balance_cfg.build() if balance_cfg else None
                if context is not None:
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
                                       non_pad_token=int((~batch["padding_mask"]).sum())) if context is not None else value.new_zeros(())
                with (args.out / f"forward_losses_rank{rank}.jsonl").open("a") as stream:
                    stream.write(json.dumps({"step": entry["step"], "microbatch": index,
                                             "ce_local": float(value.detach()), "balancing_global": float(aux.detach())}) + "\n")
                print(f"STAGE rank={rank} step={entry['step']} backward", flush=True)
                (value * dist.get_world_size() + aux).backward()
                local_ce += value.detach()
                local_aux += aux.detach()
            if gradients is not None:
                gradients.close()
            dist.all_reduce(local_ce)
            norm = clip_muon_gradients(optimizer, model, cfg["optim_cfg"]["max_grad_norm"])
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
            torch.cuda.synchronize()
        row = {"step": entry["step"], "loss": float(local_ce + local_aux),
               "ce_loss": float(local_ce), "balancing_loss": float(local_aux),
               "grad_norm": float(norm), "lr": lr, "probe": name,
               "update_max_abs": float((local.detach() - before).abs().max()),
               "peak_memory_bytes": torch.cuda.max_memory_allocated(),
               "peak_reserved_memory_bytes": torch.cuda.max_memory_reserved(),
               **timing, "train_tokens": train_tokens, "train_padded_tokens": train_padded_tokens,
               "train_tgs_local": train_tokens / timing["train_step_seconds"],
               "gradient_diagnostic_enabled": gradients is not None,
               "timing_scope": "prepared_cpu_pack_through_optimizer; excludes_extra_logits"}
        with (args.out / f"metrics_rank{rank}.jsonl").open("a") as stream:
            stream.write(json.dumps(row) + "\n")
        del batches, batch, output, context
        if os.environ.get("TRAIN_INFRA_LOGITS_FIXTURE"):
            capture_logits(model, "automodel", args.out, entry["step"] + 1)
    dist.destroy_process_group()


if __name__ == "__main__":
    with ExitStack() as precision_contexts:
        main(precision_contexts)
