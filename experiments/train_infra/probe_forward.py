"""Common-origin forward diagnostic: same XTuner pack, independent native model loaders."""

import argparse
import json
import os
import subprocess
from pathlib import Path

import torch
import torch.distributed as dist
from safetensors.torch import load_file, save_file

from experiments.train_infra.record import tensor_hash


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["xtuner", "automodel"], required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    assert dist.get_world_size() == 8
    args.out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(42)
    cfg = json.loads((args.reference / "resolved_config.json").read_text())
    fixture = load_file(str(args.fixture))
    batch = {k: v.cuda() for k, v in fixture.items()}
    positions = (batch["labels"][0] != -100).nonzero().flatten()
    assert positions.numel() > 0
    print(f"STAGE rank={rank} load_{args.backend}", flush=True)
    if args.backend == "xtuner":
        from xtuner.v1.config import FSDPConfig
        from xtuner.v1.data_proto import SequenceContext
        from xtuner.v1.loss import CELossConfig
        from xtuner.v1.model.compose.glm53 import Glm53BaseConfig

        # Resolved JSON includes read-only computed fields; use the native HF config constructor.
        model_cfg = Glm53BaseConfig.from_hf("/tmp/GLM-5.3-Flash-4layer")
        model_cfg.text_config.ep_size = 1
        model_cfg.text_config.dispatcher = "all2all"
        model_cfg.text_config.lm_loss_cfg = CELossConfig(mode="chunk", chunk_size=1024, loss_reduction="square")
        model_cfg.text_config.attention.sparse_mla_backend = "flash_mla_cudnn"
        model_cfg.text_config.attention.indexer_backend = "tilelang"
        assert model_cfg.text_config.mtp_config is None
        model_cfg.compile_cfg = model_cfg.text_config.compile_cfg = False
        model_cfg.vision_config.compile_cfg = model_cfg.projector_config.compile_cfg = False
        with torch.device("meta"):
            model = model_cfg.build()
        fsdp = FSDPConfig(ep_size=1, torch_compile=False, recompute_ratio=1.0,
                          vision_recompute_ratio=1.0, checkpoint_preserve_rng_state=False,
                          param_dtype=torch.bfloat16, reduce_dtype=torch.bfloat16)
        model.fully_shard(fsdp)
        model.from_hf("/tmp/GLM-5.3-Flash-4layer", strict=True)
        vision, projector, text = model.vision_tower, model.multi_modal_projector, model.language_model
        cu = batch["cu_seqlens"]
        maximum = int((cu[1:] - cu[:-1]).max())
        context = SequenceContext(
            input_ids=batch["input_ids"], cu_seq_lens_q=cu, cu_seq_lens_k=cu,
            max_length_q=maximum, max_length_k=maximum, device="cuda",
            num_padding=int(batch["padding_mask"].sum()), pixel_values=batch["pixel_values"].bfloat16(),
            image_grid_thw=batch["image_grid_thw"], mm_token_type_ids=batch["mm_token_type_ids"],
        )

        def forward():
            output = model(seq_ctx=context, loss_ctx=None)
            return output.logits[:, positions].detach().cpu().contiguous()
    else:
        from torch.distributed.pipelining import microbatch
        from torch.nn.attention.flex_attention import BlockMask

        microbatch.BlockMask = BlockMask
        from flash_attn import flash_attn_varlen_func
        from nemo_automodel import NeMoAutoModelForImageTextToText
        from nemo_automodel.components.distributed import activation_checkpointing, parallelizer
        from nemo_automodel.components.distributed.config import DistributedSetup, FSDP2Config
        from nemo_automodel.components.distributed.mesh import ParallelismSizes
        from nemo_automodel.components.models.common import BackendConfig, cudnn_sparse_attention
        from nemo_automodel.components.models.glm5_next import vision as native_vision
        from nemo_automodel.components.models.glm5_next.model import Glm5NextForConditionalGeneration
        from experiments.train_infra.run_automodel import register_glm_layers, register_flash_mla_compat
        from experiments.train_infra.vision_attention import install_flash_attention_2

        native_checkpoint = activation_checkpointing.checkpoint_wrapper

        def checkpoint(*a, **kw):
            kw["preserve_rng_state"] = cfg["fsdp_cfg"]["checkpoint_preserve_rng_state"]
            return native_checkpoint(*a, **kw)

        activation_checkpointing.checkpoint_wrapper = parallelizer.checkpoint_wrapper = checkpoint
        register_glm_layers(parallelizer, Glm5NextForConditionalGeneration)
        register_flash_mla_compat(cudnn_sparse_attention)
        install_flash_attention_2(native_vision, flash_attn_varlen_func)
        strategy = FSDP2Config(
            enable_compile=False, activation_checkpointing=True, defer_fsdp_grad_sync=False,
            reshard_after_forward=True,
            mp_policy=torch.distributed.fsdp.MixedPrecisionPolicy(
                param_dtype=torch.bfloat16, reduce_dtype=torch.bfloat16, output_dtype=torch.bfloat16,
                cast_forward_inputs=False),
        )
        setup = DistributedSetup.build(strategy=strategy, parallelism_sizes=ParallelismSizes(ep_size=1),
                                       activation_checkpointing=True)
        backend = BackendConfig(attn="cudnn", linear="torch", rms_norm="torch_fp32", experts="torch_mm",
                                dispatcher="torch", rope_fusion=False, gate_precision="float32",
                                fake_balanced_gate=False, enable_hf_state_dict_adapter=True,
                                enable_fsdp_optimizations=True)
        model = NeMoAutoModelForImageTextToText.from_pretrained(
            "/tmp/GLM-5.3-Flash-4layer-with-processor", dtype=torch.float32,
            _restore_loaded_dtype=False, force_hf=False, backend=backend, distributed_setup=setup,
            attn_implementation="sdpa", use_liger_kernel=False, use_sdpa_patching=False,
            has_packed_sequence=True, freeze_config={"freeze_modules": [{"glob": "*.self_attn.indexer"}]},
            text_config={"num_nextn_predict_layers": 0, "output_hidden_states": False},
        )
        assert isinstance(model, Glm5NextForConditionalGeneration)
        groups = parallelizer.get_model_layer_groups(model)
        assert len(groups["language"]) == 4 and len(groups["vision"]) == 24
        vision, projector, text = model.model.visual, None, model.model.language_model

        def forward():
            return model(
                input_ids=batch["input_ids"], pixel_values=batch["pixel_values"].bfloat16(),
                image_grid_thw=batch["image_grid_thw"], cu_seqlens=batch["cu_seqlens"],
                padding_mask=batch["padding_mask"], logits_to_keep=positions,
            ).logits.detach().cpu().contiguous()

    model.train()
    vision.requires_grad_(False)
    if projector is not None:
        projector.requires_grad_(False)
    assert vision.rotary_pos_emb.inv_freq.dtype == torch.float32
    assert not any(p.is_meta for p in model.parameters())
    visual_parameters = {}
    for module in ((vision, projector) if projector is not None else (vision,)):
        for name, param in module.named_parameters():
            clean = name.replace("_checkpoint_wrapped_module.", "").replace("_orig_mod.", "")
            keys = module.to_hf_key_list(clean) if args.backend == "xtuner" else ["model.visual." + clean]
            assert len(keys) == 1
            full = param.full_tensor() if hasattr(param, "full_tensor") else param
            if rank == 0:
                visual_parameters[keys[0]] = {"dtype": str(full.dtype), "shape": list(full.shape), "sha256": tensor_hash(full)}
            del full
    if rank == 0:
        assert len(visual_parameters) == 347
        (args.out / "visual_load_audit.json").write_text(json.dumps({
            "parameters": visual_parameters,
            "inv_freq": vision.rotary_pos_emb.inv_freq.detach().cpu().tolist(),
            "inv_freq_dtype": str(vision.rotary_pos_emb.inv_freq.dtype),
            "head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        }, indent=2))
    biases = {n: tensor_hash(v) for n, v in model.named_buffers() if n.endswith("e_score_correction_bias")}
    # The unobserved pass checks that capturing intermediate tensors does not alter the forward.
    print(f"STAGE rank={rank} unobserved_forward", flush=True)
    with torch.no_grad():
        baseline = forward()
    captures = {}
    handles = []

    def hook(name, output_key=None):
        def capture(module, inputs, output):
            if rank == 0:
                tensor = output[output_key] if output_key is not None else output
                assert isinstance(tensor, torch.Tensor), (name, type(tensor))
                captures[name] = tensor.detach().cpu().contiguous()
        return capture

    handles.append(text.embed_tokens.register_forward_hook(hook("embed_tokens")))
    handles.append(text.norm.register_forward_hook(hook("final_norm")))
    feature_module = projector if projector is not None else vision.merger
    handles.append(feature_module.register_forward_hook(hook("visual_features")))
    def capture_rope(module, inputs):
        if rank == 0:
            embeddings = inputs[3 if args.backend == "xtuner" else 2]
            captures["vision.rope_cos"] = embeddings[0].detach().cpu().contiguous()
            captures["vision.rope_sin"] = embeddings[1].detach().cpu().contiguous()
    handles.append(vision.blocks[0].register_forward_pre_hook(capture_rope))
    for module in ((vision, projector) if projector is not None else (vision,)):
        for name, child in module.named_modules():
            if name in ("patch_embed", "post_layernorm", "downsample", "merger") or (
                name.startswith("blocks.") and name.count(".") == 1
            ):
                handles.append(child.register_forward_hook(hook("vision." + name)))
    for i, layer in enumerate(text.layers.values()):
        handles.append(layer.register_forward_hook(hook(f"layer{i}", "hidden_states" if args.backend == "xtuner" else None)))
        if i == 0:
            def capture_input(module, inputs, kwargs):
                if rank == 0:
                    value = inputs[0] if inputs else kwargs["hidden_states"]
                    captures["layer0.input"] = value.detach().cpu().contiguous()
            handles.append(layer.register_forward_pre_hook(capture_input, with_kwargs=True))
            for name in ("input_layernorm", "post_attention_layernorm", "self_attn", "mlp"):
                child = getattr(layer, name)
                key = "projected_output" if args.backend == "xtuner" and name == "self_attn" else None
                handles.append(child.register_forward_hook(hook(f"layer0.{name}", key)))

    print(f"STAGE rank={rank} observed_forward", flush=True)
    with torch.no_grad():
        logits = forward()
    assert torch.equal(logits, baseline), "Forward observation changed supervised logits"
    assert biases == {n: tensor_hash(v) for n, v in model.named_buffers() if n.endswith("e_score_correction_bias")}
    assert torch.isfinite(logits).all()
    if rank == 0:
        captures.update(logits=logits, positions=positions.cpu(), labels=fixture["labels"][:, positions.cpu()])
        save_file(captures, str(args.out / "hidden_states.safetensors"))
        metadata = {"backend": args.backend, "torch": torch.__version__, "fixture": str(args.fixture),
                    "input_hashes": {k: tensor_hash(v) for k, v in fixture.items()},
                    "training": True, "freeze_vision_and_projector": True, "compile": False,
                    "cast_forward_inputs": False,
                    "no_grad": True, "world": 8, "ep": 1, "sp": 1, "optimizer_steps": 0,
                    "observation_preserves_logits": True, "router_bias_unchanged": True,
                    "experiment_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                    "diagnostic": "Common rank0 step0 full 8K pack repeated on all ranks; no training acceptance"}
        (args.out / "metadata.json").write_text(json.dumps(metadata, indent=2))
        print(json.dumps({"out": str(args.out), "logits": list(logits.shape)}), flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
