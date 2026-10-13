"""Frozen train-mode ViT diagnostic on an exact, complete captured train_infra pack."""

import argparse
import importlib.metadata
import json
import os
import subprocess
import sys
from pathlib import Path

import torch
import torch.distributed as dist
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

from experiments.train_infra.record import tensor_hash


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["xtuner", "automodel"], required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    torch.cuda.set_device(0)
    dist.init_process_group("nccl", init_method=f"file://{args.out.resolve()}/store", rank=0, world_size=1)
    fixture = load_file(str(args.fixture))
    pixels = fixture["pixel_values"].cuda().bfloat16()
    grid = fixture["image_grid_thw"].cuda()
    assert pixels.shape[0] == int(grid.prod(-1).sum())
    checkpoint = Path("/tmp/GLM-5.3-Flash-4layer")
    captures, parameters = {}, {}
    counters = {"flash_varlen": 0, "sdpa": 0}
    sdpa = torch.nn.functional.scaled_dot_product_attention

    def observed_sdpa(*a, **kw):
        counters["sdpa"] += 1
        return sdpa(*a, **kw)

    torch.nn.functional.scaled_dot_product_attention = observed_sdpa
    if args.backend == "xtuner":
        from xtuner.v1.config import FSDPConfig
        from xtuner.v1.model.compose.glm53 import Glm53BaseConfig
        import xtuner.v1.ops.attn_imp as attention

        native_flash = attention.flash_attn_varlen_func

        def observed_flash(*a, **kw):
            counters["flash_varlen"] += 1
            return native_flash(*a, **kw)

        attention.flash_attn_varlen_func = observed_flash
        cfg = Glm53BaseConfig.from_hf(checkpoint)
        cfg.vision_config.compile_cfg = cfg.projector_config.compile_cfg = False
        with torch.device("meta"):
            vision = cfg.vision_config.build()
            projector = cfg.projector_config.build()
        policy = FSDPConfig(ep_size=1, torch_compile=False, vision_recompute_ratio=0,
                            param_dtype=torch.bfloat16, reduce_dtype=torch.bfloat16)
        for module in (vision, projector):
            module.fully_shard(policy)
            _, unloaded, missing = module.from_hf(checkpoint, strict=False)
            assert not unloaded and not missing, (unloaded, missing)
        modules = (vision, projector)
        forward = lambda: projector(vision(pixels, grid))

        def canonical(module, name):
            keys = module.to_hf_key_list(module._clean_param_name(name))
            assert len(keys) == 1, keys
            return keys[0]
    else:
        from torch.distributed.pipelining import microbatch
        from torch.nn.attention.flex_attention import BlockMask

        microbatch.BlockMask = BlockMask
        from flash_attn import flash_attn_varlen_func
        from nemo_automodel.components.models.glm5_next import vision as native_vision
        from nemo_automodel.components.models.glm5_next.config import Glm5NextVisionConfig
        from experiments.train_infra.vision_attention import install_flash_attention_2

        def observed_flash(*a, **kw):
            counters["flash_varlen"] += 1
            return flash_attn_varlen_func(*a, **kw)

        install_flash_attention_2(native_vision, observed_flash)
        cfg = Glm5NextVisionConfig(**json.loads((checkpoint / "config.json").read_text())["vision_config"])
        cfg.torch_dtype = torch.float32
        vision = native_vision.Glm5NextVisionModel(cfg).float()
        mapping = json.loads((checkpoint / "model.safetensors.index.json").read_text())["weight_map"]
        state = {}
        for filename in sorted({v for k, v in mapping.items() if k.startswith("model.visual.")}):
            with safe_open(str(checkpoint / filename), framework="pt", device="cpu") as shard:
                for key in shard.keys():
                    if key.startswith("model.visual."):
                        state[key.removeprefix("model.visual.")] = shard.get_tensor(key)
        vision.load_state_dict(state, strict=True)
        del state
        vision.cuda()
        policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.bfloat16,
                                      cast_forward_inputs=False)
        for block in vision.blocks:
            fully_shard(block, mp_policy=policy)
        fully_shard(vision, mp_policy=policy)
        modules = (vision,)
        forward = lambda: vision(pixels, grid_thw=grid).pooler_output
        canonical = lambda module, name: "model.visual." + name

    def capture(name):
        def hook(module, inputs, output):
            assert name not in captures, name
            captures[name] = output.detach().cpu().contiguous()
        return hook

    for module in modules:
        module.train().requires_grad_(False)
        for name, param in module.named_parameters():
            full = param.full_tensor() if hasattr(param, "full_tensor") else param
            key = canonical(module, name)
            assert key not in parameters and param.dtype == torch.float32
            parameters[key] = {"shape": list(param.shape), "dtype": str(param.dtype), "sha256": tensor_hash(full)}
        for name, child in module.named_modules():
            if name in ("patch_embed", "post_layernorm", "downsample", "merger") or (
                name.startswith("blocks.") and name.count(".") == 1
            ):
                child.register_forward_hook(capture(name))
    assert len(parameters) == 347
    assert vision.rotary_pos_emb.inv_freq.dtype == torch.float32
    with torch.no_grad(), torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                            torch.profiler.ProfilerActivity.CUDA]) as profile:
        output = forward()
    torch.cuda.synchronize()
    assert torch.isfinite(output).all()
    assert counters == {"flash_varlen": 24, "sdpa": 0}, counters
    assert len(captures) == 28, list(captures)
    captures["features"] = output.detach().cpu().contiguous()
    save_file(captures, str(args.out / "hidden_states.safetensors"))
    events = [{"name": e.key, "count": e.count} for e in profile.key_averages()
              if "flash" in e.key.lower() or "attention" in e.key.lower()]
    metadata = {"backend": args.backend, "python": sys.executable, "torch": torch.__version__,
                "flash_attn": importlib.metadata.version("flash-attn"), "parameters": parameters,
                "inputs": {k: {"sha256": tensor_hash(v), "shape": list(v.shape)} for k, v in fixture.items()},
                "counters": counters, "profiler_events": events, "training": True, "frozen": True,
                "compile": False, "compute_dtype": "bfloat16", "master_dtype": "float32",
                "reduce_dtype": "bfloat16", "deterministic": False, "optimizer_steps": 0,
                "diagnostic": "world1/no_grad/full 8K pack pixels; not a reference training pass",
                "xtuner_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()}
    (args.out / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps({"out": str(args.out), "counters": counters, "features": list(output.shape)}), flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
