"""Replay dense MLP on a common recorded input, isolating SwiGLU precision."""

import argparse
import json
import subprocess
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file
from torch.distributed.pipelining import microbatch
from torch.nn.attention.flex_attention import BlockMask

microbatch.BlockMask = BlockMask
from nemo_automodel.components.moe.layers import MLP
from xtuner.v1.module.decoder_layer.dense_decoder_layer import DenseMLP
from experiments.train_infra.plot_forward import errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--forward", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    checkpoint = Path("/tmp/GLM-5.3-Flash-4layer")
    cfg = json.loads((checkpoint / "config.json").read_text())["text_config"]
    mapping = json.loads((checkpoint / "model.safetensors.index.json").read_text())["weight_map"]

    def capture(backend, key):
        with safe_open(str(args.forward / backend / "hidden_states.safetensors"), framework="pt") as f:
            return f.get_tensor(key).cuda()

    valid = ~load_file(str(args.fixture))["padding_mask"][0].cuda().bool()
    report = {"head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
              "scope": "Same full 8K layer0 input/checkpoint weights; single GPU, no_grad"}

    def compare(name, x, y):
        report[name] = {"all": errors(x, y), "real_tokens": errors(x[:, valid], y[:, valid])}
        print(name, report[name]["real_tokens"], flush=True)

    x = capture("xtuner", "layer0.post_attention_layernorm")
    compare("actual_mlp_inputs", x, capture("automodel", "layer0.post_attention_layernorm"))
    assert torch.equal(x[:, valid], capture("automodel", "layer0.post_attention_layernorm")[:, valid])
    xt = DenseMLP(hidden_size=cfg["hidden_size"], intermediate_size=cfg["intermediate_size"],
                  hidden_act=cfg["hidden_act"], swiglu_limit=cfg["swiglu_limit"]).cuda().bfloat16()
    am = MLP(cfg["hidden_size"], cfg["intermediate_size"], "torch", dtype=torch.bfloat16,
             swiglu_limit=cfg["swiglu_limit"]).cuda()
    with torch.no_grad():
        for name, p in xt.named_parameters():
            key = "model.language_model.layers.0.mlp." + name
            with safe_open(str(checkpoint / mapping[key]), framework="pt") as f:
                p.copy_(f.get_tensor(key))
            dict(am.named_parameters())[name].copy_(p)
        stages = [{}, {}]
        for i, module in enumerate((xt, am)):
            for name in ("gate_proj", "up_proj", "down_proj"):
                def hook(child, inputs, output, i=i, name=name):
                    stages[i][name] = output.detach()
                    if name == "down_proj":
                        stages[i]["activated"] = inputs[0].detach()
                getattr(module, name).register_forward_hook(hook)
        xo, yo = xt(x), am(x)
        for name in stages[0]:
            compare("same_input." + name, stages[0][name], stages[1][name])
        compare("xtuner_runtime_replay", xo, capture("xtuner", "layer0.mlp"))
        compare("automodel_runtime_replay", yo, capture("automodel", "layer0.mlp"))
        assert torch.equal(xo[:, valid], capture("xtuner", "layer0.mlp")[:, valid])
        assert torch.equal(yo[:, valid], capture("automodel", "layer0.mlp")[:, valid])
        bf16_output = am.down_proj(xt.act_fn(stages[1]["gate_proj"], stages[1]["up_proj"]))
        compare("same_input_bf16_activation", xo, bf16_output)
        assert torch.equal(xo, bf16_output), "Sharing the activation did not eliminate the MLP difference"
    (args.out / "comparison.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
