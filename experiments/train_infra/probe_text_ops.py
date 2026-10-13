"""Replay native mHC, RMSNorm and KDA on the same recorded layer-zero inputs."""

import argparse
import json
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file

from experiments.train_infra.plot_forward import errors
from torch.distributed.pipelining import microbatch
from torch.nn.attention.flex_attention import BlockMask

microbatch.BlockMask = BlockMask
from nemo_automodel.components.models.glm5_next.config import Glm5NextTextConfig
from nemo_automodel.components.models.glm5_next.layers import (
    Glm5NextHyperConnection, Glm5NextLinearAttention, Glm5NextRMSNorm,
)
from xtuner.v1.data_proto import SequenceContext
from xtuner.v1.model.moe.glm53 import Glm53TextMoEConfig
from xtuner.v1.module.decoder_layer.mhc import hc_pre
from xtuner.v1.module.rms_norm import RMSNorm


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--forward", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--shared-text-kernels", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(exist_ok=False, parents=True)
    model_path = Path("/tmp/GLM-5.3-Flash-4layer")
    cfg = Glm5NextTextConfig(**json.loads((model_path / "config.json").read_text())["text_config"])
    mapping = json.loads((model_path / "model.safetensors.index.json").read_text())["weight_map"]

    def weight(suffix):
        key = "model.language_model.layers.0." + suffix
        with safe_open(str(model_path / mapping[key]), framework="pt") as f:
            return f.get_tensor(key).cuda()

    def captured(backend, key):
        with safe_open(str(args.forward / backend / "hidden_states.safetensors"), framework="pt") as f:
            return f.get_tensor(key).cuda()

    fixture = {k: v.cuda() for k, v in load_file(str(args.fixture)).items()}
    valid = ~fixture["padding_mask"][0].bool()
    report = {}
    if args.shared_text_kernels:
        from nemo_automodel.components.models.glm5_next import layers
        from experiments.train_infra.text_attention import install_shared_text_kernels
        report["shared_text_kernels"] = install_shared_text_kernels(layers)

    def compare(name, x, y):
        report[name] = {"all": errors(x, y), "real_tokens": errors(x[:, valid], y[:, valid])}
        print(name, report[name]["real_tokens"], flush=True)

    streams = captured("xtuner", "layer0.input")
    assert torch.equal(streams, captured("automodel", "layer0.input"))
    hc = Glm5NextHyperConnection(cfg).cuda().bfloat16()
    hc._fp32_params.float()
    with torch.no_grad():
        hc.fn.copy_(weight("hc_attn_fn"))
        hc.base.copy_(weight("hc_attn_base"))
        hc.scale.copy_(weight("hc_attn_scale"))
        xt_collapsed, _, _ = hc_pre(streams, hc.fn, hc.scale, hc.base, cfg.hc_mult,
                                   cfg.hc_sinkhorn_iters, cfg.hc_eps, norm_eps=cfg.rms_norm_eps)
        _, _, am_collapsed = hc(streams)
        compare("hc_collapsed", xt_collapsed, am_collapsed)
        norms = [RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps).cuda().bfloat16(),
                 Glm5NextRMSNorm(cfg.hidden_size, cfg.rms_norm_eps, torch.bfloat16).cuda()]
        if args.shared_text_kernels:
            from experiments.train_infra.text_attention import install_text_rms_norm
            install_text_rms_norm(norms[1])
        for norm in norms:
            norm.weight.copy_(weight("input_layernorm.weight"))
        x, y = norms[0](xt_collapsed), norms[1](am_collapsed)
        compare("native_input_norm", x, y)
        compare("same_input_norm", norms[0](xt_collapsed), norms[1](xt_collapsed))
        compare("xtuner_runtime_replay", x, captured("xtuner", "layer0.input_layernorm"))
        compare("automodel_runtime_replay", y, captured("automodel", "layer0.input_layernorm"))
        hc._fp32_params.bfloat16()
        _, _, rounded_collapsed = hc(streams)
        compare("bf16_holder_runtime_replay", norms[1](rounded_collapsed),
                captured("automodel", "layer0.input_layernorm"))

        xt_cfg = Glm53TextMoEConfig.from_hf(model_path)
        xt = xt_cfg.linear_attention.build(hidden_size=cfg.hidden_size).cuda().bfloat16()
        xt.A_log.data = weight("self_attn.A_log").float()
        xt.dt_bias.data = weight("self_attn.dt_bias").float()
        am = Glm5NextLinearAttention(cfg, 0).cuda().bfloat16()
        am._fp32_params.float()
        for model in (xt, am):
            for name, p in model.named_parameters():
                p.copy_(weight("self_attn." + name.removeprefix("_fp32_params.")))
            model.train()
        context = SequenceContext(input_ids=fixture["input_ids"], cu_seq_lens_q=fixture["cu_seqlens"],
                                  cu_seq_lens_k=fixture["cu_seqlens"], device="cuda",
                                  max_length_q=int(fixture["cu_seqlens"].diff().max()),
                                  max_length_k=int(fixture["cu_seqlens"].diff().max()))
        captures = [{}, {}]
        for index, model in enumerate((xt, am)):
            for name in ("q_proj", "k_proj", "v_proj", "q_conv1d", "k_conv1d", "v_conv1d",
                         "f_a_proj", "f_b_proj", "b_proj", "g_a_proj", "g_b_proj", "o_norm", "o_proj"):
                def hook(module, inputs, output, name=name, index=index):
                    captures[index][name] = (output[0] if isinstance(output, tuple) else output).detach()
                    if name == "o_norm":
                        captures[index]["chunk_output"] = inputs[0].detach()
                getattr(model, name).register_forward_hook(hook)
        xo = xt(x, seq_ctx=context)["projected_output"]
        yo = am._core(x, cu_seqlens=fixture["cu_seqlens"])
        for name in captures[0]:
            compare("same_input_kda." + name, captures[0][name], captures[1][name])
        compare("same_input_kda.output", xo, yo)
        if args.shared_text_kernels:
            assert torch.equal(xt_collapsed, am_collapsed), "Same-input mHC still differs"
            assert torch.equal(x, y), "Same-input RMSNorm still differs"
            assert torch.equal(xo, yo), "Same-input native KDA still differs"
        report["diagnostic"] = "Single GPU, same full 8K layer0 inputs and checkpoint weights; no backward"
        (args.out / "comparison.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
