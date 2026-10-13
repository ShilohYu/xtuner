"""Compare captured forward tensors, preserving full logits and exporting per-token errors."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
import torch
import torch.nn.functional as F
from safetensors.torch import load_file

from experiments.train_infra.record import tensor_hash

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def errors(x, y):
    assert x.shape == y.shape, (x.shape, y.shape)
    xf, yf = x.float(), y.float()
    diff = yf - xf
    return {"shape": list(x.shape), "xtuner_dtype": str(x.dtype), "automodel_dtype": str(y.dtype),
            "bitwise_equal": x.dtype == y.dtype and torch.equal(x, y), "values_equal": torch.equal(xf, yf),
            "max_abs": diff.abs().max().item(),
            "relative_l2": (diff.norm() / xf.norm().clamp_min(1e-30)).item(),
            "unequal_fraction": (x != y).float().mean().item()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xtuner", type=Path, required=True)
    parser.add_argument("--automodel", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    a, b = (load_file(str(path / "hidden_states.safetensors")) for path in (args.xtuner, args.automodel))
    ma, mb = (json.loads((path / "metadata.json").read_text()) for path in (args.xtuner, args.automodel))
    inputs_key = "input_hashes" if "input_hashes" in ma else "inputs"
    assert ma[inputs_key] == mb[inputs_key], "Different model inputs"
    assert a.keys() == b.keys(), (a.keys(), b.keys())
    if "parameters" in ma:
        assert ma["parameters"] == mb["parameters"]
    report = {"xtuner": str(args.xtuner), "automodel": str(args.automodel),
              "inputs_equal": True, "diagnostic": ma.get("diagnostic"),
              "experiment_heads": [ma.get("experiment_head"), mb.get("experiment_head")],
              "stages": {key: errors(a[key], b[key]) for key in a if key not in ("positions", "labels")}}
    if (args.xtuner / "visual_load_audit.json").exists():
        audits = [json.loads((path / "visual_load_audit.json").read_text()) for path in (args.xtuner, args.automodel)]
        for key in ("parameters", "inv_freq", "inv_freq_dtype"):
            assert audits[0][key] == audits[1][key], f"Visual loading differs: {key}"
        report["visual_parameters_equal"] = len(audits[0]["parameters"])
    if "fixture" in ma:
        fixture = load_file(ma["fixture"])
        assert {k: tensor_hash(v) for k, v in fixture.items()} == ma["input_hashes"]
        masks = {"real_tokens": ~fixture["padding_mask"][0].bool(),
                 "supervised_tokens": fixture["labels"][0] != -100,
                 "padding": fixture["padding_mask"][0].bool()}
        report["token_scopes"] = {
            key: {scope: errors(a[key][:, mask], b[key][:, mask]) for scope, mask in masks.items() if mask.any()}
            for key in a if key.startswith("layer") or key == "final_norm"
        }
    if "logits" not in a:
        stages = ["patch_embed", *[f"blocks.{i}" for i in range(24)], "post_layernorm", "downsample", "merger"]
        fig, ax = plt.subplots(figsize=(12, 4), layout="constrained")
        ax.bar(stages, [report["stages"][key]["relative_l2"] for key in stages])
        ax.set(ylabel="Relative L2 error", title="Frozen ViT / projector; train + no_grad, compile disabled")
        ax.tick_params(axis="x", rotation=75)
        fig.savefig(args.out / "vision_hidden_states.png", dpi=180)
    else:
        assert torch.equal(a["labels"], b["labels"]) and torch.equal(a["positions"], b["positions"])
        x, y = a["logits"][0].float(), b["logits"][0].float()
        labels, positions = a["labels"][0].long(), a["positions"].long()
        ce_x, ce_y = F.cross_entropy(x, labels, reduction="none"), F.cross_entropy(y, labels, reduction="none")
        rel = (y - x).norm(dim=-1) / x.norm(dim=-1).clamp_min(1e-30)
        report["logits_summary"] = {**errors(a["logits"], b["logits"]),
                                    "top1_disagreement": (x.argmax(-1) != y.argmax(-1)).float().mean().item(),
                                    "ce_token_mean_xtuner": ce_x.mean().item(), "ce_token_mean_automodel": ce_y.mean().item()}
        rows = [{"position": int(pos), "label": int(label), "relative_l2": float(error),
                 "ce_xtuner": float(cx), "ce_automodel": float(cy), "ce_delta": float(cy-cx)}
                for pos, label, error, cx, cy in zip(positions, labels, rel, ce_x, ce_y)]
        with (args.out / "supervised_token_errors.csv").open("w") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
        stages = ["visual_features", "embed_tokens", "layer0.input", "layer0.input_layernorm", "layer0", "layer1", "layer2", "layer3", "final_norm"]
        axes[0, 0].bar(stages, [report.get("token_scopes", {}).get(key, {}).get("real_tokens", report["stages"][key])["relative_l2"] for key in stages])
        axes[0, 0].set(ylabel="Relative L2 error", title="Forward hidden states (real tokens; padding excluded)")
        axes[0, 0].tick_params(axis="x", rotation=35)
        axes[0, 1].plot(positions, rel)
        axes[0, 1].set(xlabel="Supervised token position", ylabel="Logits relative L2 error")
        # Only the scatter is downsampled; comparison and saved logits use every supervised row/vocab coordinate.
        stride = max(1, x.numel() // 5000)
        xs, ys = x.flatten()[::stride], y.flatten()[::stride]
        axes[1, 0].scatter(xs, ys, s=2, alpha=0.3)
        lo, hi = float(torch.min(xs.min(), ys.min())), float(torch.max(xs.max(), ys.max()))
        axes[1, 0].plot([lo, hi], [lo, hi], color="gray", linewidth=1)
        axes[1, 0].set(xlabel="XTuner logits", ylabel="AutoModel logits", title="Sampled coordinates for display")
        axes[1, 1].plot(positions, ce_y-ce_x)
        axes[1, 1].set(xlabel="Supervised token position", ylabel="CE delta (AutoModel - XTuner)")
        fig.suptitle("Same weight origin / same full 8K pack; frozen ViT + projector, compile disabled")
        fig.savefig(args.out / "forward_logits_hidden_states.png", dpi=180)
    (args.out / "comparison.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({key: value for key, value in report.items() if key != "stages"}, indent=2))


if __name__ == "__main__":
    main()
