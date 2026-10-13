"""Full-vocabulary logits on one fixed pack at every completed update."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
import torch
import torch.nn.functional as F
from safetensors.torch import load_file

from experiments.train_infra.plot_forward import errors

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def relative_error_statistics(relative_l2):
    """Summarize per-token full-vocabulary relative L2, with an interpolated median."""
    assert relative_l2.ndim == 1 and relative_l2.numel() > 0
    assert torch.isfinite(relative_l2).all()
    return {"token_relative_l2_max": float(relative_l2.max()),
            "token_relative_l2_mean": float(relative_l2.mean()),
            "token_relative_l2_median": float(torch.quantile(relative_l2, 0.5))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    statuses = json.loads((args.out / "run_status.json").read_text())
    assert len(statuses) == 2 and all(s["optimizer_gate_passed"] for s in statuses), statuses
    steps = {s["expected_steps"] for s in statuses}
    assert len(steps) == 1
    total = steps.pop()
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    reports = []
    rows = []
    input_hashes = None
    for updates in range(total + 1):
        paths = [args.runs / name / "logits" / f"updates_{updates:03d}" for name in ("xtuner_e2e", "automodel_e2e")]
        assert all((p / "logits.safetensors").is_file() for p in paths), f"Missing logits at update {updates}"
        metadata = [json.loads((p / "metadata.json").read_text()) for p in paths]
        assert metadata[0]["input_hashes"] == metadata[1]["input_hashes"]
        if input_hashes is None:
            input_hashes = metadata[0]["input_hashes"]
        assert metadata[0]["input_hashes"] == input_hashes, "Fixed input changed between updates"
        assert all(m["completed_updates"] == updates for m in metadata)
        a, b = [load_file(str(p / "logits.safetensors")) for p in paths]
        assert torch.equal(a["positions"], b["positions"]) and torch.equal(a["labels"], b["labels"])
        summary = errors(a["logits"], b["logits"])
        x, y = a["logits"][0].float(), b["logits"][0].float()
        positions, labels = a["positions"], a["labels"][0]
        rel = (y - x).norm(dim=-1) / x.norm(dim=-1).clamp_min(1e-30)
        cx, cy = (F.cross_entropy(z, labels, reduction="none") for z in (x, y))
        summary.update(completed_updates=updates, top1_disagreement=float((x.argmax(-1) != y.argmax(-1)).float().mean()),
                       ce_token_mean_xtuner=float(cx.mean()), ce_token_mean_automodel=float(cy.mean()))
        summary.update(relative_error_statistics(rel))
        reports.append(summary)
        label = f"{updates} completed updates"
        if updates in (0, total):
            axes[0, 0].scatter(positions, rel, s=3, label=label, alpha=0.75)
            axes[0, 1].scatter(positions, cy-cx, s=3, label=label, alpha=0.75)
            stride = max(1, x.numel() // 5000)
            xs, ys = x.flatten()[::stride], y.flatten()[::stride]
            axes[1, 0].scatter(xs, ys, s=3, alpha=0.3, label=label)
        rows.extend({"completed_updates": updates, "position": int(p), "relative_l2": float(r),
                     "ce_xtuner": float(u), "ce_automodel": float(v)} for p,r,u,v in zip(positions,rel,cx,cy))
        del a, b, x, y
    axes[0, 0].set(xlabel="Fixed-pack supervised token position", ylabel="Logits relative L2 error")
    axes[0, 1].set(xlabel="Fixed-pack supervised token position", ylabel="CE delta (AM - XT)")
    axes[1, 0].set(xlabel="XTuner logits", ylabel="AutoModel logits", title="Sampled coordinates for display only")
    axes[1, 1].plot([r["completed_updates"] for r in reports], [r["relative_l2"]*100 for r in reports], marker="o", label="Relative L2 (%)")
    axes[1, 1].plot([r["completed_updates"] for r in reports], [r["top1_disagreement"]*100 for r in reports], marker="o", label="Top1 disagreement (%)")
    axes[1, 1].set(xlabel="Completed optimizer updates", ylabel="Error (%)", xticks=[0,total])
    for ax in axes.flat:
        ax.legend(fontsize=8)
    fig.suptitle(f"Fixed 8K pack before/after {total} real training steps; native AM text / XT HF RMSNorm")
    fig.savefig(args.out / "logits_before_after_training.png", dpi=180)
    with (args.out / "fixed_pack_token_errors.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    (args.out / "logits_comparison.json").write_text(json.dumps(reports, indent=2))
    columns = ["completed_updates", "token_relative_l2_max", "token_relative_l2_mean", "token_relative_l2_median"]
    with (args.out / "logits_step_statistics.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(reports)
    # One overview lets training curves and fixed-input forward differences be read together.
    overview, panels = plt.subplots(1, 3, figsize=(18, 5), layout="constrained")
    metrics = list(csv.DictReader((args.out / "optimizer_metrics.csv").open()))
    for name, label in (("xtuner_e2e", "XTuner"), ("automodel_e2e", "AutoModel")):
        trace = [r for r in metrics if r["run"] == name and r["rank"] == "0"]
        assert [int(r["step"]) for r in trace] == list(range(total))
        for panel, key in ((panels[0], "loss"), (panels[2], "grad_norm")):
            panel.plot([int(r["step"]) for r in trace], [float(r[key]) for r in trace],
                       marker="o", markersize=3, label=f"{label} ({total}/{total} steps)")
    for key, label in (("max", "Maximum"), ("mean", "Mean"), ("median", "Median")):
        panels[1].plot([r["completed_updates"] for r in reports],
                       [r[f"token_relative_l2_{key}"]*100 for r in reports],
                       marker="o", markersize=3, label=label)
    panels[0].set(title="Loss", xlabel="Optimizer step (zero based)", ylabel="Loss")
    panels[1].set(title="Logits: per-token relative L2 statistics", xlabel="Completed optimizer updates (0 = initial)",
                  ylabel="Relative L2 error (%)")
    panels[2].set(title="Grad norm before clipping", xlabel="Optimizer step (zero based)",
                  ylabel="Global L2 norm")
    for panel in panels:
        panel.legend(fontsize=9)
        panel.grid(alpha=0.2)
    overview.suptitle(f"{total} optimizer steps | GLM-5.3-Flash 4 layers, 8 GPUs, 8K | frozen ViT/projector, compile off")
    overview.savefig(args.out / "loss_logits_grad_norm.png", dpi=180)
    print(json.dumps(reports, indent=2))


if __name__ == "__main__":
    main()
