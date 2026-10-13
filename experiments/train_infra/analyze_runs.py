"""Export actual optimizer metrics; failed or incomplete runs cannot pass acceptance."""

import argparse
import csv
import json
import math
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    metrics, statuses = [], []
    for run in sorted(args.runs.iterdir()):
        if not run.is_dir() or not (run / "training.log").is_file():
            continue
        rank_rows = []
        for rank in range(8):
            path = run / f"metrics_rank{rank}.jsonl"
            records = [json.loads(raw) for raw in path.open()] if path.is_file() else []
            rank_rows.append(records)
            metrics.extend({"run": run.name, "rank": rank, **record} for record in records)
        exit_file = run / "exit_code.txt"
        exit_code = int(exit_file.read_text()) if exit_file.is_file() else None
        counts = [len(records) for records in rank_rows]
        finite = all(math.isfinite(row[key]) for records in rank_rows for row in records
                     for key in ("loss", "grad_norm", "lr", "update_max_abs"))
        updated = all(any(row["lr"] > 0 and row["update_max_abs"] > 0 for row in records)
                      for records in rank_rows)
        status = {"run": run.name, "exit_code": exit_code, "rank_step_counts": counts,
                  "finite_metrics": finite if min(counts) else None,
                  "nonzero_updates_all_ranks": updated,
                  "optimizer_gate_passed": exit_code == 0 and counts == [20] * 8 and finite and updated}
        environment = run / "environment_rank0.json"
        native = json.loads(environment.read_text()) if environment.is_file() else {}
        config = run / "resolved_config.json"
        reference = Path(native["reference"]) if "reference" in native else None
        if reference is not None:
            status["reference_run"] = reference.name
            status["layer_groups"] = native.get("layer_groups")
            config = reference / "resolved_config.json"
        valid = not (run / "CONFIGURATION_ABORTED.txt").exists()
        if run.name.startswith("automodel_"):
            valid = valid and native.get("layer_groups") == {"language": 4, "vision": 24}
        status["configuration_validated"] = valid
        status["optimizer_gate_passed"] &= valid
        if config.is_file():
            cfg = json.loads(config.read_text())
            status.update(pack_length=cfg["dataloader_cfg"]["pack_max_length"], sp_size=cfg["sp_size"],
                          ep_size=cfg["fsdp_cfg"]["ep_size"], compile=cfg["fsdp_cfg"]["torch_compile"])
            status["reference_configuration"] = (status["pack_length"] == 65536 and
                                                 status["sp_size"] == 1 and status["ep_size"] == 1 and
                                                 status["compile"] and valid)
        if reference is not None and rank_rows[0]:
            reference_metrics = reference / "metrics_rank0.jsonl"
            baseline = {row["step"]: row for row in map(json.loads, reference_metrics.read_text().splitlines())} \
                if reference_metrics.is_file() else {}
            differences = [{"step": row["step"], **{
                f"{key}_abs_error": abs(row[key] - baseline[row["step"]][key])
                for key in ("loss", "grad_norm", "lr")}}
                for row in rank_rows[0] if row["step"] in baseline]
            status["observed_differences"] = differences
            status["numerical_acceptance"] = "Not calibrated; observed errors do not establish parity"
        statuses.append(status)
    columns = ["run", "rank", "step", "loss", "ce_loss", "balancing_loss", "grad_norm", "lr",
               "probe", "update_max_abs", "peak_memory_bytes"]
    with (args.out / "optimizer_metrics.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(metrics)
    (args.out / "run_status.json").write_text(json.dumps(statuses, indent=2))
    if metrics:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        figure, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
        for status in statuses:
            rows = sorted((row for row in metrics if row["run"] == status["run"] and row["rank"] == 0),
                          key=lambda row: row["step"])
            if rows:
                for axis, key in zip(axes, ("loss", "grad_norm")):
                    axis.plot([row["step"] for row in rows], [row[key] for row in rows], label=status["run"])
                    axis.set(xlabel="Optimizer step (zero based)", ylabel=key)
                    axis.legend(fontsize=6)
        figure.savefig(args.out / "loss_grad_norm.png", dpi=180)
    print(json.dumps(statuses, indent=2))


if __name__ == "__main__":
    main()
