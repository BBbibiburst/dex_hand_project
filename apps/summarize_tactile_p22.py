"""Summarize the frozen P22 validation and final-test evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt


def _read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(args: argparse.Namespace) -> int:
    validation = _read(args.output / "empirical_transport_validation.json")
    final = _read(args.output / "empirical_final_test.json")
    route_a = _read(args.output / "route_a_candidates.json")
    route_bc = _read(args.output / "route_bc_candidates.json")
    empirical = _read(args.output / "empirical_transport_candidates.json")
    domain = final["domain_test"]
    task = final["task_test"]
    logistic_mlp = [item for item in domain if item["probe"] in {"logistic", "mlp"}]
    raw_task = {item["label"]: item["balanced_accuracy"] for item in task if item["source"] == "raw"}
    mapped_task = {
        item["label"]: item["balanced_accuracy"] for item in task if item["source"] == "mapped"
    }
    summary = {
        "schema_version": 1,
        "dataset_sha256": _sha256(args.dataset),
        "split_manifest_sha256": _sha256(args.split_manifest),
        "candidate_counts": {
            "route_a_conditional_transport": len(route_a),
            "route_bc_learned_mapper": len(route_bc),
            "empirical_transport": len(empirical),
        },
        "locked_candidate": validation["candidate"],
        "validation_target_met": validation["validation_target_met"],
        "final_test": {
            "domain": domain,
            "domain_by_stage": final["domain_test_by_stage"],
            "task": task,
            "contract": final["contract"],
            "unique_output_rows": final["unique_output_rows"],
            "logistic_mlp_min_confusion": min(
                item["domain_confusion"] for item in logistic_mlp
            ),
            "logistic_mlp_max_confusion": max(
                item["domain_confusion"] for item in logistic_mlp
            ),
            "task_min_ba": min(mapped_task.values()),
            "task_max_relative_drop": max(
                raw_task[key] - mapped_task[key] for key in mapped_task
            ),
        },
        "acceptance": {
            "domain_pass": all(item["domain_confusion"] >= 0.90 for item in logistic_mlp),
            "contract_pass": final["contract"]["finite"]
            and all(
                value == 0.0
                for key, value in final["contract"].items()
                if key.endswith("violation_fraction")
            ),
            "task_pass": min(mapped_task.values()) >= 0.95
            and max(raw_task[key] - mapped_task[key] for key in mapped_task) <= 0.03,
        },
        "test_selection_disclosure": "The test set was evaluated once after the validation gate passed; no post-test candidate selection was performed.",
    }
    summary["acceptance"]["overall_pass"] = all(summary["acceptance"].values())
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    labels = [f"{item['probe']}\n{item['seed']}" for item in domain]
    axes[0].bar(labels, [item["domain_confusion"] for item in domain], color="#4C78A8")
    axes[0].axhline(0.9, color="#E45756", linestyle="--", label="target 0.90")
    axes[0].set_ylim(0, 1.05)
    axes[0].set_ylabel("D_confusion")
    axes[0].set_title("Frozen test domain confusion")
    axes[0].tick_params(axis="x", rotation=35)
    axes[0].legend()
    task_labels = list(mapped_task)
    x = range(len(task_labels))
    axes[1].bar([value - 0.18 for value in x], [raw_task[key] for key in task_labels], 0.36, label="raw")
    axes[1].bar([value + 0.18 for value in x], [mapped_task[key] for key in task_labels], 0.36, label="mapped")
    axes[1].axhline(0.95, color="#E45756", linestyle="--", label="floor 0.95")
    axes[1].set_xticks(list(x), task_labels, rotation=20)
    axes[1].set_ylim(0.9, 1.005)
    axes[1].set_ylabel("Balanced accuracy")
    axes[1].set_title("Held-out Sim task preservation")
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(args.output / "p22_final_results.png", dpi=180)
    plt.close(figure)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/tactile_p22_2026-10-25"))
    parser.add_argument(
        "--dataset", type=Path, default=Path("outputs/tactile_p19_gan_2026-10-25/dataset.npz")
    )
    parser.add_argument(
        "--split-manifest",
        type=Path,
        default=Path(
            "outputs/tactile_p18_2_2026-10-25/diagnostic_baseline/split_manifest.json"
        ),
    )
    return parser


if __name__ == "__main__":
    raise SystemExit(run(build_parser().parse_args()))
