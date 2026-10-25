"""Aggregate P8 evidence and render compact comparison figures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args(argv)
    root = args.root
    models = {}
    for name in ("single_mlp", "history_mlp", "gru", "gru_no_action_history"):
        models[name] = _read(root / "models" / name / "summary.json")
    evaluations = {}
    for name in (
        "single_mlp_expert_hand",
        "history_mlp_expert_hand",
        "gru_expert_hand",
        "gru_hold_hand",
        "gru_no_action_history_hold",
        "gru_single_root_memorization",
        "gru_bc_hold",
    ):
        evaluations[name] = _read(root / "hold_eval" / name / "summary.json")
    p7 = _read(Path("outputs/dp_p7_2026-10-11/validated_recovery/summary.json"))
    source_classification = {
        "source_a": {
            "description": "ten independently executed perturbed full expert roots",
            "formal_training_roots": 10,
            "strict_successes": 10,
        },
        "source_b": {
            "description": "Hold/Lift suffixes extracted from Source-A roots",
            "formal_training_roots": 10,
            "independence_note": "suffixes retain their parent root id and are never counted as new roots",
        },
        "source_c": {
            "description": "P7 BC-deviated snapshots followed by oracle expert XYZ + BC hand",
            "available_strict_successes": p7["strict_successes"],
            "formal_training_roots": 0,
            "exclusion": "single shared root and oracle XYZ; diagnostic only",
        },
    }
    result = {
        "dataset": _read(root / "lift_dataset" / "manifest.json"),
        "independence": _read(root / "dataset_audit" / "source_a.json"),
        "lift_physics": _read(root / "lift_physics" / "summary.json"),
        "models": {
            key: {
                "best_validation_loss": value["best_validation_loss"],
                "runtime_seconds": value["runtime_seconds"],
            }
            for key, value in models.items()
        },
        "evaluations": {
            key: {"successes": value["successes"], "trials": value["trials"]}
            for key, value in evaluations.items()
        },
        "source_classification": source_classification,
        "hierarchical_rollout": {
            "status": "NOT_RUN_GATE_BLOCKED",
            "reason": "GRU succeeds from an unseen expert Hold root but fails from the actual frozen-BC Hold snapshot.",
        },
    }
    (root / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    (root / "dataset_audit" / "source_classification.json").write_text(
        json.dumps(source_classification, indent=2) + "\n"
    )
    import matplotlib.pyplot as plt

    figures = root / "figures"
    figures.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for name, summary in models.items():
        history = summary["history"]
        ax.plot(
            [item["epoch"] for item in history],
            [item["validation_loss"] for item in history],
            label=name.replace("_", " "),
        )
    ax.set(xlabel="Epoch", ylabel="Normalized validation MSE", yscale="log")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures / "lift_controller_validation_loss.png", dpi=180)
    plt.close(fig)

    names = ["single_mlp_expert_hand", "history_mlp_expert_hand", "gru_hold_hand", "gru_no_action_history_hold"]
    labels = ["Single MLP", "History MLP", "GRU + action hist", "GRU state hist"]
    success = [evaluations[name]["successes"] for name in names]
    max_lift = [
        max(1000 * item["strict_replay"]["max_lift"] for item in evaluations[name]["results"])
        for name in names
    ]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    axes[0].bar(labels, success)
    axes[0].set(ylabel="FINAL_VERIFIED / 2", ylim=(0, 2))
    axes[1].bar(labels, max_lift)
    axes[1].axhline(55, color="red", linestyle="--", label="55 mm lift threshold")
    axes[1].set(ylabel="Maximum lift on either test root (mm)")
    axes[1].legend(fontsize=8)
    for axis in axes:
        axis.tick_params(axis="x", rotation=25)
        axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(figures / "hold_start_ablation.png", dpi=180)
    plt.close(fig)
    print(json.dumps({"summary": str(root / "summary.json"), "figures": str(figures)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
