"""Aggregate P9 supervisor and bounded preparation-search evidence."""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np


def _read(path):
    return json.loads(Path(path).read_text())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args(argv)
    root = args.root
    supervisor = {
        name: _read(root / "supervisor" / name / "summary.json")
        for name in ("root08_fixed_stop", "root08_stop_hold", "root09_stop_hold")
    }
    direct = [
        _read(path)
        for path in sorted(glob.glob(str(root / "direct_bc_lift" / "root_*" / "summary.json")))
    ]
    preparation = _read(root / "preparation_search" / "summary.json")
    result = {
        "supervisor": {
            key: {
                "strict_success": value["strict_replay"]["success"],
                "verification_status": value["strict_replay"]["verification_status"],
                "maximum_wrist_rise_m": value["maximum_wrist_rise_m"],
                "wrist_overshoot_m": value["wrist_overshoot_m"],
                "final_lift_m": value["strict_replay"]["final_lift"],
                "tail_max_angular_speed": value["strict_replay"]["tail_max_angular_speed"],
            }
            for key, value in supervisor.items()
        },
        "bc_roots": len(direct),
        "direct_lift_ready": sum(row["strict_replay"]["success"] for row in direct),
        "preparation_attempts": preparation["attempts"],
        "preparation_validated": preparation["preparation_validated"],
        "preparation_controller": "NOT_TRAINED_ZERO_VALIDATED_LABELS",
        "readiness_predictor": "NOT_TRAINED_SINGLE_NEGATIVE_CLASS",
        "full_hierarchical_rollout": "NOT_RUN_PREPARATION_GATE_BLOCKED",
    }
    (root / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    figures = root / "figures"
    figures.mkdir(exist_ok=True)
    import matplotlib.pyplot as plt

    original = np.load("outputs/dp_p8_2026-10-14/hold_eval/gru_hold_hand/root_08/trace.npz")
    controlled = np.load(root / "supervisor" / "root08_stop_hold" / "trace.npz")
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=False)
    for archive, label, axis in (
        (original, "Original GRU", axes[0]),
        (controlled, "Supervisor + Hold", axes[1]),
    ):
        wrist = archive["rollout_wrist"]
        obj = archive["rollout_object"]
        axis.plot(1000 * (wrist[:, 2] - wrist[0, 2]), label="wrist rise")
        axis.plot(1000 * (obj[:, 2] - obj[0, 2]), label="object rise")
        axis.axhline(85, color="red", linestyle="--", label="task wrist target")
        axis.set(ylabel="Rise (mm)", title=label)
        axis.grid(alpha=0.25)
        axis.legend()
    axes[1].set_xlabel("Control step")
    fig.tight_layout()
    fig.savefig(figures / "supervisor_overshoot_comparison.png", dpi=180)
    plt.close(fig)

    rows = preparation["results"]
    names = sorted({row["candidate"]["name"] for row in rows})
    values = [
        np.mean([row["object_following_ratio"] for row in rows if row["candidate"]["name"] == name])
        for name in names
    ]
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.bar(names, values)
    ax.axhline(0.5, color="red", linestyle="--", label="PREPARATION_VALIDATED threshold")
    ax.set(ylabel="Mean 5 mm probe following ratio")
    ax.tick_params(axis="x", rotation=30)
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures / "preparation_load_probe.png", dpi=180)
    plt.close(fig)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
