"""Analyze taxel-region response around BC multi-finger contact loss."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from source.sensors.tactile.dex_hand import SimpleBoxTactileSensor


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _groups(sensor: SimpleBoxTactileSensor) -> dict[str, np.ndarray]:
    result: dict[str, list[int]] = {f"digit_{i}": [] for i in range(5)}
    result["palm"] = []
    result["other"] = []
    for patch in sensor.patches:
        indices = list(range(patch.start, patch.stop))
        lower = patch.name.lower()
        target = "other"
        if "palm" in lower:
            target = "palm"
        else:
            for digit in range(5):
                if f"skin_{digit}_" in lower:
                    target = f"digit_{digit}"
                    break
        result[target].extend(indices)
    return {name: np.asarray(indices, dtype=np.int64) for name, indices in result.items()}


def _first_loss(opposing: np.ndarray, minimum_run: int = 10) -> int | None:
    for index in range(minimum_run, len(opposing)):
        if opposing[index - minimum_run : index].all() and not opposing[index]:
            return index
    return None


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    with np.load(args.trace, allow_pickle=False) as archive:
        arrays = {name: np.asarray(archive[name]) for name in archive.files}
    sensor = SimpleBoxTactileSensor()
    groups = _groups(sensor)
    result = {"trace": str(args.trace), "groups": {}, "warning_horizon_steps": 10}
    series = {}
    for prefix in ("expert", "bc"):
        tactile = arrays[f"{prefix}_tactile"]
        digit_contact = arrays[f"{prefix}_digit_contact_count"] > 0
        opposing = digit_contact[:, 4] & (digit_contact[:, :4].sum(axis=1) >= 2)
        grouped = {
            name: tactile[:, indices].sum(axis=1) if len(indices) else np.zeros(len(tactile))
            for name, indices in groups.items()
        }
        loss = _first_loss(opposing)
        entry = {
            "first_sustained_opposing_contact_loss": loss,
            "opposing_contact_fraction": float(opposing.mean()),
            "total_tactile_mean": float(tactile.sum(axis=1).mean()),
            "pre_loss": None,
        }
        if loss is not None:
            before = slice(max(0, loss - 10), loss)
            after = slice(loss, min(len(tactile), loss + 10))
            entry["pre_loss"] = {
                "total_10_steps_before_mean": float(tactile[before].sum(axis=1).mean()),
                "total_10_steps_after_mean": float(tactile[after].sum(axis=1).mean()),
                "regional_before": {
                    name: float(values[before].mean()) for name, values in grouped.items()
                },
                "regional_after": {
                    name: float(values[after].mean()) for name, values in grouped.items()
                },
            }
        result["groups"][prefix] = entry
        series[prefix] = (grouped, opposing)
    result["predictive_model"] = {
        "trained": False,
        "reason": (
            "Only one stable expert episode and one failed BC episode are available; an "
            "episode-level train/validation split for a failure predictor is not statistically valid."
        ),
    }
    (args.output / "tactile_warning.json").write_text(json.dumps(result, indent=2) + "\n")

    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    for axis, prefix in zip(axes, ("expert", "bc"), strict=True):
        grouped, opposing = series[prefix]
        for name in ("digit_0", "digit_1", "digit_2", "digit_3", "digit_4", "palm"):
            axis.plot(grouped[name], label=name, linewidth=1.0)
        axis.fill_between(
            np.arange(len(opposing)),
            0,
            opposing * max(1.0, axis.get_ylim()[1]),
            alpha=0.08,
            color="black",
            label="opposing contact",
        )
        loss = result["groups"][prefix]["first_sustained_opposing_contact_loss"]
        if loss is not None:
            axis.axvline(loss, color="red", linestyle="--", label=f"first loss {loss}")
        axis.set_ylabel("Summed normalized taxel response")
        axis.set_title(prefix.capitalize())
        axis.grid(alpha=0.2)
        axis.legend(ncol=4, fontsize=8)
    axes[-1].set_xlabel("Control step")
    figure.tight_layout()
    figure.savefig(args.output / "tactile_regions_and_contact_loss.png", dpi=180)
    plt.close(figure)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
