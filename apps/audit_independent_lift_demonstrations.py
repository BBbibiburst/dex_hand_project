"""Audit strict status and root-state diversity of Lift demonstrations."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from source.grasping.contracts import DemonstrationEpisode


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def audit(root: Path) -> dict:
    rows = []
    for manifest in sorted(root.glob("p*/candidate_*/manifest.json")):
        strict_path = manifest.with_name("strict_replay_final.json")
        if not strict_path.is_file():
            continue
        strict = json.loads(strict_path.read_text(encoding="utf-8"))["result"]
        episode = DemonstrationEpisode.load(manifest)
        metadata = episode.metadata
        rows.append(
            {
                "manifest": str(manifest.resolve()),
                "episode_sha256": _digest(manifest.with_name("episode.npz")),
                "seed": episode.seed,
                "source_manifest": metadata.get("source_manifest"),
                "source_a": True,
                "independent_execution": bool(metadata.get("independent_execution", False)),
                "perturbation": metadata.get("perturbation"),
                "initial_object_position": metadata.get("object_initial_position"),
                "initial_object_quaternion_wxyz": metadata.get(
                    "object_initial_quaternion_wxyz"
                ),
                "frames": len(episode.arrays["action"]),
                "strict_success": bool(strict["success"]),
                "verification_status": strict["verification_status"],
                "final_lift_m": float(strict["final_lift"]),
                "max_lift_m": float(strict["max_lift"]),
            }
        )
    accepted = [row for row in rows if row["strict_success"]]
    positions = np.asarray([row["initial_object_position"] for row in accepted], dtype=float)
    distances = []
    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            distances.append(float(np.linalg.norm(positions[i] - positions[j])))
    unique_seeds = len({row["seed"] for row in accepted})
    unique_hashes = len({row["episode_sha256"] for row in accepted})
    unique_positions = len({tuple(np.round(row["initial_object_position"], 9)) for row in accepted})
    return {
        "schema_version": 1,
        "root": str(root.resolve()),
        "attempts": len(rows),
        "strict_accepted": len(accepted),
        "strict_success_rate": len(accepted) / len(rows) if rows else 0.0,
        "unique_seeds": unique_seeds,
        "unique_episode_hashes": unique_hashes,
        "unique_initial_object_positions": unique_positions,
        "minimum_pairwise_position_distance_m": min(distances) if distances else None,
        "maximum_pairwise_position_distance_m": max(distances) if distances else None,
        "independence_gate": bool(
            accepted
            and unique_seeds == len(accepted)
            and unique_hashes == len(accepted)
            and unique_positions == len(accepted)
            and all(row["independent_execution"] for row in accepted)
        ),
        "accepted_manifests": [row["manifest"] for row in accepted],
        "rows": rows,
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = audit(args.root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key != "rows"}, indent=2))
    return 0 if summary["independence_gate"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
