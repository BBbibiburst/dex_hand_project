"""Transfer one verified Lift grasp into a PickPlace demonstration."""

from __future__ import annotations

import argparse
import json
from dataclasses import fields, replace
from pathlib import Path

import numpy as np

from source.grasping.contracts import DemonstrationEpisode
from source.grasping.executor import ExecutionConfig
from source.grasping.task_transfer import (
    PICK_PLACE_PIPELINE_VERSION,
    PickPlaceTransferConfig,
    execute_pick_place_transfer,
)
from source.grasp_pipeline.trajectory import GraspTrajectory


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lift-manifest", action="append", type=Path, dest="lift_manifests")
    parser.add_argument(
        "--lift-root",
        action="append",
        type=Path,
        dest="lift_roots",
        help="Recursively discover successful Lift episode manifests; repeat as needed.",
    )
    parser.add_argument("--object-id", help="Restrict discovered manifests to one object.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--seed-attempts", type=int, default=6)
    parser.add_argument("--maximum-candidates", type=int, default=12)
    parser.add_argument("--render-mode", choices=("human", "rgb_array"))
    parser.add_argument("--clearance-height", type=float, default=0.065)
    parser.add_argument("--release-clearance", type=float, default=0.002)
    parser.add_argument("--descend-steps", type=int, default=55)
    return parser


def _source_execution_config(episode: DemonstrationEpisode) -> ExecutionConfig:
    payload = dict(episode.metadata.get("execution_config", {}))
    allowed = {item.name for item in fields(ExecutionConfig)}
    return ExecutionConfig(**{key: value for key, value in payload.items() if key in allowed})


def discover_lift_manifests(
    manifests: list[Path] | None,
    roots: list[Path] | None,
    *,
    object_id: str | None,
) -> tuple[Path, ...]:
    candidates = [path / "manifest.json" if path.is_dir() else path for path in manifests or []]
    for root in roots or []:
        if not root.is_dir():
            raise FileNotFoundError(f"Lift root does not exist: {root}")
        candidates.extend(root.glob("**/manifest.json"))
    selected: list[Path] = []
    for path in dict.fromkeys(path.resolve() for path in candidates):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        episode_source = isinstance(payload.get("candidate"), dict)
        rl_source = payload.get("action_mode") == "grasp_edit_hybrid"
        if not payload.get("success") or not (episode_source or rl_source):
            continue
        if object_id is not None and payload.get("object_id") != object_id:
            continue
        selected.append(path)
    return tuple(selected)


def load_lift_source(path: Path) -> DemonstrationEpisode:
    """Load a direct Lift episode or reconstruct the candidate refined by PPO."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload.get("candidate"), dict):
        source = DemonstrationEpisode.load(path)
        if not source.success:
            raise ValueError(f"Lift source is not successful: {path}")
        return source

    trajectory = GraspTrajectory.load(path)
    source_manifest = Path(trajectory.source_manifest)
    if not source_manifest.is_file():
        fallback = trajectory.metadata.get("template_manifest")
        source_manifest = Path(str(fallback)) if fallback else source_manifest
    if not source_manifest.is_file() and "outputs" in source_manifest.parts:
        output_index = source_manifest.parts.index("outputs")
        source_manifest = Path.cwd().joinpath(*source_manifest.parts[output_index:])
    source = DemonstrationEpisode.load(source_manifest)
    base_grip = np.asarray(trajectory.metadata["base_grip_fractions"], dtype=np.float64)
    hand_edit = np.asarray(trajectory.metadata["hand_edit_normalized"], dtype=np.float64)
    if base_grip.shape != (6,) or hand_edit.shape != (6,):
        raise ValueError(f"Invalid PPO hand-edit metadata in {path}")
    hand_edit_fraction = float(trajectory.metadata.get("hand_edit_fraction", 0.20))
    desired_grip = np.clip(base_grip + hand_edit_fraction * hand_edit, 0.0, 1.0)
    execution = _source_execution_config(source)
    preload = np.asarray(
        [
            execution.finger_preload,
            execution.finger_preload,
            execution.finger_preload,
            execution.finger_preload,
            0.0,
            execution.thumb_grasp_preload,
        ],
        dtype=np.float64,
    )
    source.candidate = replace(
        source.candidate,
        actuator_fractions=np.clip(desired_grip - preload, 0.0, 1.0),
        backend=f"{source.candidate.backend}+mjwarp-ppo-transfer",
        metrics={
            **source.candidate.metrics,
            "ppo_source_success": 1.0,
            "ppo_source_max_lift": float(trajectory.metadata.get("mjwarp_max_lift", 0.0)),
        },
    )
    source.success = True
    source.metadata["ppo_source_manifest"] = str(path)
    return source


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.seed_attempts <= 0 or args.maximum_candidates <= 0:
        raise ValueError("--seed-attempts and --maximum-candidates must be positive.")
    manifests = discover_lift_manifests(
        args.lift_manifests, args.lift_roots, object_id=args.object_id
    )
    if not manifests:
        raise ValueError("No successful Lift episode manifests were found.")

    attempts = []
    seen_candidates: set[tuple[object, ...]] = set()
    candidate_index = 0
    for manifest in manifests:
        source = load_lift_source(manifest)
        source_trajectory = (
            GraspTrajectory.load(manifest) if "ppo_source_manifest" in source.metadata else None
        )
        identity = (
            source.object_id,
            source.candidate.seed_index,
            *np.round(source.candidate.hand_translation, 6),
            *np.round(source.candidate.hand_rotation_matrix.reshape(-1), 6),
            *np.round(source.candidate.actuator_fractions, 6),
        )
        if identity in seen_candidates:
            continue
        seen_candidates.add(identity)
        if candidate_index >= args.maximum_candidates:
            break
        candidate_index += 1
        seeds = (args.seed,) if args.seed is not None else tuple(range(args.seed_attempts))
        for seed in seeds:
            for release_mode in ("surface", "air"):
                result = execute_pick_place_transfer(
                    source.candidate,
                    seed=seed,
                    grasp_config=_source_execution_config(source),
                    source_object_position=np.asarray(
                        source.metadata["object_initial_position"], dtype=np.float64
                    ),
                    source_object_quaternion_wxyz=np.asarray(
                        source.metadata["object_initial_quaternion_wxyz"], dtype=np.float64
                    ),
                    source_episode=source,
                    source_trajectory=source_trajectory,
                    transfer_config=PickPlaceTransferConfig(
                        clearance_height=args.clearance_height,
                        release_clearance=args.release_clearance,
                        release_in_air=release_mode == "air",
                        descend_steps=1 if release_mode == "air" else args.descend_steps,
                    ),
                    render_mode=args.render_mode,
                )
                attempt_dir = (
                    args.output / "attempts" / f"c{candidate_index:02d}_s{seed:04d}_{release_mode}"
                )
                attempt_manifest = result.save(attempt_dir)
                attempts.append(
                    {
                        "source_manifest": str(manifest),
                        "seed": seed,
                        "release_mode": release_mode,
                        "success": result.success,
                        "failure_reason": result.failure_reason,
                        "manifest": str(attempt_manifest),
                    }
                )
                print(
                    f"[pick_place] candidate={candidate_index:02d} seed={seed:04d} "
                    f"release={release_mode} success={result.success} "
                    f"frames={len(result.arrays['action'])}",
                    flush=True,
                )
                if result.success:
                    best = result.save(args.output / "best_trajectory")
                    summary = {
                        "task": "pick_place",
                        "pipeline_version": PICK_PLACE_PIPELINE_VERSION,
                        "object_id": result.object_id,
                        "success": True,
                        "best_manifest": str(best),
                        "attempts": attempts,
                    }
                    args.output.mkdir(parents=True, exist_ok=True)
                    (args.output / "summary.json").write_text(
                        json.dumps(summary, indent=2), encoding="utf-8"
                    )
                    print(f"[done] success=True output={best}", flush=True)
                    return 0

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "summary.json").write_text(
        json.dumps(
            {
                "task": "pick_place",
                "pipeline_version": PICK_PLACE_PIPELINE_VERSION,
                "object_id": args.object_id,
                "success": False,
                "attempts": attempts,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[done] success=False attempts={len(attempts)}", flush=True)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
