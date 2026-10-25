"""Generate independently executed, strictly verified perturbed Lift demonstrations."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from source.envs.manipulation import make_lift_env
from source.envs.manipulation.placement import FixedTablePlacementSampler
from source.grasping.contracts import DemonstrationEpisode
from source.grasping.executor import ExecutionConfig, execute_grasp
from source.verification.profiles import FINAL_PROFILE
from source.verification.strict_replay import strict_replay_manifest

DEFAULT_OBJECTS = (
    "ycb:004_sugar_box",
    "ycb:005_tomato_soup_can",
    "ycb:007_tuna_fish_can",
)


def _slug(value: str) -> str:
    return value.replace(":", "_").replace("/", "_")


def _source_manifests(root: Path, object_id: str, maximum: int) -> tuple[Path, ...]:
    candidates: list[tuple[float, Path]] = []
    index_path = root / "lattice" / _slug(object_id) / "index.json"
    if index_path.is_file():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        for row in index.get("templates", ()):
            manifest = Path(str(row.get("manifest", "")))
            if row.get("success") and manifest.is_file():
                candidates.append((float(row.get("source_lift", 0.0)), manifest))
    for path in (root / "lattice" / _slug(object_id)).glob("*/manifest.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("success"):
            lift = float(payload.get("metadata", {}).get("object_lift", 0.0))
            candidates.append((lift, path))
    if not candidates:
        raise FileNotFoundError(f"No successful lattice manifest for {object_id} under {root}.")
    ordered = sorted(candidates, key=lambda item: (-item[0], str(item[1])))
    unique = tuple(dict.fromkeys(path.resolve() for _, path in ordered))
    return unique[:maximum]


def _explicit_source_manifests(paths: list[Path], object_id: str) -> tuple[Path, ...]:
    """Validate explicitly selected sources without silently re-ranking them."""

    sources: list[Path] = []
    for path in paths:
        manifest = (path / "manifest.json" if path.is_dir() else path).resolve()
        if not manifest.is_file():
            raise FileNotFoundError(f"Explicit source manifest does not exist: {manifest}")
        episode = DemonstrationEpisode.load(manifest)
        if episode.object_id != object_id:
            raise ValueError(
                f"Explicit source {manifest} is for {episode.object_id!r}, not {object_id!r}."
            )
        if not episode.success:
            raise ValueError(f"Explicit source did not complete its original execution: {manifest}")
        sources.append(manifest)
    return tuple(dict.fromkeys(sources))


def perturbations(count: int, xy_m: float, yaw_degrees: float):
    base = tuple(
        (x * xy_m, y * xy_m, yaw * yaw_degrees)
        for x, y, yaw in (
            (-1.0, 0.0, -1.0),
            (1.0, 0.0, 1.0),
            (0.0, -1.0, 1.0),
            (0.0, 1.0, -1.0),
            (-1.0, 1.0, 0.0),
            (1.0, -1.0, 0.0),
            (-1.0, -1.0, 0.5),
            (1.0, 1.0, -0.5),
            (-0.5, 0.5, 1.0),
            (0.5, -0.5, -1.0),
        )
    )
    if not 1 <= count <= len(base):
        raise ValueError(f"--per-object must lie in [1, {len(base)}].")
    return base[:count]


def classify_failure(reason: str | None, strict_metrics: dict | None) -> str:
    if reason is None:
        return "SUCCESS"
    lower = reason.lower()
    if "ik residual" in lower:
        return "IK_UNREACHABLE_OR_WRONG_BRANCH"
    if "approach" in lower:
        return "APPROACH_COLLISION_OR_TERMINATION"
    if "close" in lower or "hold" in lower:
        return "CLOSE_FAILURE"
    if "lift success criterion" in lower or "lift" in lower:
        return "LIFT_FAILURE"
    if reason == "FINAL_REJECTED" or strict_metrics is not None:
        return "STRICT_REPLAY_INSTABILITY"
    return "OTHER"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--object-id", action="append", dest="objects")
    parser.add_argument("--source-root", type=Path, default=Path("outputs/dex_hand_top100_v2"))
    parser.add_argument(
        "--source-manifest",
        action="append",
        type=Path,
        dest="source_manifests",
        help=(
            "Use this exact successful grasp episode as a candidate source; repeat for an "
            "ordered fallback list. Requires exactly one --object-id."
        ),
    )
    parser.add_argument("--output", type=Path, default=Path("outputs/dp_2026-10-11/verified_lift"))
    parser.add_argument("--per-object", type=int, default=5)
    parser.add_argument("--xy-mm", type=float, default=5.0)
    parser.add_argument("--yaw-deg", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=1011)
    parser.add_argument("--finger-preload", type=float, default=0.35)
    parser.add_argument("--thumb-preload", type=float, default=0.40)
    parser.add_argument("--lift-height", type=float, default=0.075)
    parser.add_argument("--verify-steps", type=int, default=60)
    parser.add_argument(
        "--close-steps",
        type=int,
        default=45,
        help="Number of linear hand-closing control steps (default: executor contract 45).",
    )
    parser.add_argument(
        "--candidate-retries",
        type=int,
        default=1,
        help="Maximum successful lattice candidates tried per fixed perturbation.",
    )
    return parser


def run(args: argparse.Namespace) -> int:
    objects = tuple(args.objects or DEFAULT_OBJECTS)
    offsets = perturbations(args.per_object, args.xy_mm / 1000.0, args.yaw_deg)
    rows = []
    args.output.mkdir(parents=True, exist_ok=True)
    if args.candidate_retries < 1:
        raise ValueError("--candidate-retries must be positive.")
    if args.close_steps <= 0:
        raise ValueError("--close-steps must be positive.")
    if args.source_manifests and len(objects) != 1:
        raise ValueError("Explicit --source-manifest requires exactly one --object-id.")
    # Validate every requested source before spending minutes on earlier
    # objects. This keeps a missing/failed source from leaving a misleadingly
    # partial multi-object experiment.
    source_sets = {}
    for object_id in objects:
        source_sets[object_id] = (
            _explicit_source_manifests(args.source_manifests, object_id)
            if args.source_manifests
            else _source_manifests(args.source_root, object_id, args.candidate_retries)
        )
    for object_index, object_id in enumerate(objects):
        sources = source_sets[object_id]
        source_episodes = tuple((source, DemonstrationEpisode.load(source)) for source in sources)
        _, primary_episode = source_episodes[0]
        base_position = np.asarray(
            primary_episode.metadata["object_initial_position"], dtype=np.float64
        )
        base_quaternion = np.asarray(
            primary_episode.metadata["object_initial_quaternion_wxyz"], dtype=np.float64
        )
        for perturbation_index, (dx, dy, yaw_deg) in enumerate(offsets):
            seed = args.seed + 100 * object_index + perturbation_index
            case_destination = args.output / _slug(object_id) / f"p{perturbation_index:02d}"
            started = time.perf_counter()
            execution_success = False
            strict_success = False
            failure_reason = None
            strict_metrics = None
            manifest = None
            source = sources[0]
            candidate_attempts = []
            for candidate_index, (source, source_episode) in enumerate(source_episodes):
                execution_success = False
                strict_success = False
                attempt_started = time.perf_counter()
                attempt_manifest = None
                attempt_reason = None
                attempt_strict = None
                try:
                    base_rotation = Rotation.from_quat(base_quaternion[[1, 2, 3, 0]])
                    yaw_rotation = Rotation.from_euler("z", yaw_deg, degrees=True)
                    perturbed_xyzw = (yaw_rotation * base_rotation).as_quat()
                    perturbed_quaternion = perturbed_xyzw[[3, 0, 1, 2]]
                    env = make_lift_env(
                        task_config={
                            "object_id": object_id,
                            "reward_shaping": False,
                            "terminate_on_success": False,
                            "placement_sampler": FixedTablePlacementSampler(
                                xy=(
                                    float(base_position[0] - 0.55 + dx),
                                    float(base_position[1] + dy),
                                ),
                                quaternion_wxyz=perturbed_quaternion,
                                world_z=float(base_position[2]),
                            ),
                        },
                        control_mode="ik",
                        enable_tactile_sensors=True,
                        episode_length=500,
                        render_mode=None,
                    )
                    try:
                        episode = execute_grasp(
                            source_episode.candidate,
                            seed=seed,
                            config=ExecutionConfig(
                                enable_tactile_sensors=True,
                                finger_preload=args.finger_preload,
                                thumb_grasp_preload=args.thumb_preload,
                                lift_height=args.lift_height,
                                verify_steps=args.verify_steps,
                                close_steps=args.close_steps,
                            ),
                            environment=env,
                        )
                    finally:
                        env.close()
                    execution_success = bool(episode.success)
                    episode.metadata.update(
                        {
                            "source_manifest": str(source),
                            "source_seed": seed,
                            "control_dt": 0.05,
                            "candidate_attempt": candidate_index,
                            "base_grasp_id": source.parent.name,
                            "perturbation": {
                                "x_m": dx,
                                "y_m": dy,
                                "yaw_degrees": yaw_deg,
                            },
                            "independent_execution": True,
                        }
                    )
                    attempt_manifest = episode.save(
                        case_destination / f"candidate_{candidate_index:02d}"
                    )
                    if execution_success:
                        verification = strict_replay_manifest(
                            attempt_manifest, profile=FINAL_PROFILE, use_cache=False
                        )
                        strict_success = bool(verification.success)
                        attempt_strict = asdict(verification)
                        if not strict_success:
                            attempt_reason = verification.verification_status
                    else:
                        attempt_reason = episode.failure_reason
                except Exception as exc:
                    attempt_reason = f"{type(exc).__name__}: {exc}"
                candidate_attempts.append(
                    {
                        "candidate_index": candidate_index,
                        "source_manifest": str(source),
                        "manifest": None if attempt_manifest is None else str(attempt_manifest),
                        "execution_success": execution_success,
                        "strict_success": strict_success,
                        "failure_reason": attempt_reason,
                        "failure_category": classify_failure(attempt_reason, attempt_strict),
                        "strict_metrics": attempt_strict,
                        "runtime_seconds": time.perf_counter() - attempt_started,
                    }
                )
                manifest = attempt_manifest
                failure_reason = attempt_reason
                strict_metrics = attempt_strict
                if strict_success:
                    break
            row = {
                "object_id": object_id,
                "source_manifest": str(source),
                "manifest": None if manifest is None else str(manifest),
                "seed": seed,
                "perturbation": {"x_m": dx, "y_m": dy, "yaw_degrees": yaw_deg},
                "execution_success": execution_success,
                "strict_success": strict_success,
                "accepted": strict_success,
                "failure_reason": failure_reason,
                "failure_category": classify_failure(failure_reason, strict_metrics),
                "strict_metrics": strict_metrics,
                "candidate_attempts": candidate_attempts,
                "runtime_seconds": time.perf_counter() - started,
            }
            rows.append(row)
            print(json.dumps(row), flush=True)
    accepted = [row for row in rows if row["accepted"]]
    summary = {
        "schema_version": 1,
        "profile": FINAL_PROFILE,
        "objects": list(objects),
        "requested_per_object": args.per_object,
        "attempts": len(rows),
        "accepted": len(accepted),
        "success_rate": len(accepted) / len(rows) if rows else 0.0,
        "total_runtime_seconds": sum(row["runtime_seconds"] for row in rows),
        "mean_runtime_seconds": (
            sum(row["runtime_seconds"] for row in rows) / len(rows) if rows else 0.0
        ),
        "failure_categories": dict(Counter(row["failure_category"] for row in rows)),
        "accepted_manifests": [row["manifest"] for row in accepted],
        "results": rows,
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return 0 if accepted else 2


def main(argv: list[str] | None = None) -> int:
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
