"""Compile a local wrist-pose lattice around UltraDexGrasp priors.

The lattice is built in object coordinates.  Candidate wrist pose edits are
prechecked with the real RM75B CPU MuJoCo IK before a full approach/close/hold/
lift execution is recorded.  MJWarp RL later chooses among these reachable
compiled trajectories with a continuous 6D wrist-edit action and separately
edits the six physical Dex Hand actuators.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from source.envs.manipulation import make_lift_env
from source.grasp_pipeline.reference import STAGE_CODES
from source.ultradexgrasp.contracts import DemonstrationEpisode, GraspCandidate
from source.ultradexgrasp.executor import (
    ExecutionConfig,
    execute_grasp,
    rank_candidates_for_scene,
)

_REQUIRED_STAGES = {
    STAGE_CODES["approach"],
    STAGE_CODES["close"],
    STAGE_CODES["hold"],
    STAGE_CODES["lift"],
    STAGE_CODES["verify"],
}


@dataclass(frozen=True)
class GraspEditTemplate:
    label: str
    manifest: Path
    source_manifest: Path
    base_seed_index: int
    source_lift: float
    success: bool
    translation_offset: tuple[float, float, float]
    rotation_offset_degrees: tuple[float, float, float]  # roll, pitch, yaw
    precheck_score: float
    precheck_position_error: float
    precheck_orientation_error: float

    @property
    def wrist_edit(self) -> tuple[float, float, float, float, float, float]:
        return (*self.translation_offset, *self.rotation_offset_degrees)


def _slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def _full_episode(path: Path, object_id: str) -> DemonstrationEpisode | None:
    try:
        episode = DemonstrationEpisode.load(path)
    except (OSError, ValueError, KeyError):
        return None
    if episode.object_id != object_id:
        return None
    stages = {int(value) for value in np.asarray(episode.arrays["stage"]).reshape(-1)}
    if not _REQUIRED_STAGES.issubset(stages):
        return None
    return episode


def discover_ultra_attempts(
    object_id: str,
    *,
    roots: tuple[Path, ...] = (
        Path("outputs/ultradexgrasp"),
        Path("outputs/ultradexgrasp_catalog"),
    ),
    maximum: int = 3,
) -> list[tuple[Path, DemonstrationEpisode]]:
    if maximum <= 0:
        raise ValueError("maximum must be positive.")
    slug = _slug(object_id)
    rows: list[tuple[Path, DemonstrationEpisode]] = []
    seen: set[Path] = set()
    for root in roots:
        object_dir = root / slug
        if not object_dir.is_dir():
            continue
        for pattern in ("seed_*/manifest.json", "seed_*/attempts/*/manifest.json"):
            for manifest in sorted(object_dir.glob(pattern)):
                manifest = manifest.resolve()
                if manifest in seen:
                    continue
                seen.add(manifest)
                episode = _full_episode(manifest, object_id)
                if episode is not None:
                    rows.append((manifest, episode))

    def key(row: tuple[Path, DemonstrationEpisode]):
        _, episode = row
        lift = float(episode.metadata.get("object_lift", 0.0))
        # Static IK reachability is not enough: a nominally reachable wrist can
        # still miss a thin object by centimetres after the dynamic approach.
        # Prefer references whose recorded controller actually reached the
        # grasp pose. This makes flat-box searches favour the well-tracked
        # side approaches instead of a high-scoring but dynamically inaccurate
        # top approach.
        position_error = float(episode.metadata.get("approach_position_error", np.inf))
        orientation_error = float(
            episode.metadata.get("approach_orientation_error", np.inf)
        )
        return (
            1 if episode.success else 0,
            lift,
            -position_error,
            -orientation_error,
        )

    rows.sort(key=key, reverse=True)
    return rows[:maximum]


def _axis_rotation(axis: str, degrees: float) -> np.ndarray:
    angle = np.deg2rad(float(degrees))
    c, s = float(np.cos(angle)), float(np.sin(angle))
    if axis == "x":
        return np.asarray([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])
    if axis == "y":
        return np.asarray([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])
    if axis == "z":
        return np.asarray([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    raise ValueError(axis)


def _rotation_delta(roll: float, pitch: float, yaw: float) -> np.ndarray:
    # Object-frame ZYX perturbation.  This edits wrist orientation in place;
    # unlike v8 it does not rotate the hand root position around the object.
    return (
        _axis_rotation("z", yaw)
        @ _axis_rotation("y", pitch)
        @ _axis_rotation("x", roll)
    )


def edit_candidate_pose(
    candidate: GraspCandidate,
    *,
    translation_offset: tuple[float, float, float],
    rotation_offset_degrees: tuple[float, float, float],
    base_index: int,
) -> GraspCandidate:
    offset = np.asarray(translation_offset, dtype=np.float64)
    roll, pitch, yaw = rotation_offset_degrees
    delta = _rotation_delta(roll, pitch, yaw)
    new_translation = candidate.hand_translation + offset
    # Keep contact metadata rigidly attached to the edited hand root.
    relative_contacts = np.asarray(candidate.contact_points) - candidate.hand_translation
    new_contacts = new_translation + relative_contacts @ delta.T
    new_normals = np.asarray(candidate.contact_normals) @ delta.T
    metrics = dict(candidate.metrics)
    metrics.update(
        {
            "lattice_base_index": float(base_index),
            "lattice_dx": float(offset[0]),
            "lattice_dy": float(offset[1]),
            "lattice_dz": float(offset[2]),
            "lattice_roll_degrees": float(roll),
            "lattice_pitch_degrees": float(pitch),
            "lattice_yaw_degrees": float(yaw),
        }
    )
    return replace(
        candidate,
        hand_translation=new_translation,
        hand_rotation_matrix=delta @ candidate.hand_rotation_matrix,
        contact_points=new_contacts,
        contact_normals=new_normals,
        metrics=metrics,
    )


def local_wrist_lattice(
    *,
    translation_step: float = 0.01,
    rotation_step_degrees: float = 15.0,
) -> tuple[tuple[tuple[float, float, float], tuple[float, float, float]], ...]:
    """Return a compact deterministic 6D local wrist lattice.

    The set intentionally contains single-axis edits plus several coupled
    yaw/XY and pitch/Z edits.  It avoids the v8 failure mode where a 90-degree
    rotation also swept the hand root around the object and became grossly
    unreachable for RM75B.
    """
    if translation_step <= 0.0 or rotation_step_degrees <= 0.0:
        raise ValueError("lattice steps must be positive.")
    t = float(translation_step)
    r = float(rotation_step_degrees)
    zero_t = (0.0, 0.0, 0.0)
    zero_r = (0.0, 0.0, 0.0)
    rows = [(zero_t, zero_r)]

    # Orientation-only local edits.
    for yaw in (-3 * r, -2 * r, -r, r, 2 * r, 3 * r):
        rows.append((zero_t, (0.0, 0.0, yaw)))
    for pitch in (-2 * r, -r, r, 2 * r):
        rows.append((zero_t, (0.0, pitch, 0.0)))
    for roll in (-r, r):
        rows.append((zero_t, (roll, 0.0, 0.0)))

    # Translation-only edits.
    for axis in range(3):
        for sign in (-1.0, 1.0):
            offset = [0.0, 0.0, 0.0]
            offset[axis] = sign * t
            rows.append((tuple(offset), zero_r))

    # Coupled edits let IK recover orientation changes with a small position
    # adjustment, which pure yaw rotation could not do in v8.
    for yaw in (-2 * r, -r, r, 2 * r):
        for axis in (0, 1):
            for sign in (-1.0, 1.0):
                offset = [0.0, 0.0, 0.0]
                offset[axis] = sign * t
                rows.append((tuple(offset), (0.0, 0.0, yaw)))
    for pitch in (-r, r):
        for sign in (-1.0, 1.0):
            rows.append(((0.0, 0.0, sign * t), (0.0, pitch, 0.0)))

    # Stable de-duplication while preserving the deliberate search order.
    unique = []
    seen = set()
    for translation, rotation in rows:
        key = tuple(round(x, 9) for x in (*translation, *rotation))
        if key not in seen:
            seen.add(key)
            unique.append((translation, rotation))
    return tuple(unique)


def _code(value: float, scale: float, prefix: str) -> str:
    integer = round(abs(value) * scale)
    sign = "p" if value >= 0.0 else "m"
    return f"{prefix}{sign}{integer:03d}"


def _label(
    base_index: int,
    seed_index: int,
    translation: tuple[float, float, float],
    rotation: tuple[float, float, float],
) -> str:
    dx, dy, dz = translation
    roll, pitch, yaw = rotation
    return "_".join(
        [
            f"b{base_index:02d}",
            f"s{seed_index:03d}",
            _code(dx, 1000.0, "x"),
            _code(dy, 1000.0, "y"),
            _code(dz, 1000.0, "z"),
            _code(roll, 1.0, "r"),
            _code(pitch, 1.0, "p"),
            _code(yaw, 1.0, "w"),
        ]
    )


def _write_index(path: Path, object_id: str, templates: list[GraspEditTemplate]) -> Path:
    payload = {
        "schema_version": 2,
        "object_id": object_id,
        "templates": [
            {
                "label": item.label,
                "manifest": str(item.manifest),
                "source_manifest": str(item.source_manifest),
                "base_seed_index": item.base_seed_index,
                "source_lift": item.source_lift,
                "success": item.success,
                "translation_offset": list(item.translation_offset),
                "rotation_offset_degrees": list(item.rotation_offset_degrees),
                "precheck_score": item.precheck_score,
                "precheck_position_error": item.precheck_position_error,
                "precheck_orientation_error": item.precheck_orientation_error,
            }
            for item in templates
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)
    return path


def build_grasp_edit_templates(
    object_id: str,
    *,
    output_root: Path = Path("outputs/grasp_edit_lattice"),
    ultra_roots: tuple[Path, ...] = (
        Path("outputs/ultradexgrasp"),
        Path("outputs/ultradexgrasp_catalog"),
    ),
    base_candidates: int = 3,
    translation_step: float = 0.01,
    rotation_step_degrees: float = 15.0,
    maximum_templates: int = 12,
    maximum_executions: int = 32,
    seed: int = 0,
    overwrite: bool = False,
    failed_only: bool = False,
    verbose: bool = False,
) -> tuple[GraspEditTemplate, ...]:
    if maximum_templates <= 0 or maximum_executions <= 0:
        raise ValueError("lattice template/execution limits must be positive.")
    sources = discover_ultra_attempts(
        object_id,
        roots=ultra_roots,
        maximum=base_candidates,
    )
    if not sources:
        raise FileNotFoundError(
            f"No full UltraDexGrasp attempts found for {object_id}."
        )

    object_output = output_root / _slug(object_id)
    object_output.mkdir(parents=True, exist_ok=True)
    execution = ExecutionConfig()
    rank_env = make_lift_env(
        task_config={"object_id": object_id},
        control_mode="ik",
        enable_tactile_sensors=False,
        render_mode=None,
    )
    exec_env = make_lift_env(
        task_config={
            "object_id": object_id,
            "reward_shaping": False,
            "terminate_on_success": False,
        },
        control_mode="ik",
        enable_tactile_sensors=False,
        render_mode=None,
    )
    templates: list[GraspEditTemplate] = []
    stats = {"source": 0, "reused": 0, "compiled": 0, "rejected": 0, "skipped_success": 0}
    try:
        observation, _ = rank_env.reset(seed=seed)
        variants: list[GraspCandidate] = []
        source_by_base: dict[int, tuple[Path, DemonstrationEpisode]] = {}
        for base_index, source in enumerate(sources):
            source_manifest, source_episode = source
            source_by_base[base_index] = source
            for translation, rotation in local_wrist_lattice(
                translation_step=translation_step,
                rotation_step_degrees=rotation_step_degrees,
            ):
                variants.append(
                    edit_candidate_pose(
                        source_episode.candidate,
                        translation_offset=translation,
                        rotation_offset_degrees=rotation,
                        base_index=base_index,
                    )
                )

        ranked = rank_candidates_for_scene(
            rank_env,
            tuple(variants),
            observation["object_pos"],
            observation["object_quat"],
            pregrasp_distance=execution.pregrasp_distance,
        )
        reachable = [
            result
            for result in ranked
            if result.maximum_position_error <= execution.position_tolerance
            and result.maximum_orientation_error <= execution.orientation_tolerance
        ]
        if verbose:
            print(
                f"[lattice:precheck] object={object_id} candidates={len(variants)} "
                f"reachable={len(reachable)} pos_tol={execution.position_tolerance:.3f} "
                f"rot_tol={execution.orientation_tolerance:.3f}",
                flush=True,
            )

        # Prefer low IK error, but add a small pose-edit norm so the nominal
        # neighborhood is compiled before more aggressive variants.
        def compile_key(result):
            metrics = result.candidate.metrics
            tnorm = np.linalg.norm(
                [metrics["lattice_dx"], metrics["lattice_dy"], metrics["lattice_dz"]]
            ) / max(translation_step, 1e-9)
            rnorm = np.linalg.norm(
                [
                    metrics["lattice_roll_degrees"],
                    metrics["lattice_pitch_degrees"],
                    metrics["lattice_yaw_degrees"],
                ]
            ) / max(rotation_step_degrees, 1e-9)
            base_index = round(metrics["lattice_base_index"])
            source_episode = source_by_base[base_index][1]
            dynamic_position_error = float(
                source_episode.metadata.get(
                    "approach_position_error", execution.position_tolerance
                )
            )
            dynamic_orientation_error = float(
                source_episode.metadata.get(
                    "approach_orientation_error", execution.orientation_tolerance
                )
            )
            tracking_penalty = (
                0.10
                * dynamic_position_error
                / max(execution.position_tolerance, 1e-9)
                + 0.05
                * dynamic_orientation_error
                / max(execution.orientation_tolerance, 1e-9)
            )
            return float(result.score + 0.015 * (tnorm + rnorm) + tracking_penalty)

        reachable.sort(key=compile_key)
        executions = 0
        for result in reachable:
            if len(templates) >= maximum_templates or executions >= maximum_executions:
                break
            candidate = result.candidate
            metrics = candidate.metrics
            base_index = round(metrics["lattice_base_index"])
            source_manifest, source_episode = source_by_base[base_index]
            translation = (
                float(metrics["lattice_dx"]),
                float(metrics["lattice_dy"]),
                float(metrics["lattice_dz"]),
            )
            rotation = (
                float(metrics["lattice_roll_degrees"]),
                float(metrics["lattice_pitch_degrees"]),
                float(metrics["lattice_yaw_degrees"]),
            )
            label = _label(
                base_index,
                source_episode.candidate.seed_index,
                translation,
                rotation,
            )
            directory = object_output / label
            manifest = directory / "manifest.json"
            nominal = all(abs(value) < 1e-10 for value in (*translation, *rotation))

            if nominal:
                episode = source_episode
                generated_manifest = source_manifest
                stats["source"] += 1
                if verbose:
                    print(
                        f"[lattice:source] {label} success={episode.success} "
                        f"score={result.score:.3f}",
                        flush=True,
                    )
            else:
                cached = None if overwrite else _full_episode(manifest, object_id)
                if cached is not None:
                    episode = cached
                    generated_manifest = manifest.resolve()
                    stats["reused"] += 1
                    if verbose:
                        print(
                            f"[lattice:reuse] {label} success={episode.success} "
                            f"score={result.score:.3f}",
                            flush=True,
                        )
                else:
                    executions += 1
                    episode = execute_grasp(
                        candidate,
                        seed=seed + base_index,
                        config=execution,
                        render_mode=None,
                        environment=exec_env,
                    )
                    episode.metadata["grasp_edit_lattice"] = {
                        "source_manifest": str(source_manifest),
                        "translation_offset": list(translation),
                        "rotation_offset_degrees": list(rotation),
                        "precheck_score": float(result.score),
                        "precheck_position_error": float(result.maximum_position_error),
                        "precheck_orientation_error": float(result.maximum_orientation_error),
                    }
                    generated_manifest = episode.save(directory)
                    full = _full_episode(generated_manifest, object_id)
                    if full is None:
                        stats["rejected"] += 1
                        if verbose:
                            print(
                                f"[lattice:reject-execution] {label} "
                                f"terminal={episode.terminal_stage} reason={episode.failure_reason}",
                                flush=True,
                            )
                        continue
                    stats["compiled"] += 1
                    if verbose:
                        print(
                            f"[lattice:compiled] {label} success={episode.success} "
                            f"lift={float(episode.metadata.get('object_lift', 0.0)):.3f}m",
                            flush=True,
                        )

            if failed_only and episode.success:
                stats["skipped_success"] += 1
                if verbose:
                    print(f"[lattice:skip-success] {label}", flush=True)
                continue
            templates.append(
                GraspEditTemplate(
                    label=label,
                    manifest=Path(generated_manifest).resolve(),
                    source_manifest=source_manifest,
                    base_seed_index=int(source_episode.candidate.seed_index),
                    source_lift=float(episode.metadata.get("object_lift", 0.0)),
                    success=bool(episode.success),
                    translation_offset=translation,
                    rotation_offset_degrees=rotation,
                    precheck_score=float(result.score),
                    precheck_position_error=float(result.maximum_position_error),
                    precheck_orientation_error=float(result.maximum_orientation_error),
                )
            )
    finally:
        rank_env.close()
        exec_env.close()

    if not templates:
        mode = "failed" if failed_only else "usable"
        raise RuntimeError(
            f"No {mode} reachable wrist-lattice templates were compiled for {object_id}. "
            "Increase --lattice-max-executions or adjust the local lattice steps."
        )
    _write_index(object_output / ("index_failed.json" if failed_only else "index.json"), object_id, templates)
    print(
        f"[lattice] candidates={len(variants)} reachable={len(reachable)} "
        f"templates={len(templates)} source={stats['source']} "
        f"compiled={stats['compiled']} reused={stats['reused']} "
        f"rejected={stats['rejected']} skipped_success={stats['skipped_success']} "
        f"failed_only={failed_only}",
        flush=True,
    )
    return tuple(templates)
