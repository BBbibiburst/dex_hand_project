"""Stable reusable API for grasp generation and validation-aware publication."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Callable

import numpy as np
import trimesh

from source.grasping.constants import DEFAULT_GRIP_PRELOAD
from source.grasping.standalone_validator import (
    TrajectoryValidationResult,
    resolve_payload_mesh_path,
    validate_grasp_config,
    validate_grasp_trajectory_payload,
)
from source.grasping.search.catalog import (
    ROOT, grasp_config_directory, grasp_config_name, load_cloud, resolve_object,
)
from source.grasping.search.common import progress
from source.grasping.search.devices import DEVICES
from source.grasping.search.engine import search
from source.grasping.search.hand_geometry import _open_fractions, surface_for
from source.grasping.search.planning import approach_direction_metadata, plan_approach
from source.grasping.search.scoring import evaluate
from source.grasping.search.serialization import payload
from source.grasping.search.types import (
    Candidate, Cloud, Device, GraspConfigSearchResult, ValidatedGraspConfigResult,
)

def _search_failure_detail(candidate: Candidate) -> str:
    reasons = ", ".join(candidate.rejection_reasons)
    return reasons or f"score={candidate.score:.4f}"


def select_executable_config(
    object_id: str | None,
    mesh_path: Path,
    cloud: Cloud,
    device: Device,
    candidates: list[Candidate],
    *,
    seed: int,
    candidate_filter: Callable[[dict], tuple[bool, dict]] | None = None,
) -> dict:
    """Select the first analytically valid candidate with an exact free approach."""
    open_surface = surface_for(device, _open_fractions(device), seed=seed + 50_000)
    surface_cache = {
        tuple(np.round(_open_fractions(device), 8)): open_surface,
    }
    for candidate_index, candidate in enumerate(candidates):
        # A task-conditioned run may deliberately pass an analytically
        # unstable but geometrically plausible seed to DexEvolve. It still
        # must have a collision-free, robot-reachable approach before the
        # expensive simulator refinement starts.
        if not candidate.valid and candidate_filter is None:
            continue
        alternatives = plan_approach(
            cloud,
            device,
            candidate,
            open_surface,
            surface_cache,
            seed=seed + 60_000 + candidate_index,
        )
        if not alternatives:
            candidate.valid = False
            candidate.rejection_reasons = (
                *candidate.rejection_reasons,
                "approach_object_collision",
            )
            candidate.score += 2.0
            continue
        candidate.approach_alternatives = alternatives
        candidate.approach_plan = alternatives[0]
        alternatives = candidate.approach_alternatives or (
            (candidate.approach_plan,) if candidate.approach_plan is not None else ()
        )
        ordered = [candidate, *(item for item in candidates if item is not candidate)]
        task_filter_rejected = False
        for plan in alternatives:
            candidate.approach_plan = plan
            candidate_payload = payload(
                object_id,
                mesh_path,
                cloud,
                device,
                ordered,
            )
            try:
                validate_grasp_trajectory_payload(candidate_payload)
            except ValueError:
                continue
            if candidate_filter is not None:
                accepted, metadata = candidate_filter(candidate_payload)
                if not accepted:
                    task_filter_rejected = True
                    continue
                candidate_payload.update(metadata)
            candidate.approach_table_clearance = plan.minimum_table_clearance
            candidates[:] = ordered
            return candidate_payload
        candidate.valid = False
        candidate.rejection_reasons = (
            *candidate.rejection_reasons,
            "robot_task_infeasible"
            if task_filter_rejected
            else "mujoco_approach_collision",
        )
        candidate.score += 2.0
    candidates.sort(key=lambda item: (not item.valid, item.score))
    return payload(object_id, mesh_path, cloud, device, candidates)


def replan_evolved_payload(
    evolved: dict,
    *,
    seed: int = 0,
    point_count: int = 2048,
) -> dict:
    """Plan a fresh collision-free approach for an evolved final grasp."""
    device_name = evolved.get("end_effector_name", "dex_hand")
    try:
        device = DEVICES[device_name]
    except KeyError as exc:
        raise ValueError(f"Unsupported end effector {device_name!r}.") from exc
    mesh_path = resolve_payload_mesh_path(evolved["mesh"])
    loaded = trimesh.load_mesh(mesh_path, process=True)
    mesh = loaded.to_geometry() if isinstance(loaded, trimesh.Scene) else loaded
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError(f"No triangle mesh in {mesh_path}")
    raw_extent = max(float(np.ptp(np.asarray(mesh.vertices), axis=0).max()), 1e-9)
    target_size = raw_extent * float(evolved["mesh_scale"])
    cloud = load_cloud(mesh_path, count=point_count, target_size=target_size, seed=seed)
    fractions = np.asarray(evolved["hand_actuator_fractions"], dtype=np.float64)
    final_surface = surface_for(device, fractions, seed=seed + 10_000)
    candidate = evaluate(
        cloud,
        device,
        final_surface,
        np.asarray(evolved["hand_rotation_matrix"], dtype=np.float64),
        np.asarray(evolved["hand_translation"], dtype=np.float64),
        roll_index=int(evolved.get("hand_orientation_roll_index", 0)),
        full_checks=True,
    )
    open_fractions = _open_fractions(device)
    open_surface = surface_for(device, open_fractions, seed=seed + 20_000)
    surface_cache = {tuple(np.round(open_fractions, 8)): open_surface}
    alternatives = plan_approach(
        cloud,
        device,
        candidate,
        open_surface,
        surface_cache,
        seed=seed + 30_000,
    )
    errors = []
    for plan in alternatives:
        replanned = dict(evolved)
        rotation = np.asarray(evolved["hand_rotation_matrix"], dtype=np.float64)
        replanned.update(
            approach_direction=plan.direction.tolist(),
            **approach_direction_metadata(plan.direction),
            approach_hand_translations=plan.approach_translations.tolist(),
            approach_hand_rotation_matrices=np.repeat(
                rotation[None, :, :], len(plan.approach_translations), axis=0
            ).tolist(),
            approach_hand_actuator_fractions=plan.approach_fractions.tolist(),
            grasp_hand_translations=plan.grasp_translations.tolist(),
            grasp_hand_rotation_matrices=np.repeat(
                rotation[None, :, :], len(plan.grasp_translations), axis=0
            ).tolist(),
            grasp_hand_actuator_fractions=plan.grasp_fractions.tolist(),
            approach_minimum_object_clearance=plan.minimum_object_clearance,
            approach_minimum_table_clearance=plan.minimum_table_clearance,
            grasp_trajectory_maximum_penetration=plan.maximum_grasp_penetration,
            grasp_trajectory_maximum_rigid_penetration=(plan.maximum_grasp_rigid_penetration),
            trajectory_replanned=True,
            approach_planner="reverse_with_upward_arc",
        )
        try:
            validate_grasp_trajectory_payload(replanned)
        except ValueError as exc:
            errors.append(str(exc))
            continue
        return replanned
    detail = " | ".join(errors[:4]) or "no collision-free approach plan"
    raise ValueError(f"Unable to replan evolved grasp trajectory: {detail}")


def search_grasp_config(
    *,
    object_id: str | None = None,
    mesh: str | Path | None = None,
    output: str | Path | None = None,
    points: int = 2048,
    joint_candidates: int = 128,
    surface_anchors: int = 24,
    rolls_per_anchor: int = 8,
    coarse_keep: int = 24,
    top_k: int = 8,
    support_margin: float = 0.008,
    seed: int = 0,
    target_size: float = 0.09,
    end_effector_name: str = "dex_hand",
    require_valid: bool = True,
    publish_invalid: bool = False,
    generator: str = "heuristic",
    graspqp_iterations: int = 120,
    candidate_filter: Callable[[dict], tuple[bool, dict]] | None = None,
) -> GraspConfigSearchResult:
    """Run the new two-stage search and write a production-schema grasp config."""
    if (object_id is None) == (mesh is None):
        raise ValueError("Provide exactly one of object_id or mesh.")
    if points <= 0:
        raise ValueError("points must be positive.")
    for name, value in (
        ("joint_candidates", joint_candidates),
        ("surface_anchors", surface_anchors),
        ("rolls_per_anchor", rolls_per_anchor),
        ("coarse_keep", coarse_keep),
        ("top_k", top_k),
    ):
        if value <= 0:
            raise ValueError(f"{name} must be positive.")
    if support_margin < 0.0:
        raise ValueError("support_margin must be non-negative.")
    if target_size <= 0.0:
        raise ValueError("target_size must be positive.")
    if generator not in {"heuristic", "graspqp"}:
        raise ValueError("generator must be 'heuristic' or 'graspqp'.")
    if graspqp_iterations <= 0:
        raise ValueError("graspqp_iterations must be positive.")
    try:
        device = DEVICES[end_effector_name]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported end effector {end_effector_name!r}; available={tuple(DEVICES)}."
        ) from exc

    mesh_path = resolve_object(object_id) if mesh is None else Path(mesh)
    if not mesh_path.is_absolute():
        mesh_path = ROOT / mesh_path
    if not mesh_path.is_file():
        raise FileNotFoundError(f"Object mesh does not exist: {mesh_path}")
    name = "custom_mesh" if object_id is None else grasp_config_name(object_id)
    output_path = (
        grasp_config_directory(end_effector_name) / f"{name}.json"
        if output is None
        else Path(output)
    )
    if not output_path.is_absolute():
        output_path = ROOT / output_path

    progress(f"[setup] loading object: {mesh_path}")
    cloud = load_cloud(
        mesh_path,
        count=points,
        target_size=target_size,
        seed=seed,
    )
    candidates = search(
        cloud,
        device,
        joint_candidates=joint_candidates,
        anchor_count=surface_anchors,
        rolls_per_anchor=rolls_per_anchor,
        coarse_keep=coarse_keep,
        top_k=top_k,
        support_margin=support_margin,
        seed=seed,
        generator=generator,
        graspqp_iterations=graspqp_iterations,
    )
    config = select_executable_config(
        object_id,
        mesh_path,
        cloud,
        device,
        candidates,
        seed=seed,
        candidate_filter=candidate_filter,
    )
    result = GraspConfigSearchResult(
        output_path=output_path,
        mesh_path=mesh_path,
        cloud=cloud,
        candidates=tuple(candidates),
        config=config,
        published=False,
    )
    if require_valid and not config["hand_fit_success"]:
        raise RuntimeError(
            f"No valid grasp was found for {object_id or mesh_path!r}: "
            f"{_search_failure_detail(result.grasp)}."
        )
    published = bool(config["hand_fit_success"] or publish_invalid)
    if published:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = output_path.with_suffix(output_path.suffix + f".tmp-{os.getpid()}")
        temporary_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
        os.replace(temporary_path, output_path)
        progress(f"[output] wrote {output_path}")
    return GraspConfigSearchResult(
        output_path=output_path,
        mesh_path=mesh_path,
        cloud=cloud,
        candidates=tuple(candidates),
        config=config,
        published=published,
    )


def generate_grasp_config(
    object_id: str,
    *,
    output: str | Path | None = None,
    **search_kwargs,
) -> Path:
    """Generate one object config and return its cached path."""
    return search_grasp_config(
        object_id=object_id,
        output=output,
        **search_kwargs,
    ).output_path


def generate_validated_grasp_config(
    object_id: str,
    *,
    output: str | Path | None = None,
    attempts: int = 3,
    validation_seconds: float = 3.0,
    settle_seconds: float = 0.8,
    grip_preload: float = DEFAULT_GRIP_PRELOAD,
    **search_kwargs,
) -> ValidatedGraspConfigResult:
    """Publish the first candidate that passes trajectory-and-hold validation."""
    if attempts <= 0:
        raise ValueError("attempts must be positive.")
    end_effector_name = str(search_kwargs.get("end_effector_name", "dex_hand"))
    name = grasp_config_name(object_id)
    output_path = (
        grasp_config_directory(end_effector_name) / f"{name}.json"
        if output is None
        else Path(output)
    )
    if not output_path.is_absolute():
        output_path = ROOT / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".candidate")
    failures: list[str] = []

    try:
        base_seed = int(search_kwargs.pop("seed", 0))
        for attempt in range(attempts):
            candidate_seed = base_seed + attempt
            try:
                search_grasp_config(
                    object_id=object_id,
                    output=temporary_path,
                    seed=candidate_seed,
                    **search_kwargs,
                )
                validation = validate_grasp_config(
                    temporary_path,
                    seconds=validation_seconds,
                    settle_seconds=settle_seconds,
                    grip_preload=grip_preload,
                )
            except Exception as exc:
                failures.append(f"seed={candidate_seed}: {exc}")
                continue
            if not validation.trajectory_hold_stable:
                failures.append(
                    f"seed={candidate_seed}: unstable "
                    f"drift={validation.position_drift:.4f}m "
                    f"rotation={validation.rotation_drift:.3f}rad "
                    f"drop={validation.vertical_drop:.4f}m "
                    f"contacts={validation.final_contacts}"
                )
                continue
            payload = json.loads(temporary_path.read_text(encoding="utf-8"))
            payload.update(
                trajectory_collision_free=True,
                trajectory_hold_stable=True,
                validation_stage="trajectory_hold_stable",
            )
            temporary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            os.replace(temporary_path, output_path)
            return ValidatedGraspConfigResult(
                output_path=output_path,
                selected_seed=candidate_seed,
                attempts_used=attempt + 1,
                validation=validation,
            )
    finally:
        temporary_path.unlink(missing_ok=True)
    detail = " | ".join(failures) or "no candidates evaluated"
    raise RuntimeError(f"No dynamically stable grasp was found for {object_id!r}: {detail}")
