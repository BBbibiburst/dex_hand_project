"""Reusable grasp and approach-path configuration search API."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path

import numpy as np

from source.assets import PROJECT_ROOT
from source.envs.manipulation.object_catalog import resolve_record, resolve_record_path
from source.grasping.hand_closure_search import HandClosureResult, search_hand_grasp
from source.grasping.mesh_pointcloud import SurfacePointCloud, sample_surface_pointcloud
from source.grasping.pika_gripper_search import PikaGraspResult, search_pika_grasp
from source.grasping.standalone_validator import (
    StandaloneValidationResult,
    validate_grasp_config,
)


@dataclass(frozen=True)
class GraspConfigSearchResult:
    """Artifacts returned by the non-visual search API."""

    output_path: Path
    mesh_path: Path
    cloud: SurfacePointCloud
    grasp: HandClosureResult | PikaGraspResult


@dataclass(frozen=True)
class ValidatedGraspConfigResult:
    """A grasp candidate that passed standalone dynamics validation."""

    output_path: Path
    selected_seed: int
    attempts_used: int
    validation: StandaloneValidationResult


def grasp_config_name(object_id: str) -> str:
    return "".join(
        character if character.isalnum() or character in "-_" else "_" for character in object_id
    )


def object_mesh_path(object_id: str) -> Path:
    record = resolve_record(object_id)
    root = resolve_record_path(record, "source_path")
    candidates = [
        root / name
        for name in record.get("model_files", ())
        if Path(name).suffix.lower() in {".stl", ".obj", ".ply"}
    ]
    visual = next((path for path in candidates if path.name == "textured.obj"), None)
    path = visual or next(iter(candidates), None)
    if path is None or not path.is_file():
        raise FileNotFoundError(f"No mesh found for {object_id!r}.")
    return path


def _payload(
    *,
    object_id: str | None,
    mesh_path: Path,
    cloud: SurfacePointCloud,
    grasp: HandClosureResult | PikaGraspResult,
    end_effector_name: str,
) -> dict:
    is_dex = isinstance(grasp, HandClosureResult)
    actuator_values = grasp.hand.actuator_values if is_dex else grasp.gripper.actuator_values
    preload_directions = np.ones_like(grasp.preload_weights) if is_dex else grasp.preload_directions
    return {
        "schema_version": 1,
        "object_id": object_id,
        "end_effector_name": end_effector_name,
        "mesh": str(mesh_path),
        "mesh_center": cloud.center.tolist(),
        "mesh_scale": cloud.scale,
        "object_table_height": float(cloud.points[:, 2].min()),
        "contact_points": grasp.contact_points.tolist(),
        "contact_normals": grasp.contact_normals.tolist(),
        "hand_actuator_fractions": grasp.actuator_fractions.tolist(),
        "hand_actuator_values": actuator_values.tolist(),
        "hand_preload_directions": preload_directions.tolist(),
        "hand_translation": grasp.translation.tolist(),
        "hand_rotation_matrix": grasp.rotation_matrix.tolist(),
        "hand_closure": grasp.closure if is_dex else None,
        "hand_maximum_penetration": grasp.maximum_penetration,
        "hand_maximum_noncontact_penetration": (grasp.maximum_noncontact_penetration),
        "hand_mean_contact_distance": grasp.mean_contact_distance,
        "hand_contacting_fingers": [int(index) for index in grasp.contacting_fingers],
        "hand_force_closure_residual": grasp.force_closure_residual,
        "hand_palmward_force_component": (grasp.palmward_force_component if is_dex else None),
        "hand_palmward_direction": (grasp.palmward_direction.tolist() if is_dex else None),
        "hand_palmward_depth": grasp.palmward_depth if is_dex else None,
        "hand_table_clearance": grasp.table_clearance,
        "hand_pca_axis_index": grasp.pca_axis_index,
        "hand_robustness_margin": grasp.robustness_margin,
        "hand_object_inside": grasp.object_inside_hand if is_dex else True,
        "hand_preload_weights": grasp.preload_weights.tolist(),
        "approach_hand_translations": grasp.approach_translations.tolist(),
        "approach_hand_rotation_matrices": (grasp.approach_rotation_matrices.tolist()),
        "approach_hand_actuator_fractions": (grasp.approach_actuator_fractions.tolist()),
        "hand_fit_success": grasp.success,
    }


def search_grasp_config(
    *,
    object_id: str | None = None,
    mesh: str | Path | None = None,
    output: str | Path | None = None,
    points: int = 2048,
    joint_candidates: int = 128,
    seed: int = 0,
    target_size: float = 0.09,
    end_effector_name: str = "dex_hand",
) -> GraspConfigSearchResult:
    """Search a grasp and approach path, then write their versioned JSON."""
    if (object_id is None) == (mesh is None):
        raise ValueError("Provide exactly one of object_id or mesh.")
    mesh_path = object_mesh_path(object_id) if mesh is None else Path(mesh)
    name = "custom_mesh" if object_id is None else grasp_config_name(object_id)
    default_directory = PROJECT_ROOT / "configs" / "grasps"
    if end_effector_name != "dex_hand":
        default_directory = default_directory / end_effector_name
    output_path = default_directory / f"{name}.json" if output is None else Path(output)
    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path
    cloud = sample_surface_pointcloud(
        mesh_path,
        count=points,
        target_size=target_size,
        seed=seed,
    )
    if end_effector_name == "dex_hand":
        grasp = search_hand_grasp(
            cloud,
            samples=joint_candidates,
            seed=seed,
        )
    elif end_effector_name == "pika_gripper":
        grasp = search_pika_grasp(
            cloud,
            samples=joint_candidates,
            seed=seed,
        )
    else:
        raise ValueError(f"Unsupported grasp end effector {end_effector_name!r}.")
    if not grasp.success:
        failures = []
        if grasp.maximum_penetration > 0.004:
            failures.append(f"pad_penetration={grasp.maximum_penetration:.4f}>0.0040m")
        if grasp.maximum_noncontact_penetration > 0.0015:
            failures.append(f"rigid_penetration={grasp.maximum_noncontact_penetration:.4f}>0.0015m")
        if is_dex := isinstance(grasp, HandClosureResult):
            object_inside = grasp.object_inside_hand
        else:
            object_inside = True
        if not object_inside:
            failures.append("object_outside_hand")
        if grasp.table_clearance < 0.005:
            failures.append(f"table_clearance={grasp.table_clearance:.4f}<0.0050m")
        opposing = (
            4 in grasp.contacting_fingers and any(finger < 4 for finger in grasp.contacting_fingers)
            if is_dex
            else len(grasp.contacting_fingers) == 2
        )
        if not opposing:
            failures.append(f"no_opposing_contacts={grasp.contacting_fingers}")
        if is_dex and grasp.palmward_force_component < 0.0:
            failures.append(f"outward_force={grasp.palmward_force_component:.3f}")
        if grasp.force_closure_residual > 0.35:
            failures.append(f"force_closure={grasp.force_closure_residual:.3f}>0.350")
        detail = ", ".join(failures) or "unknown_constraint"
        raise RuntimeError(f"No valid grasp was found for {object_id or mesh_path!r}: {detail}.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            _payload(
                object_id=object_id,
                mesh_path=mesh_path,
                cloud=cloud,
                grasp=grasp,
                end_effector_name=end_effector_name,
            ),
            indent=2,
        ),
        encoding="utf-8",
    )
    return GraspConfigSearchResult(
        output_path=output_path,
        mesh_path=mesh_path,
        cloud=cloud,
        grasp=grasp,
    )


def generate_grasp_config(
    object_id: str,
    *,
    output: str | Path | None = None,
    points: int = 2048,
    joint_candidates: int = 128,
    seed: int = 0,
    target_size: float = 0.09,
    end_effector_name: str = "dex_hand",
) -> Path:
    """Generate one object config and return its cached path."""
    return search_grasp_config(
        object_id=object_id,
        output=output,
        points=points,
        joint_candidates=joint_candidates,
        seed=seed,
        target_size=target_size,
        end_effector_name=end_effector_name,
    ).output_path


def generate_validated_grasp_config(
    object_id: str,
    *,
    output: str | Path | None = None,
    attempts: int = 3,
    points: int = 2048,
    joint_candidates: int = 128,
    seed: int = 0,
    target_size: float = 0.09,
    end_effector_name: str = "dex_hand",
    validation_seconds: float = 3.0,
    settle_seconds: float = 0.8,
    grip_preload: float = 0.25,
) -> ValidatedGraspConfigResult:
    """Search candidates and atomically publish the first dynamically stable grasp."""
    if attempts <= 0:
        raise ValueError("attempts must be positive.")
    name = grasp_config_name(object_id)
    default_directory = PROJECT_ROOT / "configs" / "grasps"
    if end_effector_name != "dex_hand":
        default_directory /= end_effector_name
    output_path = default_directory / f"{name}.json" if output is None else Path(output)
    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".candidate")
    failures: list[str] = []
    try:
        for attempt in range(attempts):
            candidate_seed = seed + attempt
            try:
                search_grasp_config(
                    object_id=object_id,
                    output=temporary_path,
                    points=points,
                    joint_candidates=joint_candidates,
                    seed=candidate_seed,
                    target_size=target_size,
                    end_effector_name=end_effector_name,
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
            if not validation.stable:
                failures.append(
                    f"seed={candidate_seed}: unstable "
                    f"drift={validation.position_drift:.4f}m "
                    f"rotation={validation.rotation_drift:.3f}rad "
                    f"drop={validation.vertical_drop:.4f}m "
                    f"contacts={validation.final_contacts}"
                )
                continue
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
