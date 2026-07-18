"""Reusable grasp and approach-path configuration search API."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

from source.assets import PROJECT_ROOT
from source.envs.manipulation.object_catalog import resolve_record, resolve_record_path
from source.grasping.hand_closure_search import HandClosureResult, search_hand_grasp
from source.grasping.mesh_pointcloud import SurfacePointCloud, sample_surface_pointcloud


@dataclass(frozen=True)
class GraspConfigSearchResult:
    """Artifacts returned by the non-visual search API."""

    output_path: Path
    mesh_path: Path
    cloud: SurfacePointCloud
    grasp: HandClosureResult


def grasp_config_name(object_id: str) -> str:
    return "".join(
        character if character.isalnum() or character in "-_" else "_"
        for character in object_id
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
    grasp: HandClosureResult,
) -> dict:
    return {
        "schema_version": 1,
        "object_id": object_id,
        "mesh": str(mesh_path),
        "mesh_center": cloud.center.tolist(),
        "mesh_scale": cloud.scale,
        "object_table_height": float(cloud.points[:, 2].min()),
        "contact_points": grasp.contact_points.tolist(),
        "contact_normals": grasp.contact_normals.tolist(),
        "hand_actuator_fractions": grasp.actuator_fractions.tolist(),
        "hand_actuator_values": grasp.hand.actuator_values.tolist(),
        "hand_translation": grasp.translation.tolist(),
        "hand_rotation_matrix": grasp.rotation_matrix.tolist(),
        "hand_closure": grasp.closure,
        "hand_maximum_penetration": grasp.maximum_penetration,
        "hand_maximum_noncontact_penetration": (
            grasp.maximum_noncontact_penetration
        ),
        "hand_mean_contact_distance": grasp.mean_contact_distance,
        "hand_contacting_fingers": list(grasp.contacting_fingers),
        "hand_force_closure_residual": grasp.force_closure_residual,
        "hand_palmward_force_component": grasp.palmward_force_component,
        "hand_palmward_direction": grasp.palmward_direction.tolist(),
        "hand_palmward_depth": grasp.palmward_depth,
        "hand_table_clearance": grasp.table_clearance,
        "hand_pca_axis_index": grasp.pca_axis_index,
        "hand_robustness_margin": grasp.robustness_margin,
        "hand_object_inside": grasp.object_inside_hand,
        "hand_preload_weights": grasp.preload_weights.tolist(),
        "approach_hand_translations": grasp.approach_translations.tolist(),
        "approach_hand_rotation_matrices": (
            grasp.approach_rotation_matrices.tolist()
        ),
        "approach_hand_actuator_fractions": (
            grasp.approach_actuator_fractions.tolist()
        ),
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
) -> GraspConfigSearchResult:
    """Search a grasp and approach path, then write their versioned JSON."""
    if (object_id is None) == (mesh is None):
        raise ValueError("Provide exactly one of object_id or mesh.")
    mesh_path = object_mesh_path(object_id) if mesh is None else Path(mesh)
    name = "custom_mesh" if object_id is None else grasp_config_name(object_id)
    output_path = (
        PROJECT_ROOT / "configs" / "grasps" / f"{name}.json"
        if output is None
        else Path(output)
    )
    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path
    cloud = sample_surface_pointcloud(
        mesh_path,
        count=points,
        target_size=target_size,
        seed=seed,
    )
    grasp = search_hand_grasp(
        cloud,
        samples=joint_candidates,
        seed=seed,
    )
    if not grasp.success:
        failures = []
        if grasp.maximum_penetration > 0.004:
            failures.append(
                f"pad_penetration={grasp.maximum_penetration:.4f}>0.0040m"
            )
        if grasp.maximum_noncontact_penetration > 0.0015:
            failures.append(
                "rigid_penetration="
                f"{grasp.maximum_noncontact_penetration:.4f}>0.0015m"
            )
        if not grasp.object_inside_hand:
            failures.append("object_outside_hand")
        if grasp.table_clearance < 0.005:
            failures.append(
                f"table_clearance={grasp.table_clearance:.4f}<0.0050m"
            )
        if not (
            4 in grasp.contacting_fingers
            and any(finger < 4 for finger in grasp.contacting_fingers)
        ):
            failures.append(
                f"no_opposing_contacts={grasp.contacting_fingers}"
            )
        if grasp.palmward_force_component < 0.0:
            failures.append(
                f"outward_force={grasp.palmward_force_component:.3f}"
            )
        if grasp.force_closure_residual > 0.35:
            failures.append(
                f"force_closure={grasp.force_closure_residual:.3f}>0.350"
            )
        detail = ", ".join(failures) or "unknown_constraint"
        raise RuntimeError(
            f"No valid grasp was found for {object_id or mesh_path!r}: {detail}."
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            _payload(
                object_id=object_id,
                mesh_path=mesh_path,
                cloud=cloud,
                grasp=grasp,
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
) -> Path:
    """Generate one object config and return its cached path."""
    return search_grasp_config(
        object_id=object_id,
        output=output,
        points=points,
        joint_candidates=joint_candidates,
        seed=seed,
        target_size=target_size,
    ).output_path
