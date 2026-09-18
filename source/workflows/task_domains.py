"""Outcome-independent task-domain rules for manipulation benchmarks."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping


TASK_DOMAIN_VERSIONS = {
    "stack": "stack-task-domain-v1",
    "push": "push-task-domain-v1",
}


@dataclass(frozen=True)
class TaskDomainAssessment:
    object_id: str
    task: str
    domain_version: str
    eligible: bool
    reasons: tuple[str, ...]
    measurements: dict[str, float | bool]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def assess_task_domain(
    task: str,
    object_id: str,
    geometry: Mapping[str, Any],
    stability: Mapping[str, Any],
) -> TaskDomainAssessment:
    """Classify an object using geometry and free-settling data only.

    Policy rollouts and task success are deliberately absent from this API, so
    a failed controller can never change the benchmark population.
    """
    task = task.lower()
    if task not in TASK_DOMAIN_VERSIONS:
        raise ValueError(f"Task-domain assessment is not defined for {task!r}.")
    extents = tuple(sorted(float(value) for value in geometry["extents_m"]))
    if len(extents) != 3 or any(value <= 0.0 for value in extents):
        raise ValueError("geometry.extents_m must contain three positive values.")
    minimum, middle, maximum = extents
    sphericity = float(geometry["sphericity"])
    convexity = float(geometry["convexity"])
    axis_ratio = float(geometry["axis_ratio"])
    stable = bool(stability.get("stable", False)) and not stability.get("error")
    reasons: list[str] = []
    if not stable:
        reasons.append("initial_pose_is_not_stable")

    if task == "stack":
        if minimum < 0.020:
            reasons.append("support_dimension_below_20mm")
        if sphericity > 0.85:
            reasons.append("too_round_for_stable_support")
        if convexity < 0.65:
            reasons.append("support_geometry_too_concave")
        if axis_ratio > 4.0:
            reasons.append("shape_too_slender_for_stable_stack")
    else:
        # This domain represents quasi-static four-finger face pushing, not
        # arbitrary rolling or poking. The transverse span must cover a useful
        # fraction of the tactile plane and the surface must be regular enough
        # to sustain a distributed contact patch.
        if minimum < 0.025:
            reasons.append("contact_span_below_25mm")
        if sphericity > 0.95:
            reasons.append("too_round_for_quasi_static_face_push")
        if convexity < 0.55:
            reasons.append("contact_surface_too_concave")
        if axis_ratio > 4.5:
            reasons.append("shape_too_slender_for_four_finger_push")

    measurements: dict[str, float | bool] = {
        "minimum_extent_m": minimum,
        "middle_extent_m": middle,
        "maximum_extent_m": maximum,
        "sphericity": sphericity,
        "convexity": convexity,
        "axis_ratio": axis_ratio,
        "initial_pose_stable": stable,
        "horizontal_displacement_m": float(
            stability.get("horizontal_displacement_m", float("nan"))
        ),
        "orientation_change_deg": float(stability.get("orientation_change_deg", float("nan"))),
    }
    return TaskDomainAssessment(
        object_id=object_id,
        task=task,
        domain_version=TASK_DOMAIN_VERSIONS[task],
        eligible=not reasons,
        reasons=tuple(reasons),
        measurements=measurements,
    )


def build_task_domain_manifest(
    task: str,
    selection: Mapping[str, Any],
    stability_report: Mapping[str, Any],
) -> dict[str, Any]:
    """Build a versioned, auditable task-valid selection manifest."""
    source_rows = selection.get("objects")
    stability_rows = stability_report.get("results")
    if not isinstance(source_rows, list) or not isinstance(stability_rows, list):
        raise ValueError("Selection objects and stability results must be lists.")
    stability_by_id = {str(row["object_id"]): row for row in stability_rows}
    assessments: list[TaskDomainAssessment] = []
    for row in source_rows:
        if not isinstance(row, Mapping):
            raise ValueError("Task-domain generation requires geometry-bearing object rows.")
        object_id = str(row["object_id"])
        if object_id not in stability_by_id:
            raise ValueError(f"Stability report is missing {object_id!r}.")
        assessments.append(
            assess_task_domain(task, object_id, row["geometry"], stability_by_id[object_id])
        )
    accepted = [item.to_dict() for item in assessments if item.eligible]
    excluded = [item.to_dict() for item in assessments if not item.eligible]
    return {
        "schema_version": 1,
        "name": f"{task.title()} task-valid benchmark",
        "task": task,
        "selection_stage": "task_valid",
        "domain_version": TASK_DOMAIN_VERSIONS[task],
        "selection_method": "geometry_and_initial_pose_stability",
        "source_selection": stability_report.get("selection"),
        "source_count": len(assessments),
        "count": len(accepted),
        "excluded_count": len(excluded),
        "objects": accepted,
        "excluded": excluded,
    }


def is_prevalidated_task_manifest(payload: Any, task: str) -> bool:
    """Return whether a payload is a current task-valid manifest for ``task``."""
    return bool(
        isinstance(payload, Mapping)
        and payload.get("selection_stage") == "task_valid"
        and payload.get("task") == task
        and payload.get("domain_version") == TASK_DOMAIN_VERSIONS.get(task)
    )
