"""Serialization of selected grasp candidates into the production schema."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from source.grasping.constants import GRASP_CONFIG_SCHEMA_VERSION, GRASP_SEARCH_STRATEGY
from source.grasping.search.planning import approach, approach_direction_metadata
from source.grasping.search.types import Candidate, Cloud, Device

def candidate_summary(candidate: Candidate) -> dict:
    return {
        "score": candidate.score,
        "valid": candidate.valid,
        "rejection_reasons": list(candidate.rejection_reasons),
        "anchor_index": candidate.anchor_index,
        "roll_index": candidate.roll_index,
        "translation": candidate.translation.tolist(),
        "rotation_matrix": candidate.rotation.tolist(),
        "actuator_fractions": candidate.surface.fractions.tolist(),
        "contacts": list(candidate.contacts),
        "contact_points": candidate.contact_points.tolist(),
        "contact_normals": candidate.contact_normals.tolist(),
        "penetration": candidate.penetration,
        "rigid_penetration": candidate.rigid_penetration,
        "mean_contact_distance": candidate.mean_distance,
        "force_closure_residual": candidate.force_closure,
        "gravity_balance_residual": candidate.gravity_balance_residual,
        "worst_disturbance_residual": candidate.disturbance_residual,
        "contact_normal_coverage": candidate.normal_coverage,
        "table_clearance": candidate.table_clearance,
        "approach_table_clearance": candidate.approach_table_clearance,
        "approach_planned": candidate.approach_plan is not None,
        "approach_maximum_penetration": (
            candidate.approach_plan.maximum_penetration
            if candidate.approach_plan is not None
            else None
        ),
        "approach_minimum_object_clearance": (
            candidate.approach_plan.minimum_object_clearance
            if candidate.approach_plan is not None
            else None
        ),
        "grasp_maximum_penetration": (
            candidate.approach_plan.maximum_grasp_penetration
            if candidate.approach_plan is not None
            else None
        ),
        "grasp_maximum_rigid_penetration": (
            candidate.approach_plan.maximum_grasp_rigid_penetration
            if candidate.approach_plan is not None
            else None
        ),
    }


def payload(
    object_id: str | None,
    mesh_path: Path,
    cloud: Cloud,
    device: Device,
    candidates: list[Candidate],
) -> dict:
    candidate = candidates[0]
    translations, fractions = approach(candidate)
    if candidate.approach_plan is not None:
        grasp_translations = candidate.approach_plan.grasp_translations
        grasp_fractions = candidate.approach_plan.grasp_fractions
    else:
        grasp_translations = candidate.translation[None, :]
        grasp_fractions = candidate.surface.fractions[None, :]
    opposing = (
        4 in candidate.contacts and any(label < 4 for label in candidate.contacts)
        if device.name == "dex_hand"
        else 0 in candidate.contacts and 1 in candidate.contacts
    )
    success = candidate.valid and opposing
    preload_directions = (
        np.ones(len(device.actuators))
        if device.name == "dex_hand"
        else -np.ones(len(device.actuators))
    )
    preload_weights = np.ones(len(device.actuators))
    if device.name == "dex_hand":
        preload_weights[4] = 0.0
    return {
        "schema_version": GRASP_CONFIG_SCHEMA_VERSION,
        "search_strategy": GRASP_SEARCH_STRATEGY,
        "object_id": object_id,
        "end_effector_name": device.name,
        "mesh": str(mesh_path),
        "mesh_center": cloud.center.tolist(),
        "mesh_scale": cloud.scale,
        "object_table_height": float(cloud.points[:, 2].min()),
        "contact_points": candidate.contact_points.tolist(),
        "contact_normals": candidate.contact_normals.tolist(),
        "hand_actuator_fractions": candidate.surface.fractions.tolist(),
        "hand_actuator_values": candidate.surface.actuator_values.tolist(),
        "hand_preload_directions": preload_directions.tolist(),
        "hand_preload_weights": preload_weights.tolist(),
        "hand_translation": candidate.translation.tolist(),
        "hand_rotation_matrix": candidate.rotation.tolist(),
        "hand_mean_actuator_fraction": float(np.mean(candidate.surface.fractions)),
        "hand_maximum_penetration": candidate.penetration,
        "hand_maximum_noncontact_penetration": candidate.rigid_penetration,
        "hand_mean_contact_distance": candidate.mean_distance,
        "hand_contacting_fingers": list(candidate.contacts),
        "hand_force_closure_residual": candidate.force_closure,
        "hand_gravity_balance_residual": candidate.gravity_balance_residual,
        "hand_worst_disturbance_residual": candidate.disturbance_residual,
        "hand_contact_normal_coverage": candidate.normal_coverage,
        "hand_table_clearance": candidate.table_clearance,
        "approach_minimum_table_clearance": candidate.approach_table_clearance,
        "approach_maximum_object_penetration": (
            candidate.approach_plan.maximum_penetration
            if candidate.approach_plan is not None
            else None
        ),
        "approach_minimum_object_clearance": (
            candidate.approach_plan.minimum_object_clearance
            if candidate.approach_plan is not None
            else None
        ),
        "grasp_trajectory_maximum_penetration": (
            candidate.approach_plan.maximum_grasp_penetration
            if candidate.approach_plan is not None
            else None
        ),
        "grasp_trajectory_maximum_rigid_penetration": (
            candidate.approach_plan.maximum_grasp_rigid_penetration
            if candidate.approach_plan is not None
            else None
        ),
        "approach_direction": (
            candidate.approach_plan.direction.tolist()
            if candidate.approach_plan is not None
            else None
        ),
        **(
            approach_direction_metadata(candidate.approach_plan.direction)
            if candidate.approach_plan is not None
            else {}
        ),
        "hand_orientation_roll_index": candidate.roll_index,
        "hand_contact_distance_margin": max(0.0, 0.005 - candidate.mean_distance),
        "approach_hand_translations": translations.tolist(),
        "approach_hand_rotation_matrices": np.repeat(
            candidate.rotation[None, :, :], len(translations), axis=0
        ).tolist(),
        "approach_hand_actuator_fractions": fractions.tolist(),
        "grasp_hand_translations": grasp_translations.tolist(),
        "grasp_hand_rotation_matrices": np.repeat(
            candidate.rotation[None, :, :],
            len(grasp_translations),
            axis=0,
        ).tolist(),
        "grasp_hand_actuator_fractions": grasp_fractions.tolist(),
        "hand_fit_success": success,
        "search_debug_fallback_used": not candidate.valid,
        "search_candidate_count_saved": len(candidates),
        "search_candidates": [candidate_summary(item) for item in candidates],
    }
