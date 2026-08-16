"""Candidate archive, ranking, and atomic publication helpers."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def _payload_after_robot_lift_attempts(
    preferred_payload: dict,
    attempted_payload: dict,
    *,
    robot_lift_verified: bool,
) -> dict:
    """Publish a successful Lift candidate, otherwise restore the trajectory-first choice."""
    return dict(attempted_payload if robot_lift_verified else preferred_payload)


def _write_payload_atomic(path: Path, payload: dict) -> None:
    """Publish one grasp payload without exposing a partially-written JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def _robot_candidate_precheck_key(
    payload: dict, individual_fitness: float, precheck: dict
) -> tuple:
    """Prioritize executable, collision-free and well-cleared robot candidates."""
    return (
        0 if precheck["precheck_passed"] else 1,
        1 if precheck["table_collision"] else 0,
        float(precheck["maximum_ik_position_error"]),
        float(precheck["maximum_ik_orientation_error"]),
        -float(payload.get("trajectory_minimum_table_clearance", 0.0)),
        -float(individual_fitness),
    )


def _candidate_is_diverse(
    candidate: dict,
    archive: list[dict],
    *,
    translation_threshold: float = 0.025,
    rotation_threshold: float = np.deg2rad(15.0),
    joint_threshold: float = 0.08,
) -> bool:
    """Reject near-identical wrist poses and hand shapes across search attempts."""
    translation = np.asarray(candidate["hand_translation"], dtype=np.float64)
    rotation = np.asarray(candidate["hand_rotation_matrix"], dtype=np.float64)
    joints = np.asarray(candidate["hand_actuator_fractions"], dtype=np.float64)
    for existing in archive:
        existing_translation = np.asarray(existing["hand_translation"], dtype=np.float64)
        existing_rotation = np.asarray(existing["hand_rotation_matrix"], dtype=np.float64)
        existing_joints = np.asarray(existing["hand_actuator_fractions"], dtype=np.float64)
        translation_distance = float(np.linalg.norm(translation - existing_translation))
        relative_trace = float(np.trace(rotation @ existing_rotation.T))
        rotation_distance = float(np.arccos(np.clip((relative_trace - 1.0) / 2.0, -1.0, 1.0)))
        joint_distance = float(np.sqrt(np.mean(np.square(joints - existing_joints))))
        if (
            translation_distance < translation_threshold
            and rotation_distance < rotation_threshold
            and joint_distance < joint_threshold
        ):
            return False
    return True


def _approach_bins(candidate: dict) -> set[str]:
    value = candidate.get("approach_bin")
    return {str(value)} if value else set()


def _append_diverse_candidates(
    archive: list[dict], candidates: list[dict], *, maximum: int
) -> None:
    for candidate in candidates:
        if len(archive) >= maximum:
            break
        if _candidate_is_diverse(candidate, archive):
            archive.append(dict(candidate))


def _incomplete_attempt_key(row: dict) -> tuple:
    """Rank failed attempts by progress toward an executable robot Lift."""
    lift = row.get("robot_lift") or {}
    phase_rank = {
        "precheck": 0,
        "approach": 1,
        "grasp": 2,
        "lift": 3,
        "verify": 4,
        "done": 5,
    }.get(str(lift.get("final_phase") or ""), -1)
    return (
        0 if lift.get("robot_lift_verified") else 1,
        -phase_rank,
        1 if lift.get("table_collision") else 0,
        float(row.get("vertical_drop", float("inf"))),
        float(row.get("position_drift", float("inf"))),
        float(row.get("rotation_drift", float("inf"))),
        -int(row.get("final_contacts", 0)),
    )
