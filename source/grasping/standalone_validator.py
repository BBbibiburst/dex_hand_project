"""Standalone hand-and-object physics validation without a robot environment."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import mujoco
import numpy as np

from source.geometry import mat_to_quat
from source.grasping.constants import (
    DEFAULT_GRIP_PRELOAD,
    SUPPORTED_GRASP_CONFIG_SCHEMA_VERSIONS,
)
from source.robots.registry import get_hand


@dataclass(frozen=True)
class StandaloneValidationResult:
    stable: bool
    initial_displacement: float
    position_drift: float
    rotation_drift: float
    vertical_drop: float
    initial_contacts: int
    final_contacts: int
    simulated_seconds: float
    survival_fraction: float = 1.0
    lift_fraction: float = 1.0
    numerical_failure: bool = False
    failure_phase: str | None = None


def validate_grasp_config(
    path: str | Path,
    *,
    seconds: float = 3.0,
    settle_seconds: float = 0.8,
    grip_preload: float = DEFAULT_GRIP_PRELOAD,
) -> StandaloneValidationResult:
    """Load and dynamically validate one versioned grasp configuration."""
    config_path = Path(path)
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") not in SUPPORTED_GRASP_CONFIG_SCHEMA_VERSIONS:
        raise ValueError(f"Unsupported or missing schema_version in {config_path}.")
    if payload.get("hand_fit_success") is not True:
        raise ValueError(f"Grasp {config_path} did not pass mesh fitting.")

    end_effector_name = payload.get("end_effector_name", "dex_hand")
    actuator_names = tuple(get_hand(end_effector_name).position_actuator_names)
    model, data = build_standalone_model(
        object_mesh=payload["mesh"],
        mesh_center=np.asarray(payload["mesh_center"], dtype=np.float64),
        mesh_scale=float(payload["mesh_scale"]),
        hand_translation=np.asarray(payload["hand_translation"], dtype=np.float64),
        hand_rotation_matrix=np.asarray(
            payload["hand_rotation_matrix"],
            dtype=np.float64,
        ),
        object_table_height=payload.get("object_table_height"),
        end_effector_name=end_effector_name,
    )
    execute_configured_grasp_trajectory(
        model,
        data,
        payload,
        actuator_names=actuator_names,
    )
    set_hand_targets(
        model,
        data,
        np.asarray(payload["hand_actuator_values"], dtype=np.float64),
        grip_preload=grip_preload,
        preload_weights=np.asarray(payload["hand_preload_weights"], dtype=np.float64),
        preload_directions=np.asarray(
            payload.get("hand_preload_directions", np.ones(len(actuator_names))),
            dtype=np.float64,
        ),
        actuator_names=actuator_names,
    )
    return validate_standalone(
        model,
        data,
        seconds=seconds,
        settle_seconds=settle_seconds,
    )


def validate_grasp_payload_direct(
    payload: dict,
    *,
    seconds: float = 3.0,
    settle_seconds: float = 0.8,
    grip_preload: float = DEFAULT_GRIP_PRELOAD,
) -> StandaloneValidationResult:
    """Evaluate a final grasp state without replaying its approach trajectory.

    Simulator-in-the-loop optimizers mutate the final wrist pose independently
    of motion planning. This entry point deliberately evaluates that state
    directly; a new approach must be planned after evolution succeeds.
    """
    end_effector_name = payload.get("end_effector_name", "dex_hand")
    actuator_names = tuple(get_hand(end_effector_name).position_actuator_names)
    model, data = build_standalone_model(
        object_mesh=payload["mesh"],
        mesh_center=np.asarray(payload["mesh_center"], dtype=np.float64),
        mesh_scale=float(payload["mesh_scale"]),
        hand_translation=np.asarray(payload["hand_translation"], dtype=np.float64),
        hand_rotation_matrix=np.asarray(payload["hand_rotation_matrix"], dtype=np.float64),
        object_table_height=payload.get("object_table_height"),
        end_effector_name=end_effector_name,
    )
    set_hand_targets(
        model,
        data,
        np.asarray(payload["hand_actuator_values"], dtype=np.float64),
        grip_preload=grip_preload,
        preload_weights=np.asarray(payload["hand_preload_weights"], dtype=np.float64),
        preload_directions=np.asarray(
            payload.get("hand_preload_directions", np.ones(len(actuator_names))),
            dtype=np.float64,
        ),
        actuator_names=actuator_names,
    )
    return validate_standalone(
        model,
        data,
        seconds=seconds,
        settle_seconds=settle_seconds,
    )


def validate_grasp_payload_dynamic(
    payload: dict,
    *,
    seconds: float = 1.5,
    settle_seconds: float = 0.25,
    grip_preload: float | None = None,
    approach_distance: float = 0.025,
    close_seconds: float = 0.2,
    disturbance_force: float = 1.0,
) -> StandaloneValidationResult:
    """Evaluate a short executable grasp: local approach, close, preload, disturb."""
    previous_warning_handler = mujoco.get_mju_user_warning()
    mujoco.set_mju_user_warning(lambda _message: None)
    try:
        return _validate_grasp_payload_dynamic(
            payload,
            seconds=seconds,
            settle_seconds=settle_seconds,
            grip_preload=grip_preload,
            approach_distance=approach_distance,
            close_seconds=close_seconds,
            disturbance_force=disturbance_force,
        )
    finally:
        mujoco.set_mju_user_warning(previous_warning_handler)


def _validate_grasp_payload_dynamic(
    payload: dict,
    *,
    seconds: float,
    settle_seconds: float,
    grip_preload: float | None,
    approach_distance: float,
    close_seconds: float,
    disturbance_force: float,
) -> StandaloneValidationResult:
    if approach_distance < 0.0 or close_seconds <= 0.0 or disturbance_force < 0.0:
        raise ValueError("Dynamic grasp durations, distance, and force must be valid.")
    end_effector_name = payload.get("end_effector_name", "dex_hand")
    descriptor = get_hand(end_effector_name)
    actuator_names = tuple(descriptor.position_actuator_names)
    final_translation = np.asarray(payload["hand_translation"], dtype=np.float64)
    rotation = np.asarray(payload["hand_rotation_matrix"], dtype=np.float64)
    direction = np.asarray(payload.get("approach_direction", rotation[:, 2]), dtype=np.float64)
    direction /= max(float(np.linalg.norm(direction)), 1e-9)
    pregrasp_translation = final_translation + approach_distance * direction
    final_fractions = np.asarray(payload["hand_actuator_fractions"], dtype=np.float64)
    preload_directions = np.asarray(
        payload.get("hand_preload_directions", np.ones(len(actuator_names))),
        dtype=np.float64,
    )
    preload_weights = np.asarray(payload["hand_preload_weights"], dtype=np.float64)
    # Open only the closing actuators. Thumb opposition remains at its optimized angle.
    pregrasp_fractions = np.clip(
        final_fractions - 0.35 * preload_directions * preload_weights,
        0.0,
        1.0,
    )
    model, data = build_standalone_model(
        object_mesh=payload["mesh"],
        mesh_center=np.asarray(payload["mesh_center"], dtype=np.float64),
        mesh_scale=float(payload["mesh_scale"]),
        hand_translation=pregrasp_translation,
        hand_rotation_matrix=rotation,
        object_table_height=payload.get("object_table_height"),
        end_effector_name=end_effector_name,
    )
    phase_steps = max(2, int(np.ceil(close_seconds / model.opt.timestep)))
    for step in range(phase_steps):
        alpha = (step + 1) / phase_steps
        translation = (1.0 - alpha) * pregrasp_translation + alpha * final_translation
        fractions = (1.0 - alpha) * pregrasp_fractions + alpha * final_fractions
        set_hand_fraction_targets(model, data, fractions, actuator_names=actuator_names)
        set_object_pose_for_hand_pose(model, data, translation, rotation)
        mujoco.mj_step(model, data)
        set_object_pose_for_hand_pose(model, data, translation, rotation)
        if not _state_is_finite(data):
            return _numerical_failure_result(model, "approach_close", step + 1)
    candidate_preload = float(
        payload.get("evolution_grip_preload", DEFAULT_GRIP_PRELOAD)
        if grip_preload is None
        else grip_preload
    )
    set_hand_targets(
        model,
        data,
        np.asarray(payload["hand_actuator_values"], dtype=np.float64),
        grip_preload=candidate_preload,
        preload_weights=preload_weights,
        preload_directions=preload_directions,
        actuator_names=actuator_names,
    )
    return validate_standalone(
        model,
        data,
        seconds=seconds,
        settle_seconds=settle_seconds,
        disturbance_force=disturbance_force,
    )


def validate_grasp_trajectory_payload(
    payload: dict,
    *,
    steps_per_waypoint: int = 10,
) -> None:
    """Build the configured end effector and verify its free-space approach."""
    end_effector_name = payload.get("end_effector_name", "dex_hand")
    actuator_names = tuple(get_hand(end_effector_name).position_actuator_names)
    model, data = build_standalone_model(
        object_mesh=payload["mesh"],
        mesh_center=np.asarray(payload["mesh_center"], dtype=np.float64),
        mesh_scale=float(payload["mesh_scale"]),
        hand_translation=np.asarray(payload["hand_translation"], dtype=np.float64),
        hand_rotation_matrix=np.asarray(
            payload["hand_rotation_matrix"],
            dtype=np.float64,
        ),
        object_table_height=payload.get("object_table_height"),
        end_effector_name=end_effector_name,
    )
    execute_configured_grasp_trajectory(
        model,
        data,
        payload,
        actuator_names=actuator_names,
        steps_per_waypoint=steps_per_waypoint,
    )


def build_standalone_model(
    *,
    object_mesh: str | Path,
    mesh_center: np.ndarray,
    mesh_scale: float,
    hand_translation: np.ndarray,
    hand_rotation_matrix: np.ndarray,
    object_table_height: float | None = None,
    density: float = 500.0,
    end_effector_name: str = "dex_hand",
) -> tuple[mujoco.MjModel, mujoco.MjData]:
    """Build one registered end effector and one free mesh object."""
    descriptor = get_hand(end_effector_name)
    spec = mujoco.MjSpec.from_file(str(descriptor.xml_path.resolve()))
    mesh = spec.add_mesh()
    mesh.name = "validation_object_mesh"
    mesh.file = str(Path(object_mesh).resolve())
    mesh.scale = [float(mesh_scale)] * 3
    mesh.refpos = np.asarray(mesh_center, dtype=np.float64).tolist()

    # Search output expresses hand pose in object coordinates:
    # p_object = R_hand * p_hand + t_hand. Invert it because the standalone
    # hand root remains at the MJCF origin and the object is the free body.
    hand_rotation = np.asarray(hand_rotation_matrix, dtype=np.float64)
    hand_translation = np.asarray(hand_translation, dtype=np.float64)
    object_rotation = hand_rotation.T
    object_position = -(object_rotation @ hand_translation)

    body = spec.worldbody.add_body()
    body.name = "validation_object_body"
    body.pos = object_position.tolist()
    body.quat = mat_to_quat(object_rotation).tolist()
    joint = body.add_joint()
    joint.name = "validation_object_freejoint"
    joint.type = mujoco.mjtJoint.mjJNT_FREE
    joint.damping = np.zeros(3)
    joint.frictionloss = 0.0
    joint.armature = 0.0
    geom = body.add_geom()
    geom.name = "validation_object_collision"
    geom.type = mujoco.mjtGeom.mjGEOM_MESH
    geom.meshname = mesh.name
    geom.density = float(density)
    geom.friction = [1.0, 0.005, 0.0001]
    geom.condim = 4

    if object_table_height is not None:
        table_point_object = np.asarray(
            [0.0, 0.0, float(object_table_height)],
            dtype=np.float64,
        )
        table_point_hand = object_rotation @ (table_point_object - hand_translation)
        table_normal = object_rotation @ np.asarray([0.0, 0.0, 1.0])
        reference = (
            np.asarray([1.0, 0.0, 0.0])
            if abs(table_normal[0]) < 0.9
            else np.asarray([0.0, 1.0, 0.0])
        )
        table_x = np.cross(reference, table_normal)
        table_x /= np.linalg.norm(table_x)
        table_y = np.cross(table_normal, table_x)
        table_rotation = np.column_stack([table_x, table_y, table_normal])
        table = spec.worldbody.add_geom()
        table.name = "validation_table_visual"
        table.type = mujoco.mjtGeom.mjGEOM_PLANE
        table.pos = table_point_hand.tolist()
        table.quat = mat_to_quat(table_rotation).tolist()
        table.size = [0.25, 0.25, 0.001]
        table.rgba = [0.45, 0.45, 0.48, 0.45]
        table.contype = 0
        table.conaffinity = 0

    model = spec.compile()
    data = mujoco.MjData(model)
    return model, data


def set_hand_targets(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    actuator_values: np.ndarray,
    *,
    grip_preload: float = 0.0,
    preload_weights: np.ndarray | None = None,
    preload_directions: np.ndarray | None = None,
    actuator_names: tuple[str, ...] | None = None,
) -> None:
    default_names = (
        "act_push_0_j",
        "act_push_1_j",
        "act_push_2_j",
        "act_push_3_j",
        "thumb_rotate_act_push_j",
        "thumb_grasp_act_push_j",
    )
    names = default_names if actuator_names is None else actuator_names
    values = np.asarray(actuator_values, dtype=np.float64)
    if values.shape != (len(names),):
        raise ValueError("actuator_values size must match actuator_names.")
    if not 0.0 <= grip_preload <= 1.0:
        raise ValueError("grip_preload must be in [0, 1].")
    weights = (
        (
            np.asarray([1.0, 1.0, 1.0, 1.0, 0.0, 1.0])
            if len(names) == 6
            else np.ones(len(names), dtype=np.float64)
        )
        if preload_weights is None
        else np.asarray(preload_weights, dtype=np.float64)
    )
    directions = (
        np.ones(len(names), dtype=np.float64)
        if preload_directions is None
        else np.asarray(preload_directions, dtype=np.float64)
    )
    if weights.shape != (len(names),) or np.any((weights < 0.0) | (weights > 1.0)):
        raise ValueError("preload_weights must match actuators and lie in [0, 1].")
    if directions.shape != (len(names),) or np.any(~np.isin(directions, (-1.0, 1.0))):
        raise ValueError("preload_directions must contain only -1 or 1.")
    for name, value, weight, direction in zip(names, values, weights, directions, strict=True):
        actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        if actuator_id < 0:
            raise RuntimeError(f"Standalone hand actuator {name!r} is missing.")
        # Four fingers and thumb grasp receive extra closure after the
        # collision-free geometric pose has been initialized. Thumb rotation
        # keeps the optimized opposition angle.
        low, high = model.actuator_ctrlrange[actuator_id]
        endpoint = high if direction > 0.0 else low
        value = value + grip_preload * weight * (endpoint - value)
        data.ctrl[actuator_id] = value


def set_hand_fraction_targets(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    actuator_fractions: np.ndarray,
    *,
    actuator_names: tuple[str, ...] | None = None,
) -> None:
    """Set six hand controls from normalized actuator fractions."""
    default_names = (
        "act_push_0_j",
        "act_push_1_j",
        "act_push_2_j",
        "act_push_3_j",
        "thumb_rotate_act_push_j",
        "thumb_grasp_act_push_j",
    )
    names = default_names if actuator_names is None else actuator_names
    fractions = np.asarray(actuator_fractions, dtype=np.float64)
    if fractions.shape != (len(names),) or np.any((fractions < 0.0) | (fractions > 1.0)):
        raise ValueError("actuator_fractions must match actuators and lie in [0, 1].")
    for name, fraction in zip(names, fractions, strict=True):
        actuator_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_ACTUATOR,
            name,
        )
        low, high = model.actuator_ctrlrange[actuator_id]
        data.ctrl[actuator_id] = low + fraction * (high - low)


def set_object_pose_for_hand_pose(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    hand_translation: np.ndarray,
    hand_rotation_matrix: np.ndarray,
) -> None:
    """Pin the object so a fixed hand displays a searched relative hand pose."""
    hand_rotation = np.asarray(hand_rotation_matrix, dtype=np.float64)
    hand_translation = np.asarray(hand_translation, dtype=np.float64)
    object_rotation = hand_rotation.T
    object_position = -(object_rotation @ hand_translation)
    joint_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        "validation_object_freejoint",
    )
    qpos_address = int(model.jnt_qposadr[joint_id])
    dof_address = int(model.jnt_dofadr[joint_id])
    data.qpos[qpos_address : qpos_address + 3] = object_position
    data.qpos[qpos_address + 3 : qpos_address + 7] = mat_to_quat(object_rotation)
    data.qvel[dof_address : dof_address + 6] = 0.0
    mujoco.mj_forward(model, data)


def execute_configured_grasp_trajectory(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    payload: dict,
    *,
    actuator_names: tuple[str, ...],
    steps_per_waypoint: int = 10,
    step_callback=None,
) -> None:
    """Execute and collision-check the configured approach and closing path."""
    if steps_per_waypoint <= 0:
        raise ValueError("steps_per_waypoint must be positive.")
    actuator_count = len(actuator_names)
    approach_translations = np.asarray(
        payload.get("approach_hand_translations", []),
        dtype=np.float64,
    )
    approach_rotations = np.asarray(
        payload.get("approach_hand_rotation_matrices", []),
        dtype=np.float64,
    )
    approach_fractions = np.asarray(
        payload.get("approach_hand_actuator_fractions", []),
        dtype=np.float64,
    )
    grasp_translations = np.asarray(
        payload.get("grasp_hand_translations", [payload["hand_translation"]]),
        dtype=np.float64,
    )
    grasp_rotations = np.asarray(
        payload.get(
            "grasp_hand_rotation_matrices",
            [payload["hand_rotation_matrix"]],
        ),
        dtype=np.float64,
    )
    grasp_fractions = np.asarray(
        payload.get(
            "grasp_hand_actuator_fractions",
            [payload["hand_actuator_fractions"]],
        ),
        dtype=np.float64,
    )
    approach_count = len(approach_translations)
    grasp_count = len(grasp_translations)
    if (
        approach_count < 2
        or approach_rotations.shape != (approach_count, 3, 3)
        or approach_fractions.shape != (approach_count, actuator_count)
        or grasp_count < 1
        or grasp_rotations.shape != (grasp_count, 3, 3)
        or grasp_fractions.shape != (grasp_count, actuator_count)
    ):
        raise ValueError("Grasp config has malformed approach/grasp trajectories.")

    object_geom = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_GEOM,
        "validation_object_collision",
    )
    end_effector_name = payload.get("end_effector_name", "dex_hand")

    def is_allowed_grasp_geom(geom_id: int) -> bool:
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
        if end_effector_name == "pika_gripper":
            return "gripper_left_link" in name or "gripper_right_link" in name
        return any(f"skin_{finger}_" in name for finger in range(5))

    def execute_waypoints(
        translations: np.ndarray,
        rotations: np.ndarray,
        fractions: np.ndarray,
        *,
        phase: str,
        reject_contacts: bool,
        reject_rigid_contacts: bool,
    ) -> None:
        for waypoint_index, (translation, rotation, waypoint_fractions) in enumerate(
            zip(translations, rotations, fractions, strict=True)
        ):
            set_hand_fraction_targets(
                model,
                data,
                waypoint_fractions,
                actuator_names=actuator_names,
            )
            for _ in range(steps_per_waypoint):
                set_object_pose_for_hand_pose(model, data, translation, rotation)
                mujoco.mj_step(model, data)
                set_object_pose_for_hand_pose(model, data, translation, rotation)
                for contact_index in range(data.ncon):
                    geom1 = int(data.contact[contact_index].geom1)
                    geom2 = int(data.contact[contact_index].geom2)
                    if object_geom not in (geom1, geom2):
                        continue
                    hand_geom = geom2 if geom1 == object_geom else geom1
                    if reject_contacts or (
                        reject_rigid_contacts and not is_allowed_grasp_geom(hand_geom)
                    ):
                        hand_geom_name = (
                            mujoco.mj_id2name(
                                model,
                                mujoco.mjtObj.mjOBJ_GEOM,
                                hand_geom,
                            )
                            or f"geom#{hand_geom}"
                        )
                        raise ValueError(
                            f"{phase} trajectory collides with the object via "
                            f"{hand_geom_name} at waypoint "
                            f"{waypoint_index + 1}/{len(translations)}."
                        )
                if step_callback is not None:
                    step_callback(model, data, waypoint_index, len(translations))

    execute_waypoints(
        approach_translations,
        approach_rotations,
        approach_fractions,
        phase="Approach",
        reject_contacts=True,
        reject_rigid_contacts=True,
    )
    execute_waypoints(
        grasp_translations,
        grasp_rotations,
        grasp_fractions,
        phase="Grasp",
        reject_contacts=False,
        reject_rigid_contacts=True,
    )


def validate_standalone(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    seconds: float = 3.0,
    settle_seconds: float = 0.8,
    disturbance_force: float = 0.0,
    step_callback=None,
) -> StandaloneValidationResult:
    """Simulate a fixed hand holding a free object under gravity."""
    if seconds <= 0 or settle_seconds < 0:
        raise ValueError("seconds must be positive and settle_seconds non-negative.")
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "validation_object_body")
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "validation_object_freejoint")
    qpos_address = int(model.jnt_qposadr[joint_id])
    dof_address = int(model.jnt_dofadr[joint_id])
    fixed_object_pose = data.qpos[qpos_address : qpos_address + 7].copy()
    settle_steps = int(np.ceil(settle_seconds / model.opt.timestep))
    for settle_step in range(settle_steps):
        mujoco.mj_step(model, data)
        if not _state_is_finite(data):
            return _numerical_failure_result(model, "settle", settle_step + 1)
        data.qpos[qpos_address : qpos_address + 7] = fixed_object_pose
        data.qvel[dof_address : dof_address + 6] = 0.0
        mujoco.mj_forward(model, data)

    mujoco.mj_forward(model, data)
    object_geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "validation_object_collision")
    initial_contacts = sum(
        int(data.contact[index].geom1) == object_geom
        or int(data.contact[index].geom2) == object_geom
        for index in range(data.ncon)
    )
    initial_position = data.xpos[body_id].copy()
    initial_quaternion = data.xquat[body_id].copy()
    steps = int(np.ceil(seconds / model.opt.timestep))
    seating_step = min(steps - 1, int(np.ceil(1.0 / model.opt.timestep)))
    seated_position = initial_position.copy()
    survived_steps = 0
    failure_phase = None
    hold_steps = max(1, int(np.ceil(0.2 * steps)))
    directions = np.asarray(
        [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]],
        dtype=np.float64,
    )
    for step in range(steps):
        if disturbance_force > 0.0 and step >= hold_steps:
            disturbance_step = step - hold_steps
            disturbance_steps = max(steps - hold_steps, 1)
            direction_index = min(
                len(directions) - 1,
                disturbance_step * len(directions) // disturbance_steps,
            )
            data.xfrc_applied[body_id, :3] = disturbance_force * directions[direction_index]
        mujoco.mj_step(model, data)
        if not _state_is_finite(data):
            return _numerical_failure_result(model, "disturbance", step + 1)
        if step == seating_step:
            seated_position = data.xpos[body_id].copy()
        displacement = float(np.linalg.norm(data.xpos[body_id] - initial_position))
        if displacement > 0.08 or initial_position[2] - data.xpos[body_id, 2] > 0.05:
            failure_phase = "disturbance"
            break
        survived_steps = step + 1
        if step_callback is not None:
            step_callback(model, data, step, steps)

    final_position = data.xpos[body_id].copy()
    final_quaternion = data.xquat[body_id].copy()
    initial_displacement = float(np.linalg.norm(final_position - initial_position))
    position_drift = float(np.linalg.norm(final_position - seated_position))
    quaternion_dot = abs(float(np.dot(initial_quaternion, final_quaternion)))
    rotation_drift = float(2.0 * np.arccos(np.clip(quaternion_dot, 0.0, 1.0)))
    vertical_drop = float(initial_position[2] - final_position[2])
    final_contacts = sum(
        int(data.contact[index].geom1) == object_geom
        or int(data.contact[index].geom2) == object_geom
        for index in range(data.ncon)
    )
    survival_fraction = survived_steps / steps
    stable = (
        survival_fraction >= 1.0
        and position_drift <= 0.01
        and rotation_drift <= 0.35
        and vertical_drop <= 0.015
        and final_contacts >= 2
    )
    return StandaloneValidationResult(
        stable=stable,
        initial_displacement=initial_displacement,
        position_drift=position_drift,
        rotation_drift=rotation_drift,
        vertical_drop=vertical_drop,
        initial_contacts=int(initial_contacts),
        final_contacts=int(final_contacts),
        simulated_seconds=float(steps * model.opt.timestep),
        survival_fraction=float(survival_fraction),
        lift_fraction=float(min(1.0, survived_steps / hold_steps)),
        failure_phase=failure_phase,
    )


def _state_is_finite(data: mujoco.MjData) -> bool:
    return bool(
        np.all(np.isfinite(data.qpos))
        and np.all(np.isfinite(data.qvel))
        and np.all(np.isfinite(data.qacc))
        and np.max(np.abs(data.qvel), initial=0.0) < 1e4
        and np.max(np.abs(data.qacc), initial=0.0) < 1e8
    )


def _numerical_failure_result(
    model: mujoco.MjModel, phase: str, completed_steps: int
) -> StandaloneValidationResult:
    return StandaloneValidationResult(
        stable=False,
        initial_displacement=float("inf"),
        position_drift=float("inf"),
        rotation_drift=float("inf"),
        vertical_drop=float("inf"),
        initial_contacts=0,
        final_contacts=0,
        simulated_seconds=float(completed_steps * model.opt.timestep),
        survival_fraction=0.0,
        lift_fraction=0.0,
        numerical_failure=True,
        failure_phase=phase,
    )
