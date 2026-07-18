"""Standalone hand-and-object physics validation without a robot environment."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import mujoco
import numpy as np

from source.geometry import mat_to_quat
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


def validate_grasp_config(
    path: str | Path,
    *,
    seconds: float = 3.0,
    settle_seconds: float = 0.8,
    grip_preload: float = 0.25,
) -> StandaloneValidationResult:
    """Load and dynamically validate one versioned grasp configuration."""
    config_path = Path(path)
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
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


def validate_standalone(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    seconds: float = 3.0,
    settle_seconds: float = 0.8,
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
    for _ in range(settle_steps):
        mujoco.mj_step(model, data)
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
    for step in range(steps):
        mujoco.mj_step(model, data)
        if step == seating_step:
            seated_position = data.xpos[body_id].copy()
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
    stable = (
        position_drift <= 0.01
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
    )
