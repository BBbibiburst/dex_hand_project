"""Cartesian execution and episode recording for synthesized grasps."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from source.envs.manipulation import make_lift_env
from source.geometry import mat_to_quat, normalize_quat
from source.ultradexgrasp.contracts import DemonstrationEpisode, GraspCandidate
from source.ultradexgrasp.hand_surrogate import OPEN_FRACTIONS

STAGE_CODES = {
    "settle": 0,
    "transit": 1,
    "pregrasp": 2,
    "approach": 3,
    "close": 4,
    "hold": 5,
    "lift": 6,
    "verify": 7,
}


@dataclass(frozen=True)
class ExecutionConfig:
    pregrasp_distance: float = 0.09
    lift_height: float = 0.065
    finger_preload: float = 0.15
    thumb_grasp_preload: float = 0.20
    transit_clearance: float = 0.16
    settle_steps: int = 12
    transit_steps: int = 70
    pregrasp_steps: int = 55
    approach_steps: int = 45
    close_steps: int = 45
    hold_steps: int = 40
    lift_steps: int = 90
    verify_steps: int = 30
    position_tolerance: float = 0.025
    orientation_tolerance: float = 0.22
    enable_tactile_sensors: bool = False

    @property
    def maximum_steps(self) -> int:
        return (
            self.settle_steps
            + self.transit_steps
            + self.pregrasp_steps
            + self.approach_steps
            + self.close_steps
            + self.hold_steps
            + self.lift_steps
            + self.verify_steps
        )

    def validate(self) -> None:
        if self.pregrasp_distance <= 0.0 or self.lift_height <= 0.0:
            raise ValueError("pregrasp_distance and lift_height must be positive.")
        if not 0.0 <= self.finger_preload <= 0.15:
            raise ValueError("finger_preload must lie in [0, 0.15].")
        if not 0.0 <= self.thumb_grasp_preload <= 0.20:
            raise ValueError("thumb_grasp_preload must lie in [0, 0.20].")
        step_names = (
            "settle_steps",
            "transit_steps",
            "pregrasp_steps",
            "approach_steps",
            "close_steps",
            "hold_steps",
            "lift_steps",
            "verify_steps",
        )
        if any(getattr(self, name) <= 0 for name in step_names):
            raise ValueError("Every execution stage must contain at least one step.")


@dataclass(frozen=True)
class ReachabilityResult:
    candidate: GraspCandidate
    score: float
    maximum_position_error: float
    maximum_orientation_error: float


def quaternion_wxyz_to_matrix(quaternion: np.ndarray) -> np.ndarray:
    quaternion = normalize_quat(np.asarray(quaternion, dtype=np.float64))
    return Rotation.from_quat(quaternion[[1, 2, 3, 0]]).as_matrix()


def hand_attach_rotation(env) -> np.ndarray:
    degrees = (
        env.arm_descriptor.hand_attach_rot_xyz_deg
        if env.config.hand_attach_rot_xyz_deg is None
        else env.config.hand_attach_rot_xyz_deg
    )
    return Rotation.from_euler("xyz", degrees, degrees=True).as_matrix()


def candidate_world_pose(
    candidate: GraspCandidate,
    object_position: np.ndarray,
    object_quaternion_wxyz: np.ndarray,
    attach_rotation: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return world hand position, hand rotation, and IK-site quaternion."""
    object_rotation = quaternion_wxyz_to_matrix(object_quaternion_wxyz)
    hand_position = np.asarray(object_position, dtype=np.float64) + (
        object_rotation @ candidate.hand_translation
    )
    hand_rotation = object_rotation @ candidate.hand_rotation_matrix
    ee_rotation = hand_rotation @ np.asarray(attach_rotation, dtype=np.float64).T
    return hand_position, hand_rotation, mat_to_quat(ee_rotation)


def rank_candidates_for_scene(
    env,
    candidates: tuple[GraspCandidate, ...],
    object_position: np.ndarray,
    object_quaternion_wxyz: np.ndarray,
    *,
    pregrasp_distance: float = 0.09,
) -> tuple[ReachabilityResult, ...]:
    """Rank candidates with the environment's actual RM75B IK model."""
    if pregrasp_distance <= 0.0:
        raise ValueError("pregrasp_distance must be positive.")
    arm = env.controller.arm_controller
    saved_qpos = env.data.qpos.copy()
    saved_qvel = env.data.qvel.copy()
    saved_ctrl = env.data.ctrl.copy()
    saved_previous_target = (
        None if arm._prev_target_q is None else arm._prev_target_q.copy()
    )
    saved_filtered_velocity = (
        None if arm._filtered_velocity is None else arm._filtered_velocity.copy()
    )
    saved_previous_ee = (
        None if arm._prev_ee_target is None else arm._prev_ee_target.copy()
    )
    saved_max_velocity = arm.max_joint_velocity
    saved_velocity_filter = arm.velocity_filter_alpha
    attach = hand_attach_rotation(env)
    results: list[ReachabilityResult] = []
    try:
        arm.max_joint_velocity = 100.0
        arm.velocity_filter_alpha = 1.0
        for candidate in candidates:
            env.data.qpos[:] = saved_qpos
            env.data.qvel[:] = saved_qvel
            env.data.ctrl[:] = saved_ctrl
            arm._prev_target_q = saved_qpos[arm.qpos_addrs].copy()
            arm._filtered_velocity = np.zeros_like(arm._prev_target_q)
            mujoco.mj_forward(env.model, env.data)
            grasp_position, hand_rotation, grasp_quaternion = candidate_world_pose(
                candidate,
                object_position,
                object_quaternion_wxyz,
                attach,
            )
            pregrasp_position = grasp_position - hand_rotation @ np.asarray(
                [0.0, pregrasp_distance, 0.0]
            )
            position_errors = []
            orientation_errors = []
            travel = 0.0
            previous_position = env.data.site_xpos[arm.site_id].copy()
            for position in (pregrasp_position, grasp_position):
                target_qpos = arm._solve_ik(
                    env.model,
                    env.data,
                    position,
                    grasp_quaternion,
                )
                env.data.qpos[arm.qpos_addrs] = target_qpos
                mujoco.mj_forward(env.model, env.data)
                actual_position = env.data.site_xpos[arm.site_id]
                actual_quaternion = mat_to_quat(env.data.site_xmat[arm.site_id])
                position_errors.append(float(np.linalg.norm(actual_position - position)))
                orientation_errors.append(
                    _orientation_error(actual_quaternion, grasp_quaternion)
                )
                travel += float(np.linalg.norm(actual_position - previous_position))
                previous_position = actual_position.copy()
            maximum_position_error = max(position_errors)
            maximum_orientation_error = max(orientation_errors)
            score = (
                20.0 * maximum_position_error
                + 0.6 * maximum_orientation_error
                + 0.08 * travel
            )
            results.append(
                ReachabilityResult(
                    candidate=candidate,
                    score=score,
                    maximum_position_error=maximum_position_error,
                    maximum_orientation_error=maximum_orientation_error,
                )
            )
    finally:
        arm.max_joint_velocity = saved_max_velocity
        arm.velocity_filter_alpha = saved_velocity_filter
        arm._prev_target_q = saved_previous_target
        arm._filtered_velocity = saved_filtered_velocity
        arm._prev_ee_target = saved_previous_ee
        env.data.qpos[:] = saved_qpos
        env.data.qvel[:] = saved_qvel
        env.data.ctrl[:] = saved_ctrl
        mujoco.mj_forward(env.model, env.data)
    return tuple(sorted(results, key=lambda result: result.score))


def actuator_targets_from_fractions(env, fractions: np.ndarray) -> np.ndarray:
    fractions = np.asarray(fractions, dtype=np.float64)
    if fractions.shape != (6,) or np.any((fractions < 0.0) | (fractions > 1.0)):
        raise ValueError("Dex Hand actuator fractions must have shape (6,) in [0, 1].")
    controller = env.controller.hand_controller
    return (
        controller.ctrl_low + fractions * (controller.ctrl_high - controller.ctrl_low)
    ).astype(np.float32)


def _slerp_wxyz(first: np.ndarray, second: np.ndarray, fraction: float) -> np.ndarray:
    first = normalize_quat(np.asarray(first, dtype=np.float64))
    second = normalize_quat(np.asarray(second, dtype=np.float64))
    dot = float(np.clip(np.dot(first, second), -1.0, 1.0))
    if dot < 0.0:
        second = -second
        dot = -dot
    if dot > 0.9995:
        return normalize_quat(first + fraction * (second - first))
    angle = np.arccos(dot)
    return normalize_quat(
        np.sin((1.0 - fraction) * angle) / np.sin(angle) * first
        + np.sin(fraction * angle) / np.sin(angle) * second
    )


def _orientation_error(first: np.ndarray, second: np.ndarray) -> float:
    dot = abs(float(np.dot(normalize_quat(first), normalize_quat(second))))
    return float(2.0 * np.arccos(np.clip(dot, 0.0, 1.0)))


class _Recorder:
    def __init__(self) -> None:
        self.values: dict[str, list[Any]] = {
            "qpos": [],
            "qvel": [],
            "ctrl": [],
            "action": [],
            "object_position": [],
            "object_quaternion_wxyz": [],
            "stage": [],
            "reward": [],
            "task_success": [],
            "robot_object_contact_count": [],
            "robot_object_normal_force": [],
            "robot_object_digit_contact_count": [],
            "robot_object_digit_normal_force": [],
        }

    def append(
        self,
        env,
        observation: dict[str, Any],
        action: np.ndarray,
        stage: str,
        reward: float,
        success: bool,
    ) -> None:
        self.values["qpos"].append(env.data.qpos.astype(np.float32).copy())
        self.values["qvel"].append(env.data.qvel.astype(np.float32).copy())
        self.values["ctrl"].append(env.data.ctrl.astype(np.float32).copy())
        self.values["action"].append(np.asarray(action, dtype=np.float32).copy())
        self.values["object_position"].append(
            np.asarray(observation["object_pos"], dtype=np.float32).copy()
        )
        self.values["object_quaternion_wxyz"].append(
            np.asarray(observation["object_quat"], dtype=np.float32).copy()
        )
        self.values["stage"].append(STAGE_CODES[stage])
        self.values["reward"].append(float(reward))
        self.values["task_success"].append(bool(success))
        contact_count, normal_force, digit_counts, digit_forces = _robot_object_contact_summary(env)
        self.values["robot_object_contact_count"].append(contact_count)
        self.values["robot_object_normal_force"].append(normal_force)
        self.values["robot_object_digit_contact_count"].append(digit_counts)
        self.values["robot_object_digit_normal_force"].append(digit_forces)

    def arrays(self) -> dict[str, np.ndarray]:
        arrays = {name: np.asarray(values) for name, values in self.values.items()}
        arrays["stage"] = arrays["stage"].astype(np.int16)
        arrays["reward"] = arrays["reward"].astype(np.float32)
        arrays["task_success"] = arrays["task_success"].astype(np.bool_)
        arrays["robot_object_contact_count"] = arrays[
            "robot_object_contact_count"
        ].astype(np.int16)
        arrays["robot_object_normal_force"] = arrays["robot_object_normal_force"].astype(
            np.float32
        )
        arrays["robot_object_digit_contact_count"] = arrays[
            "robot_object_digit_contact_count"
        ].astype(np.int16)
        arrays["robot_object_digit_normal_force"] = arrays[
            "robot_object_digit_normal_force"
        ].astype(np.float32)
        return arrays


def _contact_digit(env, geom_id: int) -> int:
    geom_name = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
    body_id = int(env.model.geom_bodyid[geom_id])
    body_name = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_BODY, body_id) or ""
    text = f"{geom_name} {body_name}".lower()
    for digit in range(5):
        if f"skin_{digit}_" in text:
            return digit
    aliases = {
        0: ("little", "pinky", "finger_0", "finger0"),
        1: ("ring", "finger_1", "finger1"),
        2: ("middle", "finger_2", "finger2"),
        3: ("index", "finger_3", "finger3"),
        4: ("thumb",),
    }
    for digit, names in aliases.items():
        if any(name in text for name in names):
            return digit
    return -1


def _robot_object_contact_summary(env) -> tuple[int, float, np.ndarray, np.ndarray]:
    bindings = getattr(env.task, "bindings", None)
    if bindings is None or "object" not in bindings.objects:
        return 0, 0.0, np.zeros(5, dtype=np.int16), np.zeros(5, dtype=np.float64)
    object_geoms = {int(value) for value in bindings.objects["object"].geom_ids}
    robot_geoms = {int(value) for value in bindings.robot_geom_ids}
    contact_force = np.zeros(6, dtype=np.float64)
    digit_counts = np.zeros(5, dtype=np.int16)
    digit_forces = np.zeros(5, dtype=np.float64)
    count = 0
    normal_force = 0.0
    for index in range(env.data.ncon):
        contact = env.data.contact[index]
        pair = {int(contact.geom1), int(contact.geom2)}
        object_pair = pair.intersection(object_geoms)
        robot_pair = pair.intersection(robot_geoms)
        if not object_pair or not robot_pair:
            continue
        mujoco.mj_contactForce(env.model, env.data, index, contact_force)
        force = abs(float(contact_force[0]))
        count += 1
        normal_force += force
        digit = _contact_digit(env, next(iter(robot_pair)))
        if digit >= 0:
            digit_counts[digit] += 1
            digit_forces[digit] += force
    return count, normal_force, digit_counts, digit_forces


def _step(env, recorder: _Recorder, action: np.ndarray, stage: str):
    observation, reward, terminated, truncated, info = env.step(action)
    success = bool(info.get("task_success", False))
    recorder.append(env, observation, action, stage, reward, success)
    return observation, success, bool(terminated or truncated)


def _run_pose_segment(
    env,
    recorder: _Recorder,
    *,
    stage: str,
    target_position: np.ndarray,
    target_quaternion: np.ndarray,
    start_hand: np.ndarray,
    target_hand: np.ndarray,
    steps: int,
) -> tuple[dict[str, Any], bool, bool]:
    arm = env.controller.arm_controller
    mujoco.mj_forward(env.model, env.data)
    start_position = env.data.site_xpos[arm.site_id].astype(np.float64).copy()
    start_quaternion = mat_to_quat(env.data.site_xmat[arm.site_id])
    observation: dict[str, Any] = {}
    succeeded = False
    ended = False
    for index in range(steps):
        fraction = (index + 1) / steps
        position = (1.0 - fraction) * start_position + fraction * target_position
        quaternion = _slerp_wxyz(start_quaternion, target_quaternion, fraction)
        hand = (1.0 - fraction) * start_hand + fraction * target_hand
        action = np.concatenate([position, quaternion, hand]).astype(np.float32)
        observation, step_success, ended = _step(env, recorder, action, stage)
        succeeded = succeeded or step_success
        if ended:
            break
    return observation, succeeded, ended


def _target_error(env, position: np.ndarray, quaternion: np.ndarray) -> tuple[float, float]:
    arm = env.controller.arm_controller
    mujoco.mj_forward(env.model, env.data)
    actual_position = env.data.site_xpos[arm.site_id]
    actual_quaternion = mat_to_quat(env.data.site_xmat[arm.site_id])
    return (
        float(np.linalg.norm(actual_position - position)),
        _orientation_error(actual_quaternion, quaternion),
    )


def execute_grasp(
    candidate: GraspCandidate,
    *,
    seed: int = 0,
    config: ExecutionConfig | None = None,
    render_mode: str | None = None,
    environment=None,
) -> DemonstrationEpisode:
    """Execute one candidate and return a self-contained demonstration episode."""
    config = config or ExecutionConfig()
    config.validate()
    owns_environment = environment is None
    env = environment or make_lift_env(
        task_config={"object_id": candidate.object_id},
        control_mode="ik",
        enable_tactile_sensors=config.enable_tactile_sensors,
        episode_length=config.maximum_steps + 20,
        render_mode=render_mode,
    )
    recorder = _Recorder()
    terminal_stage = "settle"
    failure_reason: str | None = None
    metadata: dict[str, Any] = {
        "execution_config": asdict(config),
        "stage_codes": STAGE_CODES,
    }

    try:
        observation, _ = env.reset(seed=seed)
        arm = env.controller.arm_controller
        current_position = env.data.site_xpos[arm.site_id].astype(np.float64).copy()
        current_quaternion = mat_to_quat(env.data.site_xmat[arm.site_id])
        open_hand = actuator_targets_from_fractions(env, OPEN_FRACTIONS)
        closed_hand = actuator_targets_from_fractions(env, candidate.actuator_fractions)
        grip_fractions = np.clip(
            candidate.actuator_fractions
            + np.asarray(
                [
                    config.finger_preload,
                    config.finger_preload,
                    config.finger_preload,
                    config.finger_preload,
                    0.0,
                    config.thumb_grasp_preload,
                ]
            ),
            0.0,
            1.0,
        )
        grip_hand = actuator_targets_from_fractions(env, grip_fractions)

        observation, _, ended = _run_pose_segment(
            env,
            recorder,
            stage="settle",
            target_position=current_position,
            target_quaternion=current_quaternion,
            start_hand=env.controller.hand_controller.current_action(env.model, env.data),
            target_hand=open_hand,
            steps=config.settle_steps,
        )
        if ended:
            failure_reason = "environment ended during settle"
        else:
            object_position = np.asarray(observation["object_pos"], dtype=np.float64)
            object_quaternion = np.asarray(observation["object_quat"], dtype=np.float64)
            grasp_position, hand_rotation, grasp_quaternion = candidate_world_pose(
                candidate,
                object_position,
                object_quaternion,
                hand_attach_rotation(env),
            )
            pregrasp_position = grasp_position - hand_rotation @ np.asarray(
                [0.0, config.pregrasp_distance, 0.0]
            )
            transit_z = max(
                float(pregrasp_position[2] + config.transit_clearance),
                float(env.task.table_top_z + config.transit_clearance + 0.10),
            )
            transit_position = np.asarray(
                [pregrasp_position[0], pregrasp_position[1], transit_z],
                dtype=np.float64,
            )
            metadata.update(
                {
                    "object_initial_position": object_position.tolist(),
                    "object_initial_quaternion_wxyz": object_quaternion.tolist(),
                    "grasp_ee_position": grasp_position.tolist(),
                    "grasp_ee_quaternion_wxyz": grasp_quaternion.tolist(),
                    "pregrasp_ee_position": pregrasp_position.tolist(),
                }
            )

            stages = (
                ("transit", transit_position, current_quaternion, config.transit_steps),
                ("pregrasp", pregrasp_position, grasp_quaternion, config.pregrasp_steps),
            )
            for stage, position, quaternion, steps in stages:
                terminal_stage = stage
                observation, _, ended = _run_pose_segment(
                    env,
                    recorder,
                    stage=stage,
                    target_position=position,
                    target_quaternion=quaternion,
                    start_hand=open_hand,
                    target_hand=open_hand,
                    steps=steps,
                )
                position_error, orientation_error = _target_error(env, position, quaternion)
                metadata[f"{stage}_position_error"] = position_error
                metadata[f"{stage}_orientation_error"] = orientation_error
                if ended:
                    failure_reason = f"environment ended during {stage}"
                    break
                if stage != "transit" and (
                    position_error > config.position_tolerance
                    or orientation_error > config.orientation_tolerance
                ):
                    failure_reason = (
                        f"{stage} IK residual is too large: "
                        f"position={position_error:.4f}, orientation={orientation_error:.4f}"
                    )
                    break

            if failure_reason is None:
                terminal_stage = "approach"
                observation, _, ended = _run_pose_segment(
                    env,
                    recorder,
                    stage="approach",
                    target_position=grasp_position,
                    target_quaternion=grasp_quaternion,
                    start_hand=open_hand,
                    target_hand=closed_hand,
                    steps=config.approach_steps,
                )
                position_error, orientation_error = _target_error(
                    env,
                    grasp_position,
                    grasp_quaternion,
                )
                metadata["approach_position_error"] = position_error
                metadata["approach_orientation_error"] = orientation_error
                if ended:
                    failure_reason = "environment ended during approach"
                elif (
                    position_error > config.position_tolerance
                    or orientation_error > config.orientation_tolerance
                ):
                    failure_reason = (
                        "approach IK residual is too large: "
                        f"position={position_error:.4f}, orientation={orientation_error:.4f}"
                    )

            if failure_reason is None:
                terminal_stage = "close"
                observation, _, ended = _run_pose_segment(
                    env,
                    recorder,
                    stage="close",
                    target_position=grasp_position,
                    target_quaternion=grasp_quaternion,
                    start_hand=closed_hand,
                    target_hand=grip_hand,
                    steps=config.close_steps,
                )
                if ended:
                    failure_reason = "environment ended during close"

            if failure_reason is None:
                terminal_stage = "hold"
                observation, _, ended = _run_pose_segment(
                    env,
                    recorder,
                    stage="hold",
                    target_position=grasp_position,
                    target_quaternion=grasp_quaternion,
                    start_hand=grip_hand,
                    target_hand=grip_hand,
                    steps=config.hold_steps,
                )
                if ended:
                    failure_reason = "environment ended during hold"

            if failure_reason is None:
                terminal_stage = "lift"
                lift_position = grasp_position + np.asarray([0.0, 0.0, config.lift_height])
                observation, _, ended = _run_pose_segment(
                    env,
                    recorder,
                    stage="lift",
                    target_position=lift_position,
                    target_quaternion=grasp_quaternion,
                    start_hand=grip_hand,
                    target_hand=grip_hand,
                    steps=config.lift_steps,
                )
                if ended:
                    failure_reason = "environment ended during lift"

            if failure_reason is None:
                terminal_stage = "verify"
                observation, _, _ = _run_pose_segment(
                    env,
                    recorder,
                    stage="verify",
                    target_position=lift_position,
                    target_quaternion=grasp_quaternion,
                    start_hand=grip_hand,
                    target_hand=grip_hand,
                    steps=config.verify_steps,
                )

        arrays = recorder.arrays()
        verify_success = arrays["task_success"][arrays["stage"] == STAGE_CODES["verify"]]
        success_fraction = float(verify_success.mean()) if len(verify_success) else 0.0
        success = bool(
            len(verify_success)
            and verify_success[-1]
            and success_fraction >= 0.8
        )
        metadata["verify_success_fraction"] = success_fraction
        if not success and failure_reason is None:
            failure_reason = "object did not satisfy the lift success criterion"
        if len(arrays["object_position"]):
            metadata["object_final_position"] = arrays["object_position"][-1].tolist()
            metadata["object_lift"] = float(
                arrays["object_position"][-1, 2] - arrays["object_position"][0, 2]
            )
        metadata["action_layout"] = list(env.controller.ik_action_layout())
        return DemonstrationEpisode(
            object_id=candidate.object_id,
            seed=seed,
            candidate=candidate,
            arrays=arrays,
            success=success,
            terminal_stage=terminal_stage,
            failure_reason=failure_reason,
            metadata=metadata,
        )
    finally:
        if owns_environment:
            env.close()
