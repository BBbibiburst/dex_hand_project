"""Full-robot execution validation for published Lift grasp configs."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import json
import tempfile

import mujoco
import numpy as np

from source.envs.manipulation import make_manipulation_env
from source.geometry import mat_to_quat
from source.scripted.lift import LiftStrategy

MAXIMUM_PRECHECK_POSITION_ERROR = 0.06
MAXIMUM_PRECHECK_ORIENTATION_ERROR = 0.35


def _ik_waypoint_is_reachable(position_error: float, orientation_error: float) -> bool:
    return (
        position_error <= MAXIMUM_PRECHECK_POSITION_ERROR
        and orientation_error <= MAXIMUM_PRECHECK_ORIENTATION_ERROR
    )


def _precheck_strategy_waypoints(env, strategy: LiftStrategy, observation: dict) -> dict:
    """Check all configured wrist waypoints without advancing dynamics."""
    object_position = np.asarray(observation["object_pos"], dtype=np.float64)
    object_quaternion = np.asarray(observation["object_quat"], dtype=np.float64)
    strategy._ensure_grasp_template(env, object_position, object_quaternion)
    _, _, _, yaw, tool_roll = strategy._select_reachable_template_pose(
        env, object_position, object_quaternion
    )
    approach_positions, approach_quaternions = strategy._world_approach_waypoints(
        object_position, object_quaternion, yaw, tool_roll
    )
    grasp_positions, grasp_quaternions = strategy._world_grasp_waypoints(
        object_position, object_quaternion, yaw, tool_roll
    )
    arm = env.controller.arm_controller
    previous_max_joint_velocity = arm.max_joint_velocity
    previous_velocity_filter = arm.velocity_filter_alpha
    previous_target_q = None if arm._prev_target_q is None else arm._prev_target_q.copy()
    previous_filtered_velocity = (
        None if arm._filtered_velocity is None else arm._filtered_velocity.copy()
    )
    arm.max_joint_velocity = 100.0
    arm.velocity_filter_alpha = 1.0
    arm._prev_target_q = None
    arm._filtered_velocity = None
    previous_q = env.data.qpos[arm.qpos_addrs].copy()
    maximum_position_error = 0.0
    maximum_orientation_error = 0.0
    table_collision = False
    reason = None
    try:
        for waypoint_index, (position, quaternion) in enumerate(
            zip(
                np.concatenate([approach_positions, grasp_positions]),
                np.concatenate([approach_quaternions, grasp_quaternions]),
                strict=True,
            )
        ):
            target_q = arm._solve_ik(env.model, env.data, position, quaternion)
            env.data.qpos[arm.qpos_addrs] = target_q
            mujoco.mj_forward(env.model, env.data)
            actual_position = env.data.site_xpos[arm.site_id]
            actual_quaternion = mat_to_quat(env.data.site_xmat[arm.site_id])
            position_error = float(np.linalg.norm(actual_position - position))
            orientation_error = float(
                2.0
                * np.arccos(
                    np.clip(abs(float(np.dot(actual_quaternion, quaternion))), 0.0, 1.0)
                )
            )
            maximum_position_error = max(maximum_position_error, position_error)
            maximum_orientation_error = max(maximum_orientation_error, orientation_error)
            if not _ik_waypoint_is_reachable(position_error, orientation_error):
                reason = (
                    f"robot_ik_unreachable_waypoint_{waypoint_index}:"
                    f"position_error={position_error:.4f},"
                    f"orientation_error={orientation_error:.4f}"
                )
                break
            for progress in np.linspace(0.0, 1.0, 13)[1:]:
                env.data.qpos[arm.qpos_addrs] = previous_q + progress * (target_q - previous_q)
                mujoco.mj_forward(env.model, env.data)
                if LiftStrategy._robot_table_collision(env):
                    table_collision = True
                    reason = f"robot_table_collision_waypoint_{waypoint_index}"
                    break
            if reason is not None:
                break
            previous_q = target_q
    finally:
        arm.max_joint_velocity = previous_max_joint_velocity
        arm.velocity_filter_alpha = previous_velocity_filter
        arm._prev_target_q = previous_target_q
        arm._filtered_velocity = previous_filtered_velocity
    return {
        "precheck_passed": reason is None,
        "precheck_reason": reason,
        "maximum_ik_position_error": maximum_position_error,
        "maximum_ik_orientation_error": maximum_orientation_error,
        "table_collision": table_collision,
    }


def precheck_robot_lift_candidates(
    object_id: str,
    candidates: list[dict],
    *,
    seed: int = 0,
) -> list[dict]:
    """Precheck many candidates while compiling the full robot scene only once."""
    if not candidates:
        return []
    env = make_manipulation_env(
        "lift",
        task_config={
            "object_id": object_id,
            "reward_shaping": True,
            "terminate_on_success": False,
        },
        control_mode="ik",
        control_dt=1.0 / 20,
        episode_length=900,
        enable_tactile_sensors=True,
        render_mode=None,
    )
    results = []
    try:
        with tempfile.TemporaryDirectory(prefix="dex_robot_precheck_") as directory:
            path = Path(directory) / "candidate.json"
            for index, payload in enumerate(candidates):
                path.write_text(json.dumps(payload), encoding="utf-8")
                observation, _ = env.reset(seed=seed)
                strategy = LiftStrategy(reuse_grasp_config=True, grasp_config_path=path)
                strategy.reset()
                try:
                    result = _precheck_strategy_waypoints(env, strategy, observation)
                except Exception as exc:
                    result = {
                        "precheck_passed": False,
                        "precheck_reason": f"{type(exc).__name__}: {exc}",
                        "maximum_ik_position_error": float("inf"),
                        "maximum_ik_orientation_error": float("inf"),
                        "table_collision": False,
                    }
                result["candidate_index"] = index
                results.append(result)
    finally:
        env.close()
    return results


@dataclass(frozen=True)
class RobotLiftValidationResult:
    precheck_passed: bool
    precheck_reason: str | None
    maximum_ik_position_error: float
    maximum_ik_orientation_error: float
    robot_lift_verified: bool
    table_collision: bool
    steps: int
    final_phase: str
    aborted: bool
    error: str | None = None
    restart_count: int = 0
    last_grasp_hand_error: float = 0.0
    last_contact_fingers: tuple[int, ...] = ()
    last_wrist_position_error: float = 0.0
    last_wrist_orientation_error: float = 0.0

    def as_dict(self) -> dict:
        return asdict(self)


def validate_robot_lift(
    object_id: str,
    grasp_config_path: str | Path,
    *,
    seed: int = 0,
    max_steps: int = 900,
    fps: int = 20,
) -> RobotLiftValidationResult:
    """Execute approach, grasp, lift and verify in the complete robot scene."""
    if max_steps <= 0 or fps <= 0:
        raise ValueError("max_steps and fps must be positive.")
    env = make_manipulation_env(
        "lift",
        task_config={
            "object_id": object_id,
            "reward_shaping": True,
            "terminate_on_success": False,
        },
        control_mode="ik",
        control_dt=1.0 / fps,
        episode_length=max_steps,
        enable_tactile_sensors=True,
        render_mode=None,
    )
    strategy = LiftStrategy(
        reuse_grasp_config=True,
        grasp_config_path=Path(grasp_config_path),
        grasp_candidate_index=0,
    )
    steps = 0
    table_collision = False
    error = None
    precheck_passed = False
    precheck_reason = None
    maximum_position_error = 0.0
    maximum_orientation_error = 0.0
    try:
        observation, info = env.reset(seed=seed)
        strategy.reset()
        precheck = _precheck_strategy_waypoints(env, strategy, observation)
        precheck_passed = bool(precheck["precheck_passed"])
        precheck_reason = precheck["precheck_reason"]
        maximum_position_error = float(precheck["maximum_ik_position_error"])
        maximum_orientation_error = float(precheck["maximum_ik_orientation_error"])
        table_collision = bool(precheck["table_collision"])
        if not precheck_passed:
            return RobotLiftValidationResult(
                precheck_passed=False,
                precheck_reason=precheck_reason or "robot_trajectory_infeasible",
                maximum_ik_position_error=maximum_position_error,
                maximum_ik_orientation_error=maximum_orientation_error,
                robot_lift_verified=False,
                table_collision=table_collision,
                steps=0,
                final_phase="precheck",
                aborted=False,
                error=None,
            )
        observation, info = env.reset(seed=seed)
        strategy.reset()
        for step in range(max_steps):
            action, _ = strategy.tick(observation, info, step, env)
            observation, _, terminated, truncated, info = env.step(action)
            steps = step + 1
            if LiftStrategy._robot_table_collision(env):
                table_collision = True
                break
            if (
                strategy.state.strategy_verified_success
                or strategy.finished
                or strategy.aborted
                or terminated
                or truncated
            ):
                break
            if strategy.restart_count > 0:
                break
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    finally:
        env.close()
    return RobotLiftValidationResult(
        precheck_passed=precheck_passed,
        precheck_reason=precheck_reason,
        maximum_ik_position_error=maximum_position_error,
        maximum_ik_orientation_error=maximum_orientation_error,
        robot_lift_verified=bool(strategy.state.strategy_verified_success and not table_collision),
        table_collision=table_collision,
        steps=steps,
        final_phase=strategy.last_failed_phase or strategy.phase_name,
        aborted=strategy.aborted,
        error=error,
        restart_count=strategy.restart_count,
        last_grasp_hand_error=strategy.last_grasp_hand_error,
        last_contact_fingers=strategy.last_contact_fingers,
        last_wrist_position_error=strategy.last_wrist_position_error,
        last_wrist_orientation_error=strategy.last_wrist_orientation_error,
    )
