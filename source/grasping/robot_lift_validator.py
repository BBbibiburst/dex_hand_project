"""Full-robot execution validation for published Lift grasp configs."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import mujoco
import numpy as np

from source.envs.manipulation import make_manipulation_env
from source.geometry import mat_to_quat
from source.scripted.lift import LiftStrategy


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
        previous_q = env.data.qpos[arm.qpos_addrs].copy()
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
                * np.arccos(np.clip(abs(float(np.dot(actual_quaternion, quaternion))), 0.0, 1.0))
            )
            maximum_position_error = max(maximum_position_error, position_error)
            maximum_orientation_error = max(maximum_orientation_error, orientation_error)
            # A single call to this controller's iterative IK can be far from
            # its eventual closed-loop solution. Joint interpolation is only
            # meaningful when both endpoint solutions are already credible.
            if position_error <= 0.06 and orientation_error <= 0.35:
                for progress in np.linspace(0.0, 1.0, 13)[1:]:
                    env.data.qpos[arm.qpos_addrs] = previous_q + progress * (target_q - previous_q)
                    mujoco.mj_forward(env.model, env.data)
                    if LiftStrategy._robot_table_collision(env):
                        table_collision = True
                        precheck_reason = f"robot_table_collision_waypoint_{waypoint_index}"
                        break
            if table_collision:
                break
            previous_q = target_q
        else:
            precheck_passed = True
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
        final_phase=strategy.phase_name,
        aborted=strategy.aborted,
        error=error,
    )
