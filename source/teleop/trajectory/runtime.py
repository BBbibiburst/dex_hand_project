"""Authoritative MuJoCo replay and validation for teleop trajectories."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mujoco
import numpy as np
from mujoco import viewer

from source.envs.manipulation import make_task
from source.envs.rl_env import RLEnvConfig, RobotGymEnv
from source.runtime.pacing import RealtimePacer
from source.teleop.trajectory.contracts import TeleopTrajectory
from source.viz.overlays import clear_markers, draw_label, draw_pose_frame


@dataclass(frozen=True)
class TrajectoryReplayResult:
    frames: int
    finite: bool
    task_success_ever: bool
    final_task_success: bool
    hold_success_fraction: float
    initial_object_z: float
    max_object_z: float
    final_object_z: float
    max_object_lift: float
    qpos_rmse: float | None
    object_position_rmse: float | None
    robot_table_contacts: int
    object_table_contacts: int
    robot_object_contacts: int
    robot_self_contacts: int
    max_penetration: float
    viewer_closed: bool

    @property
    def stable(self) -> bool:
        return self.finite and self.final_task_success and self.hold_success_fraction >= 0.8

    def to_dict(self) -> dict[str, Any]:
        return {
            "frames": self.frames,
            "finite": self.finite,
            "task_success_ever": self.task_success_ever,
            "final_task_success": self.final_task_success,
            "hold_success_fraction": self.hold_success_fraction,
            "stable": self.stable,
            "initial_object_z": self.initial_object_z,
            "max_object_z": self.max_object_z,
            "final_object_z": self.final_object_z,
            "max_object_lift": self.max_object_lift,
            "qpos_rmse": self.qpos_rmse,
            "object_position_rmse": self.object_position_rmse,
            "contacts": {
                "robot_table": self.robot_table_contacts,
                "object_table": self.object_table_contacts,
                "robot_object": self.robot_object_contacts,
                "robot_self": self.robot_self_contacts,
                "max_penetration_m": self.max_penetration,
            },
            "viewer_closed": self.viewer_closed,
        }


def make_trajectory_env(trajectory: TeleopTrajectory) -> RobotGymEnv:
    metadata = trajectory.metadata
    config_payload = dict(metadata.get("env_config") or {})
    if not config_payload:
        raise ValueError("Trajectory metadata does not contain env_config.")
    config = RLEnvConfig(**config_payload)
    task_name = str(metadata.get("task", "lift"))
    task_config = dict(metadata.get("task_config") or {})
    task = make_task(task_name, **task_config)
    env = RobotGymEnv(task=task, config=config, render_mode=None)
    env.reset(seed=int(metadata.get("seed", 0)))
    restore_trajectory_initial_state(env, trajectory)
    return env


def restore_trajectory_initial_state(env: RobotGymEnv, trajectory: TeleopTrajectory) -> None:
    if trajectory.initial_qpos.shape != env.data.qpos.shape:
        raise ValueError(
            f"Trajectory qpos shape {trajectory.initial_qpos.shape} does not match env {env.data.qpos.shape}."
        )
    if trajectory.initial_qvel.shape != env.data.qvel.shape:
        raise ValueError("Trajectory qvel shape does not match current environment.")
    if trajectory.initial_ctrl.shape != env.data.ctrl.shape:
        raise ValueError("Trajectory ctrl shape does not match current environment.")

    env.data.qpos[:] = trajectory.initial_qpos
    env.data.qvel[:] = trajectory.initial_qvel
    env.data.ctrl[:] = trajectory.initial_ctrl
    mujoco.mj_forward(env.model, env.data)
    # Synchronize controller history (IK branch, target velocity filters) to the
    # restored robot configuration, then restore the exact recorded controls.
    env.controller.reset(env.model, env.data, rng=env.np_random, options=None)
    env.data.ctrl[:] = trajectory.initial_ctrl
    mujoco.mj_forward(env.model, env.data)
    env.elapsed_steps = 0


def _object_position(observation: dict[str, Any]) -> np.ndarray:
    if "object_pos" not in observation:
        raise KeyError("Teleop grasp trajectories currently require the Lift observation key 'object_pos'.")
    return np.asarray(observation["object_pos"], dtype=np.float64)


def _contact_sets(env: RobotGymEnv) -> tuple[set[int], set[int], set[int]]:
    bindings = getattr(env.task, "bindings", None)
    object_ids: set[int] = set()
    robot_ids: set[int] = set()
    if bindings is not None:
        for binding in bindings.objects.values():
            object_ids.update(int(value) for value in binding.geom_ids)
        robot_ids.update(int(value) for value in bindings.robot_geom_ids)
    table_ids: set[int] = set()
    for geom_id in range(env.model.ngeom):
        name = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
        lower = name.lower()
        if "table" in lower or lower == "floor":
            table_ids.add(geom_id)
    return object_ids, robot_ids, table_ids


def _count_contacts(env: RobotGymEnv, sets) -> tuple[int, int, int, int, float]:
    object_ids, robot_ids, table_ids = sets
    robot_table = object_table = robot_object = robot_self = 0
    max_penetration = 0.0
    for index in range(env.data.ncon):
        contact = env.data.contact[index]
        first, second = int(contact.geom1), int(contact.geom2)
        pair = {first, second}
        max_penetration = max(max_penetration, max(0.0, -float(contact.dist)))
        if pair & table_ids:
            other = second if first in table_ids else first
            if other in object_ids:
                object_table += 1
            elif other in robot_ids:
                robot_table += 1
        if (first in robot_ids and second in object_ids) or (
            second in robot_ids and first in object_ids
        ):
            robot_object += 1
        if first in robot_ids and second in robot_ids:
            robot_self += 1
    return robot_table, object_table, robot_object, robot_self, max_penetration


def replay_teleop_trajectory(
    trajectory: TeleopTrajectory,
    *,
    render: bool = False,
    realtime: bool = True,
    hold_seconds: float = 1.0,
    compare_reference: bool = True,
) -> TrajectoryReplayResult:
    """Replay through the project's normal IK controller and MuJoCo physics."""
    if hold_seconds < 0.0:
        raise ValueError("hold_seconds must be non-negative.")
    env = make_trajectory_env(trajectory)
    handle = viewer.launch_passive(env.model, env.data) if render else None
    pacer = RealtimePacer()
    pacer.reset(float(env.data.time))
    contact_sets = _contact_sets(env)
    qpos_error_sq: list[float] = []
    object_error_sq: list[float] = []
    successes: list[bool] = []
    object_positions: list[np.ndarray] = []
    total_robot_table = total_object_table = total_robot_object = total_robot_self = 0
    maximum_penetration = 0.0
    finite = True
    viewer_closed = False
    frames = 0

    initial_object = _object_position(env._get_observation()).copy()
    try:
        for index, action in enumerate(trajectory.actions):
            if handle is not None and not handle.is_running():
                viewer_closed = True
                break
            observation, _, _, _, info = env.step(action)
            frames += 1
            object_pos = _object_position(observation)
            object_positions.append(object_pos.copy())
            success = bool(info.get("task_success", False))
            successes.append(success)
            finite = finite and all(
                np.all(np.isfinite(value))
                for value in (env.data.qpos, env.data.qvel, env.data.ctrl, object_pos)
            )
            if compare_reference and index < len(trajectory.observed_qpos):
                qpos_delta = env.data.qpos - trajectory.observed_qpos[index]
                qpos_error_sq.append(float(np.mean(qpos_delta * qpos_delta)))
                obj_delta = object_pos - trajectory.observed_object_position[index]
                object_error_sq.append(float(np.mean(obj_delta * obj_delta)))
            counts = _count_contacts(env, contact_sets)
            total_robot_table += counts[0]
            total_object_table += counts[1]
            total_robot_object += counts[2]
            total_robot_self += counts[3]
            maximum_penetration = max(maximum_penetration, counts[4])

            if handle is not None:
                clear_markers(handle)
                draw_pose_frame(
                    handle,
                    action[:3],
                    action[3:7],
                    axis_length=0.06,
                    label="TRAJ TARGET",
                )
                draw_label(
                    handle,
                    np.asarray([0.0, -0.32, 1.15], dtype=np.float32),
                    f"frame {index + 1}/{trajectory.horizon} | success {success}",
                )
                handle.sync()
            if realtime:
                pacer.sleep_until(float(env.data.time))

        hold_steps = int(round(hold_seconds / trajectory.control_dt))
        hold_success: list[bool] = []
        if frames and hold_steps:
            final_action = trajectory.actions[min(frames, trajectory.horizon) - 1]
            for _ in range(hold_steps):
                if handle is not None and not handle.is_running():
                    viewer_closed = True
                    break
                observation, _, _, _, info = env.step(final_action)
                object_pos = _object_position(observation)
                object_positions.append(object_pos.copy())
                success = bool(info.get("task_success", False))
                hold_success.append(success)
                finite = finite and all(
                    np.all(np.isfinite(value))
                    for value in (env.data.qpos, env.data.qvel, env.data.ctrl, object_pos)
                )
                counts = _count_contacts(env, contact_sets)
                total_robot_table += counts[0]
                total_object_table += counts[1]
                total_robot_object += counts[2]
                total_robot_self += counts[3]
                maximum_penetration = max(maximum_penetration, counts[4])
                if handle is not None:
                    handle.sync()
                if realtime:
                    pacer.sleep_until(float(env.data.time))
        else:
            hold_success = []

        positions = np.asarray(object_positions, dtype=np.float64)
        if len(positions):
            z_values = positions[:, 2]
            max_z = float(np.max(z_values))
            final_z = float(z_values[-1])
        else:
            max_z = final_z = float(initial_object[2])
        final_success = bool(hold_success[-1]) if hold_success else bool(successes[-1]) if successes else False
        hold_fraction = (
            float(np.mean(hold_success))
            if hold_success
            else (1.0 if final_success else 0.0)
        )
        return TrajectoryReplayResult(
            frames=frames,
            finite=bool(finite),
            task_success_ever=bool(any(successes) or any(hold_success)),
            final_task_success=final_success,
            hold_success_fraction=hold_fraction,
            initial_object_z=float(initial_object[2]),
            max_object_z=max_z,
            final_object_z=final_z,
            max_object_lift=max_z - float(initial_object[2]),
            qpos_rmse=(float(np.sqrt(np.mean(qpos_error_sq))) if qpos_error_sq else None),
            object_position_rmse=(
                float(np.sqrt(np.mean(object_error_sq))) if object_error_sq else None
            ),
            robot_table_contacts=total_robot_table,
            object_table_contacts=total_object_table,
            robot_object_contacts=total_robot_object,
            robot_self_contacts=total_robot_self,
            max_penetration=maximum_penetration,
            viewer_closed=viewer_closed,
        )
    finally:
        if handle is not None:
            handle.close()
        env.close()
