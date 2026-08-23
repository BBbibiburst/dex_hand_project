"""Replay an RL-refined low-level trajectory in authoritative C MuJoCo."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

from source.envs.manipulation import make_lift_env
from source.grasp_pipeline.trajectory import GraspTrajectory


@dataclass(frozen=True)
class ReplayResult:
    success: bool
    success_fraction: float
    object_lift: float
    frames: int


def replay_grasp_trajectory(
    trajectory_or_manifest: str | Path,
    *,
    render_mode: str | None = None,
    verify_tail: int = 20,
) -> ReplayResult:
    trajectory = GraspTrajectory.load(trajectory_or_manifest)
    control_dt = float(trajectory.metadata.get("control_dt", 0.05))
    source_seed = int(trajectory.metadata.get("source_seed", 0))
    env = make_lift_env(
        task_config={
            "object_id": trajectory.object_id,
            "reward_shaping": False,
            "terminate_on_success": False,
        },
        control_mode="position",
        enable_tactile_sensors=True,
        render_mode=render_mode,
        control_dt=control_dt,
        episode_length=len(trajectory.controls) + 10,
    )
    try:
        env.reset(seed=source_seed)
        if trajectory.initial_qpos.shape != env.data.qpos.shape:
            raise ValueError("RL trajectory qpos does not match the current robot model.")
        env.data.qpos[:] = trajectory.initial_qpos
        env.data.qvel[:] = trajectory.initial_qvel
        mujoco.mj_forward(env.model, env.data)
        initial_z = float(env.task._body_pos(env.model, env.data, "object")[2])
        successes: list[bool] = []
        for control in trajectory.controls:
            if control.shape != env.action_space.shape:
                raise ValueError(
                    f"RL control shape {control.shape} does not match {env.action_space.shape}."
                )
            _, _, terminated, truncated, info = env.step(control.astype(np.float32))
            successes.append(bool(info.get("task_success", False)))
            if terminated or truncated:
                break
        tail = successes[-min(verify_tail, len(successes)) :] if successes else []
        success_fraction = float(np.mean(tail)) if tail else 0.0
        final_z = float(env.task._body_pos(env.model, env.data, "object")[2])
        return ReplayResult(
            success=bool(tail and tail[-1] and success_fraction >= 0.8),
            success_fraction=success_fraction,
            object_lift=final_z - initial_z,
            frames=len(successes),
        )
    finally:
        env.close()
