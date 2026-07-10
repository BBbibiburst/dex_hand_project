# -*- coding: utf-8 -*-
"""Gymnasium environment for descriptor-driven robot assemblies.

The environment owns timing, rendering, and runtime diagnostics. Control
logic is delegated to a descriptor-built composite controller; task logic is
delegated to ``RobotTask`` implementations; tactile sensing is delegated to
whatever ``TactileSensorBase`` the end effector's descriptor produces.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import time
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
from gymnasium import spaces
import mujoco
from mujoco import viewer
import numpy as np

from source.control.composite import build_robot_controller
from source.viz.overlays import clear_markers, draw_stats_label
from source.robots.config import (
    apply_config_overrides,
    dataclass_from_robot_config,
    load_robot_config,
)
from source.robots.builder import build_robot_spec
from source.robots.scene import add_basic_scene
from source.sensors.base import NullTactileSensor, TactileSensorBase
from source.envs.core.tasks import NoopTask, RobotTask
from source.robots.registry import get_arm, get_base, get_hand


Array = np.ndarray
Observation = Dict[str, Any]


@dataclass(frozen=True)
class RLEnvConfig:
    arm_name: str = "rm75b"
    hand_name: str = "dex_hand"
    base_name: str = "rethink_minimal_mount"
    hand_attach_rot_xyz_deg: Optional[Tuple[float, float, float]] = None
    attach_point_name: Optional[str] = None
    base_mount_site_name: Optional[str] = None
    control_dt: float = 0.05  # 20 Hz controller updates.
    episode_length: int = 500
    add_default_scene: bool = True
    enable_task_objects: bool = False
    hand_prefix: Optional[str] = None
    stats_interval: float = 0.5
    control_mode: str = "position"  # "position" or "ik".
    ee_site_name: Optional[str] = None
    include_hand_action: bool = True
    normalized_position: bool = False
    enable_tactile_sensors: bool = True
    tactile_backend: str = "simple_box"
    tactile_options: Optional[Dict[str, Any]] = None


def load_env_config(
    robot_config_path: Optional[str] = None,
    **overrides: Any,
) -> RLEnvConfig:
    """Load ``RLEnvConfig`` from the global robot config JSON."""
    config_data = apply_config_overrides(load_robot_config(robot_config_path), overrides)
    return dataclass_from_robot_config(RLEnvConfig, config_data)


class RobotGymEnv(gym.Env):
    """Gymnasium environment for a registered arm + end-effector + base.

    Delegates control to a descriptor-built composite controller and task logic to a
    ``RobotTask`` instance. Supports both direct position control and
    end-effector IK modes.

    Tactile sensing is resolved from the end-effector descriptor's
    ``tactile_sensor_factory`` when ``config.enable_tactile_sensors`` is
    True and the descriptor provides one; otherwise ``NullTactileSensor`` is
    used. Pass ``tactile_sensor`` explicitly to override either behaviour.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(
        self,
        *,
        task: Optional[RobotTask] = None,
        tactile_sensor: Optional[TactileSensorBase] = None,
        config: Optional[RLEnvConfig] = None,
        render_mode: Optional[str] = None,
        robot_config_path: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.config = config or load_env_config(robot_config_path)
        self.arm_descriptor = get_arm(self.config.arm_name)
        self.hand_descriptor = get_hand(self.config.hand_name)
        self.base_descriptor = get_base(self.config.base_name)
        self.hand_prefix = (
            self.hand_descriptor.default_prefix
            if self.config.hand_prefix is None
            else self.config.hand_prefix
        )
        self.task = task or NoopTask()
        self.controller = build_robot_controller(
            arm_descriptor=self.arm_descriptor,
            hand_descriptor=self.hand_descriptor,
            hand_prefix=self.hand_prefix,
            control_mode=self.config.control_mode,
            ee_site_name=self.config.ee_site_name or self.arm_descriptor.ee_site_name,
            include_hand_action=self.config.include_hand_action,
            normalized_position=self.config.normalized_position,
        )

        if tactile_sensor is not None:
            self.tactile_sensor: TactileSensorBase = tactile_sensor
        elif self.config.enable_tactile_sensors and self.hand_descriptor.tactile_sensor_factory:
            tactile_options = dict(self.config.tactile_options or {})
            if self.config.hand_name == "dex_hand":
                from source.sensors.tactile.dex_hand import create_dex_hand_tactile_sensor

                self.tactile_sensor = create_dex_hand_tactile_sensor(
                    self.config.tactile_backend,
                    **tactile_options,
                )
            else:
                if tactile_options:
                    raise ValueError(
                        "tactile_options are currently supported only for dex_hand."
                    )
                self.tactile_sensor = self.hand_descriptor.tactile_sensor_factory()
        else:
            self.tactile_sensor = NullTactileSensor()

        self.render_mode = render_mode
        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            raise ValueError(
                f"render_mode must be one of {self.metadata['render_modes']} or None."
            )

        spec = build_robot_spec(
            arm_descriptor=self.arm_descriptor,
            hand_descriptor=self.hand_descriptor,
            base_descriptor=self.base_descriptor,
            rot_xyz_deg=self.config.hand_attach_rot_xyz_deg,
            attach_point_name=self.config.attach_point_name,
            base_mount_site_name=self.config.base_mount_site_name,
            hand_prefix=self.hand_prefix,
            tactile_sensor=self.tactile_sensor,
            add_tactile_sensors=self.config.enable_tactile_sensors,
        )
        self._augment_spec(spec)
        self.model = spec.compile()
        self.data = mujoco.MjData(self.model)

        self.controller.bind(self.model, self.data)
        self.tactile_sensor.bind(self.model, self.data)
        self.task.bind(self.model)

        self.action_space = self.controller.action_space
        obs_spaces = {
            "qpos": spaces.Box(-np.inf, np.inf, shape=(self.model.nq,), dtype=np.float32),
            "qvel": spaces.Box(-np.inf, np.inf, shape=(self.model.nv,), dtype=np.float32),
            "ctrl": spaces.Box(-np.inf, np.inf, shape=(self.model.nu,), dtype=np.float32),
            "tactile": self.tactile_sensor.observation_space,
        }
        obs_spaces.update(self.task.observation_space)
        self.observation_space = spaces.Dict(obs_spaces)

        self.physics_steps_per_control = max(
            1, int(round(self.config.control_dt / self.model.opt.timestep))
        )
        self.controller.set_timestep(self.config.control_dt)
        self.elapsed_steps = 0
        self._initial_qpos = self.data.qpos.copy()
        self._initial_qvel = self.data.qvel.copy()
        self._renderer: Optional[mujoco.Renderer] = None
        self._viewer: Optional[viewer.Handle] = None
        self._reset_simulation_stats()

    @property
    def is_rendering(self) -> bool:
        """Whether a live passive viewer window is currently open."""
        return self._viewer is not None

    def set_control_mode(self, control_mode: str) -> None:
        self.controller.set_control_mode(control_mode)
        self.action_space = self.controller.action_space
        self.config = replace(self.config, control_mode=control_mode)

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[Observation, Dict[str, Any]]:
        super().reset(seed=seed)
        self.elapsed_steps = 0
        self.data.qpos[:] = self._initial_qpos
        self.data.qvel[:] = self._initial_qvel
        self.data.ctrl[:] = 0.0
        self._reset_simulation_stats()

        info: Dict[str, Any] = {}
        info.update(
            self.controller.reset(self.model, self.data, rng=self.np_random, options=options)
        )
        info.update(
            self.tactile_sensor.reset(self.model, self.data, rng=self.np_random, options=options)
        )
        info.update(self.task.reset(self.model, self.data, rng=self.np_random, options=options))

        mujoco.mj_forward(self.model, self.data)
        if self.render_mode == "human":
            self.render()
        return self._get_observation(), info

    def step(self, action: Any) -> Tuple[Observation, float, bool, bool, Dict[str, Any]]:
        info: Dict[str, Any] = {}
        info.update(self.controller.apply_action(self.model, self.data, action))
        self.step_physics(self.physics_steps_per_control, control_updates=1)

        self.elapsed_steps += 1
        obs = self._get_observation()
        task_result = self.task.evaluate(obs, np.asarray(action), self.model, self.data)
        reward = task_result.reward
        terminated = task_result.terminated
        truncated = self.elapsed_steps >= self.config.episode_length

        info.update(task_result.info)
        info["task_success"] = task_result.success
        info.update(self._get_info(obs))
        if self.render_mode == "human":
            self.render()
        return obs, float(reward), bool(terminated), bool(truncated), info

    def step_physics(self, physics_steps: int = 1, *, control_updates: int = 0) -> None:
        for _ in range(physics_steps):
            mujoco.mj_step(self.model, self.data)
        self.record_simulation_steps(physics_steps=physics_steps, control_updates=control_updates)

    def render(self) -> Optional[Array]:
        if self.render_mode is None:
            return None
        if self.render_mode == "human":
            self._render_human()
            return None
        if self.render_mode != "rgb_array":
            raise NotImplementedError(f"render_mode={self.render_mode!r} is not implemented.")
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model)
        self._renderer.update_scene(self.data)
        return self._renderer.render()

    def close(self) -> None:
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    def _render_human(self) -> None:
        if self._viewer is None:
            self._viewer = viewer.launch_passive(self.model, self.data)
        if not self._viewer.is_running():
            self._viewer.close()
            self._viewer = None
            return
        clear_markers(self._viewer)
        draw_stats_label(self._viewer, self._simulation_stats, control_label="ctrl")
        self._viewer.sync()

    # ------------------------------------------------------------------
    # Simulation statistics
    # ------------------------------------------------------------------

    def _reset_simulation_stats(self) -> None:
        now = time.perf_counter()
        self._stats_wall_start = now
        self._stats_last_wall = now
        self._stats_sim_start = float(self.data.time)
        self._stats_last_sim = float(self.data.time)
        self._stats_step_count = 0
        self._stats_last_step_count = 0
        self._stats_control_count = 0
        self._stats_last_control_count = 0
        self._simulation_stats: Dict[str, float] = {
            "sim_step_hz": 0.0,
            "real_time_factor": 0.0,
            "control_hz": 0.0,
            "sim_time": 0.0,
            "wall_time": 0.0,
        }

    def record_simulation_steps(
        self,
        *,
        physics_steps: int = 1,
        control_updates: int = 0,
    ) -> Dict[str, float]:
        self._stats_step_count += physics_steps
        self._stats_control_count += control_updates
        now = time.perf_counter()
        elapsed = now - self._stats_last_wall
        if elapsed >= self.config.stats_interval:
            sim_elapsed = float(self.data.time) - self._stats_last_sim
            step_elapsed = self._stats_step_count - self._stats_last_step_count
            control_elapsed = self._stats_control_count - self._stats_last_control_count
            self._simulation_stats = {
                "sim_step_hz": step_elapsed / elapsed,
                "real_time_factor": sim_elapsed / elapsed,
                "control_hz": control_elapsed / elapsed,
                "sim_time": float(self.data.time) - self._stats_sim_start,
                "wall_time": now - self._stats_wall_start,
            }
            self._stats_last_wall = now
            self._stats_last_sim = float(self.data.time)
            self._stats_last_step_count = self._stats_step_count
            self._stats_last_control_count = self._stats_control_count
        return self._simulation_stats.copy()

    @property
    def simulation_stats(self) -> Dict[str, float]:
        return self._simulation_stats.copy()

    # ------------------------------------------------------------------
    # Spec augmentation (scene & task objects)
    # ------------------------------------------------------------------

    def _augment_spec(self, spec: mujoco.MjSpec) -> None:
        if self.config.add_default_scene:
            add_basic_scene(spec)
        self.task.augment_spec(spec)
        if self.config.enable_task_objects:
            self._add_placeholder_task_objects(spec)

    def _add_placeholder_task_objects(self, spec: mujoco.MjSpec) -> None:
        table = spec.worldbody.add_body()
        table.name = "task_table"
        table.pos = [0.55, 0.0, 0.35]
        table_geom = table.add_geom()
        table_geom.name = "task_table_top"
        table_geom.type = mujoco.mjtGeom.mjGEOM_BOX
        table_geom.size = [0.35, 0.35, 0.03]
        table_geom.rgba = [0.45, 0.42, 0.38, 1.0]

    # ------------------------------------------------------------------
    # Observation & info
    # ------------------------------------------------------------------

    def _get_observation(self) -> Observation:
        obs: Observation = {
            "qpos": self.data.qpos.astype(np.float32).copy(),
            "qvel": self.data.qvel.astype(np.float32).copy(),
            "ctrl": self.data.ctrl.astype(np.float32).copy(),
            "tactile": self.tactile_sensor.read(self.model, self.data),
        }
        obs.update(self.task.get_observation(self.model, self.data))
        return obs

    def _get_info(self, obs: Observation) -> Dict[str, Any]:
        _ = obs
        return {
            "elapsed_steps": self.elapsed_steps,
            "physics_steps_per_control": self.physics_steps_per_control,
            "simulation_stats": self.simulation_stats,
        }


def make_env(
    *,
    task: Optional[RobotTask] = None,
    config: Optional[RLEnvConfig] = None,
    render_mode: Optional[str] = None,
    control_mode: Optional[str] = None,
    robot_config_path: Optional[str] = None,
    **config_overrides: Any,
) -> RobotGymEnv:
    resolved_config = config or load_env_config(robot_config_path, **config_overrides)
    if control_mode is not None:
        resolved_config = replace(resolved_config, control_mode=control_mode)
    return RobotGymEnv(task=task, config=resolved_config, render_mode=render_mode)
