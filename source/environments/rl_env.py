# -*- coding: utf-8 -*-
"""Gymnasium environment for RM75B + dex hand.

The environment owns timing, rendering, and runtime diagnostics.
Control logic is delegated to ``Rm75bDexHandController``;
task logic is delegated to ``DexHandTask`` implementations.
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
from source.environments.controllers import Rm75bDexHandController
from source.environments.overlays import clear_markers, draw_stats_label
from source.environments.robot_builder import DEFAULT_HAND_PREFIX, build_combined_spec
from source.environments.scene import add_basic_scene
from source.environments.tactile_sensors import (
    DexHandTouchSensor,
    NullTactileSensor,
    TactileSensorBase,
)
from source.environments.tasks import DexHandTask, NoopTask


Array = np.ndarray
Observation = Dict[str, Any]


@dataclass(frozen=True)
class RLEnvConfig:
    control_dt: float = 0.05  # 20 Hz controller updates.
    episode_length: int = 500
    add_default_scene: bool = True
    enable_task_objects: bool = False
    hand_prefix: str = DEFAULT_HAND_PREFIX
    stats_interval: float = 0.5
    control_mode: str = "position"  # "position" or "ik".
    ee_site_name: str = "right_hand_site"
    include_hand_action: bool = True
    normalized_position: bool = False
    enable_tactile_sensors: bool = True


class DexHandGymEnv(gym.Env):
    """Gymnasium environment for RM75B + dex hand.

    Delegates control to ``Rm75bDexHandController`` and task logic to a
    ``DexHandTask`` instance.  Supports both direct position control and
    end-effector IK modes.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(
        self,
        *,
        task: Optional[DexHandTask] = None,
        tactile_sensor: Optional[TactileSensorBase] = None,
        config: Optional[RLEnvConfig] = None,
        render_mode: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.config = config or RLEnvConfig()
        self.task = task or NoopTask()
        self.controller = Rm75bDexHandController(
            hand_prefix=self.config.hand_prefix,
            control_mode=self.config.control_mode,
            ee_site_name=self.config.ee_site_name,
            include_hand_action=self.config.include_hand_action,
            normalized_position=self.config.normalized_position,
        )
        if tactile_sensor is not None:
            self.tactile_sensor = tactile_sensor
        elif self.config.enable_tactile_sensors:
            self.tactile_sensor = DexHandTouchSensor(hand_prefix=self.config.hand_prefix)
        else:
            self.tactile_sensor = NullTactileSensor()
        self.render_mode = render_mode

        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            raise ValueError(
                f"render_mode must be one of {self.metadata['render_modes']} or None."
            )

        spec = build_combined_spec(
            hand_prefix=self.config.hand_prefix,
            add_tactile_sensors=self.config.enable_tactile_sensors,
        )
        self._augment_spec(spec)
        self.model = spec.compile()
        self.data = mujoco.MjData(self.model)

        self.controller.bind(self.model, self.data)
        self.tactile_sensor.bind(self.model, self.data)

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
            self.controller.reset(
                self.model, self.data, rng=self.np_random, options=options
            )
        )
        info.update(
            self.tactile_sensor.reset(
                self.model, self.data, rng=self.np_random, options=options
            )
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
        reward, reward_info = self.task.compute_reward(
            obs, action, self.model, self.data
        )
        terminated, terminated_info = self.task.is_terminated(obs, self.model, self.data)
        truncated = self.elapsed_steps >= self.config.episode_length

        info.update(reward_info)
        info.update(terminated_info)
        info.update(self._get_info(obs))
        if self.render_mode == "human":
            self.render()
        return obs, float(reward), bool(terminated), bool(truncated), info

    def step_physics(self, physics_steps: int = 1, *, control_updates: int = 0) -> None:
        for _ in range(physics_steps):
            mujoco.mj_step(self.model, self.data)
        self.record_simulation_steps(
            physics_steps=physics_steps,
            control_updates=control_updates,
        )

    def render(self) -> Optional[Array]:
        if self.render_mode is None:
            return None
        if self.render_mode == "human":
            self._render_human()
            return None
        if self.render_mode != "rgb_array":
            raise NotImplementedError(
                f"render_mode={self.render_mode!r} is not implemented."
            )
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
        draw_stats_label(
            self._viewer, self._simulation_stats, control_label="ctrl"
        )
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
            control_elapsed = (
                self._stats_control_count - self._stats_last_control_count
            )
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
    task: Optional[DexHandTask] = None,
    config: Optional[RLEnvConfig] = None,
    render_mode: Optional[str] = None,
    control_mode: Optional[str] = None,
) -> DexHandGymEnv:
    resolved_config = config or RLEnvConfig()
    if control_mode is not None:
        resolved_config = replace(resolved_config, control_mode=control_mode)
    return DexHandGymEnv(task=task, config=resolved_config, render_mode=render_mode)
