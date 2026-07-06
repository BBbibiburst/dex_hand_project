# -*- coding: utf-8 -*-
"""Gymnasium environment for RM75B + dex hand.

This module intentionally exposes one environment and one controller. The
controller supports both direct actuator position control and end-effector IK;
the environment owns timing, viewer markers, and runtime diagnostics.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import time
from typing import Any, Dict, Optional, Sequence, Tuple

import gymnasium as gym
from gymnasium import spaces
import mujoco
from mujoco import viewer
import numpy as np

try:
    from .controllers import Rm75bDexHandController
    from .robot_builder import DEFAULT_HAND_PREFIX, build_combined_spec
    from .tactile_sensors import NullTactileSensor, TactileSensorBase
except ImportError:  # Allow running this file directly for local debugging.
    from controllers import Rm75bDexHandController
    from robot_builder import DEFAULT_HAND_PREFIX, build_combined_spec
    from tactile_sensors import NullTactileSensor, TactileSensorBase


Array = np.ndarray
Observation = Dict[str, Any]


def _quat_multiply(lhs: Array, rhs: Array) -> Array:
    lw, lx, ly, lz = lhs
    rw, rx, ry, rz = rhs
    return np.asarray(
        [
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ],
        dtype=np.float32,
    )


def _axis_angle_quat(axis: Array, angle: float) -> Array:
    axis = np.asarray(axis, dtype=np.float32)
    axis /= np.linalg.norm(axis)
    half = 0.5 * angle
    return np.concatenate([[np.cos(half)], np.sin(half) * axis]).astype(np.float32)


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


class DexHandGymEnv(gym.Env):
    """One environment for direct position control and IK control."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(
        self,
        tactile_sensor: Optional[TactileSensorBase] = None,
        config: Optional[RLEnvConfig] = None,
        render_mode: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.config = config or RLEnvConfig()
        self.controller = Rm75bDexHandController(
            hand_prefix=self.config.hand_prefix,
            control_mode=self.config.control_mode,
            ee_site_name=self.config.ee_site_name,
            include_hand_action=self.config.include_hand_action,
            normalized_position=self.config.normalized_position,
        )
        self.tactile_sensor = tactile_sensor or NullTactileSensor()
        self.render_mode = render_mode

        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            raise ValueError(f"render_mode must be one of {self.metadata['render_modes']} or None.")

        spec = build_combined_spec(hand_prefix=self.config.hand_prefix)
        self._augment_spec(spec)
        self.model = spec.compile()
        self.data = mujoco.MjData(self.model)

        self.controller.bind(self.model, self.data)
        self.tactile_sensor.bind(self.model, self.data)

        self.action_space = self.controller.action_space
        self.observation_space = spaces.Dict(
            {
                "qpos": spaces.Box(-np.inf, np.inf, shape=(self.model.nq,), dtype=np.float32),
                "qvel": spaces.Box(-np.inf, np.inf, shape=(self.model.nv,), dtype=np.float32),
                "ctrl": spaces.Box(-np.inf, np.inf, shape=(self.model.nu,), dtype=np.float32),
                "tactile": self.tactile_sensor.observation_space,
                **self._task_observation_space(),
            }
        )

        self.physics_steps_per_control = max(
            1,
            int(round(self.config.control_dt / self.model.opt.timestep)),
        )
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
        info.update(self.controller.reset(self.model, self.data, rng=self.np_random, options=options))
        info.update(self.tactile_sensor.reset(self.model, self.data, rng=self.np_random, options=options))
        info.update(self._reset_task(options or {}))

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
        reward, reward_info = self._compute_reward(obs, action)
        terminated, terminated_info = self._is_terminated(obs)
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
        self.clear_viewer_markers(self._viewer)
        self.draw_simulation_stats_label(self._viewer)
        self._viewer.sync()

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
        self._simulation_stats = {
            "sim_step_hz": 0.0,
            "real_time_factor": 0.0,
            "control_hz": 0.0,
            "sim_time": 0.0,
            "wall_time": 0.0,
        }
        self._simulation_stats_text = "sim: measuring..."

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
            self._simulation_stats_text = (
                f"sim {self._simulation_stats['sim_step_hz']:5.0f} Hz | "
                f"RTF {self._simulation_stats['real_time_factor']:4.2f} | "
                f"ctrl {self._simulation_stats['control_hz']:4.1f} Hz"
            )
            self._stats_last_wall = now
            self._stats_last_sim = float(self.data.time)
            self._stats_last_step_count = self._stats_step_count
            self._stats_last_control_count = self._stats_control_count
        return self._simulation_stats.copy()

    @property
    def simulation_stats(self) -> Dict[str, float]:
        return self._simulation_stats.copy()

    @property
    def simulation_stats_text(self) -> str:
        return self._simulation_stats_text

    def format_simulation_stats(self, control_label: str = "ctrl") -> str:
        return self._simulation_stats_text.replace("ctrl", control_label, 1)

    def clear_viewer_markers(self, handle: viewer.Handle) -> None:
        handle.user_scn.ngeom = 0

    def draw_target_marker(
        self,
        handle: viewer.Handle,
        target_pos: Array,
        *,
        radius: float = 0.018,
        rgba: Optional[Sequence[float]] = None,
    ) -> None:
        scene = handle.user_scn
        if scene.ngeom >= scene.maxgeom:
            return
        color = [0.0, 0.9, 1.0, 0.55] if rgba is None else rgba
        mujoco.mjv_initGeom(
            scene.geoms[scene.ngeom],
            mujoco.mjtGeom.mjGEOM_SPHERE,
            np.asarray([radius, radius, radius], dtype=np.float64),
            np.asarray(target_pos, dtype=np.float64),
            np.eye(3, dtype=np.float64).reshape(9),
            np.asarray(color, dtype=np.float32),
        )
        scene.ngeom += 1

    def draw_viewer_label(
        self,
        handle: viewer.Handle,
        label_pos: Array,
        text: str,
        *,
        rgba: Optional[Sequence[float]] = None,
    ) -> None:
        scene = handle.user_scn
        if scene.ngeom >= scene.maxgeom:
            return
        color = [1.0, 1.0, 1.0, 1.0] if rgba is None else rgba
        geom = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(
            geom,
            mujoco.mjtGeom.mjGEOM_LABEL,
            np.zeros(3, dtype=np.float64),
            np.asarray(label_pos, dtype=np.float64),
            np.eye(3, dtype=np.float64).reshape(9),
            np.asarray(color, dtype=np.float32),
        )
        geom.label = text
        scene.ngeom += 1

    def draw_simulation_stats_label(
        self,
        handle: viewer.Handle,
        label_pos: Optional[Array] = None,
        *,
        control_label: str = "ctrl",
    ) -> None:
        if label_pos is None:
            label_pos = np.asarray([0.0, -0.25, 1.2], dtype=np.float32)
        self.draw_viewer_label(
            handle,
            label_pos,
            self.format_simulation_stats(control_label=control_label),
        )

    def run_ik_sine_viewer(
        self,
        *,
        frequency: float = 0.08,
        render_fps: float = 60.0,
        realtime: bool = True,
        radius: float = 0.03,
        z_scale: float = 0.02,
        rot_scale: float = 0.25,
        hand_scale: float = 0.85,
        draw_target_marker: bool = True,
        target_marker_size: float = 0.018,
        draw_stats_label: bool = True,
        print_error: bool = False,
    ) -> None:
        previous_mode = self.controller.control_mode
        if previous_mode != "ik":
            self.set_control_mode("ik")

        controller = self.controller
        base_action = controller.current_ik_action(self.model, self.data)
        base_pos = base_action[:3].copy()
        base_quat = base_action[3:7].copy()
        stats_label_pos = base_pos + np.asarray([0.0, -0.16, 0.16], dtype=np.float32)

        hand_center = None
        hand_half_range = None
        if controller.include_hand_action:
            hand_low = controller.ctrl_low[7:].astype(np.float32)
            hand_high = controller.ctrl_high[7:].astype(np.float32)
            hand_center = 0.5 * (hand_low + hand_high)
            hand_half_range = 0.5 * (hand_high - hand_low)

        render_dt = 1.0 / render_fps
        wall_start = time.perf_counter()
        sim_start = float(self.data.time)
        last_control_sim_time = -np.inf
        last_print_second = -1
        action = base_action.copy()
        info: Dict[str, Any] = {"ik_error": np.zeros(6, dtype=np.float32), "ik_iterations": 0}

        with viewer.launch_passive(self.model, self.data) as handle:
            while handle.is_running():
                frame_sim_start = float(self.data.time)
                frame_steps = 0
                frame_control_updates = 0

                while float(self.data.time) - frame_sim_start < render_dt:
                    sim_elapsed = float(self.data.time) - sim_start
                    if float(self.data.time) - last_control_sim_time >= self.config.control_dt:
                        last_control_sim_time = float(self.data.time)
                        phase = 2.0 * np.pi * frequency * sim_elapsed
                        action = base_action.copy()
                        action[0] = base_pos[0] + radius * np.cos(phase)
                        action[1] = base_pos[1] + radius * np.sin(phase)
                        action[2] = base_pos[2] + z_scale * np.sin(2.0 * phase)

                        yaw = _axis_angle_quat([0.0, 0.0, 1.0], rot_scale * np.sin(phase))
                        pitch = _axis_angle_quat([0.0, 1.0, 0.0], 0.5 * rot_scale * np.cos(phase))
                        action[3:7] = _quat_multiply(_quat_multiply(yaw, pitch), base_quat)

                        if hand_center is not None and hand_half_range is not None:
                            hand_phase = phase + 0.6 * np.arange(hand_center.size, dtype=np.float32)
                            action[7:] = hand_center + hand_scale * hand_half_range * np.sin(hand_phase)

                        info = controller.apply_action(self.model, self.data, action)
                        frame_control_updates += 1

                    mujoco.mj_step(self.model, self.data)
                    frame_steps += 1

                self.record_simulation_steps(
                    physics_steps=frame_steps,
                    control_updates=frame_control_updates,
                )
                self.clear_viewer_markers(handle)
                if draw_target_marker:
                    self.draw_target_marker(handle, action[:3], radius=target_marker_size)
                if draw_stats_label:
                    self.draw_simulation_stats_label(handle, stats_label_pos, control_label="IK")
                handle.sync()

                sim_elapsed = float(self.data.time) - sim_start
                if print_error and int(sim_elapsed) != last_print_second:
                    last_print_second = int(sim_elapsed)
                    print(
                        "ik_error=",
                        np.array2string(info["ik_error"], precision=4, suppress_small=True),
                        "iterations=",
                        info["ik_iterations"],
                        "stats=",
                        self.format_simulation_stats(control_label="IK"),
                    )

                if realtime:
                    sleep_time = wall_start + sim_elapsed - time.perf_counter()
                    if sleep_time > 0.0:
                        time.sleep(sleep_time)

        if previous_mode != "ik":
            self.set_control_mode(previous_mode)

    def _augment_spec(self, spec: mujoco.MjSpec) -> None:
        if self.config.add_default_scene:
            self._add_default_scene(spec)
        if self.config.enable_task_objects:
            self._add_placeholder_task_objects(spec)

    def _add_default_scene(self, spec: mujoco.MjSpec) -> None:
        floor = spec.worldbody.add_geom()
        floor.name = "floor"
        floor.type = mujoco.mjtGeom.mjGEOM_PLANE
        floor.size = [0.0, 0.0, 0.05]
        floor.rgba = [0.25, 0.25, 0.25, 1.0]
        spec.worldbody.add_light(
            name="rl_top_light",
            pos=[0.0, 0.0, 4.0],
            dir=[0.0, 0.0, -1.0],
            diffuse=[1.5, 1.5, 1.5],
            ambient=[0.5, 0.5, 0.5],
        )

    def _add_placeholder_task_objects(self, spec: mujoco.MjSpec) -> None:
        table = spec.worldbody.add_body()
        table.name = "task_table"
        table.pos = [0.55, 0.0, 0.35]
        table_geom = table.add_geom()
        table_geom.name = "task_table_top"
        table_geom.type = mujoco.mjtGeom.mjGEOM_BOX
        table_geom.size = [0.35, 0.35, 0.03]
        table_geom.rgba = [0.45, 0.42, 0.38, 1.0]

    def _get_observation(self) -> Observation:
        obs: Observation = {
            "qpos": self.data.qpos.astype(np.float32).copy(),
            "qvel": self.data.qvel.astype(np.float32).copy(),
            "ctrl": self.data.ctrl.astype(np.float32).copy(),
            "tactile": self.tactile_sensor.read(self.model, self.data),
        }
        obs.update(self._get_task_observation())
        return obs

    def _task_observation_space(self) -> Dict[str, spaces.Space]:
        return {}

    def _get_task_observation(self) -> Observation:
        return {}

    def _reset_task(self, options: dict) -> Dict[str, Any]:
        _ = options
        return {}

    def _compute_reward(self, obs: Observation, action: Any) -> Tuple[float, Dict[str, Any]]:
        _ = obs
        _ = action
        return 0.0, {}

    def _is_terminated(self, obs: Observation) -> Tuple[bool, Dict[str, Any]]:
        _ = obs
        return False, {}

    def _get_info(self, obs: Observation) -> Dict[str, Any]:
        _ = obs
        return {
            "elapsed_steps": self.elapsed_steps,
            "physics_steps_per_control": self.physics_steps_per_control,
            "simulation_stats": self.simulation_stats,
        }


def make_env(
    config: Optional[RLEnvConfig] = None,
    render_mode: Optional[str] = None,
    *,
    control_mode: Optional[str] = None,
) -> DexHandGymEnv:
    resolved_config = config or RLEnvConfig()
    if control_mode is not None:
        resolved_config = replace(resolved_config, control_mode=control_mode)
    return DexHandGymEnv(config=resolved_config, render_mode=render_mode)


def _parse_demo_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dex hand environment demos.")
    parser.add_argument("--demo", choices=("ik_sine", "random"), default="ik_sine")
    parser.add_argument("--frequency", type=float, default=0.08)
    parser.add_argument("--render-fps", type=float, default=60.0)
    parser.add_argument("--no-realtime", action="store_true")
    parser.add_argument("--radius", type=float, default=0.03)
    parser.add_argument("--z-scale", type=float, default=0.02)
    parser.add_argument("--rot-scale", type=float, default=0.25)
    parser.add_argument("--hand-scale", type=float, default=0.85)
    parser.add_argument("--no-target-marker", action="store_true")
    parser.add_argument("--target-marker-size", type=float, default=0.018)
    parser.add_argument("--no-stats-label", action="store_true")
    parser.add_argument("--print-error", action="store_true")
    return parser.parse_args()


def _run_random_demo() -> None:
    env = make_env(render_mode="human", control_mode="position")
    obs, _ = env.reset(seed=0)
    reward = 0.0
    terminated = False
    truncated = False
    info: Dict[str, Any] = {}
    try:
        while True:
            obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
            if env._viewer is None:
                break
            if terminated or truncated:
                obs, info = env.reset()
                terminated = False
                truncated = False
            time.sleep(env.config.control_dt)
    finally:
        env.close()
    print("observation keys:", list(obs.keys()))
    print("action shape:", env.action_space.shape)
    print("reward:", reward)
    print("terminated:", terminated)
    print("truncated:", truncated)
    print("info:", info)


def _run_ik_sine_demo(args: argparse.Namespace) -> None:
    env = make_env(control_mode="ik")
    env.reset(seed=0)
    try:
        env.run_ik_sine_viewer(
            frequency=args.frequency,
            render_fps=args.render_fps,
            realtime=not args.no_realtime,
            radius=args.radius,
            z_scale=args.z_scale,
            rot_scale=args.rot_scale,
            hand_scale=args.hand_scale,
            draw_target_marker=not args.no_target_marker,
            target_marker_size=args.target_marker_size,
            draw_stats_label=not args.no_stats_label,
            print_error=args.print_error,
        )
    finally:
        env.close()


if __name__ == "__main__":
    demo_args = _parse_demo_args()
    if demo_args.demo == "random":
        _run_random_demo()
    else:
        _run_ik_sine_demo(demo_args)
