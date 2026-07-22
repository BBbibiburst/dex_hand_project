"""Headless end-to-end smoke tests for the project's core simulation paths.

Run after code changes with::

    python -m tools.run_smoke_checks

The suite deliberately avoids viewers, OpenCV windows, real hardware, datasets,
and long-running training.  A non-zero exit code means at least one core check
failed, making this suitable for local automation and CI.
"""

from __future__ import annotations

import argparse
import importlib
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from source.envs.manipulation import make_manipulation_env, registered_tasks
from source.envs.manipulation.object_catalog import (
    lift_object_ids,
    pick_place_object_ids,
    push_object_ids,
    stack_object_ids,
)
from source.envs.rl_env import RobotGymEnv, load_env_config
from source.robots.config import descriptors_from_robot_config, load_robot_config
from source.sensors.tactile.signal_processing import (
    TactileSignalProcessor,
    TaxelPatch,
)


DEFAULT_PROFILES = (
    "configs/robot_profiles/rm75b_dex_hand.json",
    "configs/robot_profiles/rm75b_pika_gripper.json",
)


@dataclass(frozen=True)
class CheckResult:
    name: str
    status: str
    duration: float
    detail: str = ""


def _assert_observation(env: RobotGymEnv, observation: dict) -> None:
    if not env.observation_space.contains(observation):
        shapes = {key: np.asarray(value).shape for key, value in observation.items()}
        raise AssertionError(f"Observation is outside observation_space: {shapes}")
    for name, value in observation.items():
        array = np.asarray(value)
        if np.issubdtype(array.dtype, np.number) and not np.all(np.isfinite(array)):
            raise AssertionError(f"Observation {name!r} contains NaN or infinity.")


def _step_current_action(env: RobotGymEnv, steps: int) -> None:
    observation, _ = env.reset(seed=0)
    _assert_observation(env, observation)
    for _ in range(steps):
        action = env.controller.current_action(env.model, env.data).astype(np.float32)
        if not env.action_space.contains(action):
            raise AssertionError(
                f"Controller current action {action.shape} is outside {env.action_space}."
            )
        observation, reward, terminated, truncated, info = env.step(action)
        _assert_observation(env, observation)
        if not np.isfinite(reward):
            raise AssertionError(f"Reward is not finite: {reward}")
        if not isinstance(info, dict):
            raise AssertionError("Environment info is not a dictionary.")
        if terminated or truncated:
            break


def _check_profile(profile: str, steps: int) -> str:
    config = load_env_config(profile)
    env = RobotGymEnv(config=config, render_mode=None)
    try:
        _step_current_action(env, steps)
        tactile_shape = env.observation_space["tactile"].shape
        env.set_control_mode("ik")
        _step_current_action(env, steps)
        return (
            f"hand={config.hand_name}, model=(nq={env.model.nq}, nu={env.model.nu}), "
            f"tactile={tactile_shape}, modes=position+ik"
        )
    finally:
        env.close()


def _check_task(task_name: str, profile: str, steps: int) -> str:
    env = make_manipulation_env(
        task_name,
        robot_config_path=profile,
        control_mode="ik",
        enable_tactile_sensors=False,
        render_mode=None,
    )
    try:
        _step_current_action(env, steps)
        return f"observation keys={sorted(env.observation_space.spaces)}"
    finally:
        env.close()


def _check_object_catalog() -> str:
    lift_ids = lift_object_ids()
    pick_ids = pick_place_object_ids()
    push_ids = push_object_ids()
    stack_ids = stack_object_ids()
    if len(lift_ids) < 100 or len(pick_ids) < 100 or len(push_ids) < 100 or len(stack_ids) < 10:
        raise AssertionError(
            f"Insufficient object coverage: lift={len(lift_ids)}, "
            f"pick_place={len(pick_ids)}, push={len(push_ids)}, stack={len(stack_ids)}"
        )
    if not any(item.startswith("ycb:") for item in lift_ids):
        raise AssertionError("Lift catalogue contains no YCB objects.")
    if not any(item.startswith("egad:") for item in lift_ids):
        raise AssertionError("Lift catalogue contains no EGAD objects.")
    return (
        f"lift={len(lift_ids)}, pick_place={len(pick_ids)}, "
        f"push={len(push_ids)}, stack={len(stack_ids)}"
    )


def _check_surface_data(profile: str) -> str:
    config = load_robot_config(profile)
    _, hand, _ = descriptors_from_robot_config(config)
    if hand.tactile_sensor_factory is None:
        return f"hand={hand.name}: no tactile backend (not applicable)"
    sensor = hand.tactile_sensor_factory(
        str(config.get("tactile_backend", "simple_box")),
        **dict(config.get("tactile_options") or {}),
    )
    names = tuple(sensor.surface_patch_names())
    defaults = tuple(sensor.default_surface_patch_names())
    if not names or not defaults or not set(defaults).issubset(names):
        raise AssertionError(f"Invalid surface patch declarations: {names=}, {defaults=}")
    sample_count = 0
    for name in defaults:
        plot = sensor.surface_plot_data(name)
        if plot.samples.shape != (plot.rows * plot.cols, 3):
            raise AssertionError(
                f"Patch {name!r} samples have shape {plot.samples.shape}, "
                f"expected {(plot.rows * plot.cols, 3)}."
            )
        if not np.all(np.isfinite(plot.samples)):
            raise AssertionError(f"Patch {name!r} contains non-finite surface points.")
        sample_count += len(plot.samples)
    return f"hand={hand.name}, patches={defaults}, samples={sample_count}"


def _check_tactile_diffusion() -> str:
    patch = TaxelPatch("grid", rows=3, cols=3, kind="test", start=0, stop=9)
    raw = np.zeros(9, dtype=np.float32)
    raw[4] = 1.0
    output = TactileSignalProcessor({"crosstalk": 0.16, "gaussian_sigma": 0.0}).process(
        raw, (patch,)
    )
    neighbors = np.delete(output, 4)
    if not np.allclose(neighbors, neighbors[0]) or not np.isclose(output.sum(), 1.0):
        raise AssertionError(f"Eight-neighbor diffusion is asymmetric: {output.reshape(3, 3)}")
    return f"center={output[4]:.3f}, each neighbor={neighbors[0]:.3f}"


def _check_imports() -> str:
    modules = (
        "source.teleop.bluetooth_glove.bluetooth_glove_test",
        "source.teleop.glove_processing",
        "source.teleop.mapping",
        "source.teleop.vive.coordinates",
        "source.teleop.vive.hand_skeleton",
        "source.teleop.vive.vive_link_test",
        "source.teleop.vive.vive_glove_hand_control",
        "apps.collect_teleop_lerobot",
        "apps.collect_scripted_lerobot",
        "examples.robot_preview",
        "examples.random_control",
        "examples.ik_trajectory",
        "examples.manipulation_task_playback",
        "tools.tactile.preview_layout",
        "tools.tactile.interactive_probe",
        "tools.tactile.plot_surfaces",
        "tools.tactile.validate_contacts",
        "tools.grasping.search_grasp",
        "tools.grasping.validate_grasp",
        "tools.grasping.validate_scripted_strategy",
        "tools.grasping.benchmark_catalog",
        "tools.grasping.visualize_benchmark",
    )
    for module in modules:
        importlib.import_module(module)
    return f"imported {len(modules)} runnable application, example, and tool modules"


def _run_check(name: str, function: Callable[[], str]) -> CheckResult:
    started = time.perf_counter()
    try:
        detail = function()
    except Exception:
        return CheckResult(
            name, "FAIL", time.perf_counter() - started, traceback.format_exc().rstrip()
        )
    return CheckResult(name, "PASS", time.perf_counter() - started, detail)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile",
        action="append",
        default=[],
        help="Robot config to test; repeat for multiple profiles (default: Dex Hand and Pika).",
    )
    parser.add_argument("--steps", type=int, default=2, help="Control steps per mode/task.")
    parser.add_argument("--skip-tasks", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    return parser.parse_args()


def run(args: argparse.Namespace) -> int:
    if args.steps < 1:
        raise ValueError("--steps must be at least 1.")
    profiles = tuple(args.profile) or DEFAULT_PROFILES
    checks: list[tuple[str, Callable[[], str]]] = [
        ("entrypoint imports", _check_imports),
        ("tactile 8-neighbor diffusion", _check_tactile_diffusion),
        ("manipulation object catalogue", _check_object_catalog),
    ]
    for profile in profiles:
        label = Path(profile).stem
        checks.append((f"robot profile: {label}", lambda p=profile: _check_profile(p, args.steps)))
        checks.append((f"surface data: {label}", lambda p=profile: _check_surface_data(p)))
    if not args.skip_tasks:
        task_profile = next(
            (profile for profile in profiles if "pika" in profile.lower()), profiles[0]
        )
        for task_name in registered_tasks():
            checks.append(
                (
                    f"manipulation task: {task_name}",
                    lambda name=task_name: _check_task(name, task_profile, args.steps),
                )
            )

    print("Headless smoke test (no viewer, camera, or hardware)\n")
    results: list[CheckResult] = []
    for name, function in checks:
        print(f"[....] {name}", flush=True)
        result = _run_check(name, function)
        results.append(result)
        print(f"[{result.status}] {name} ({result.duration:.2f}s)")
        if result.detail:
            prefix = "       "
            print("\n".join(prefix + line for line in result.detail.splitlines()))
        if result.status == "FAIL" and args.fail_fast:
            break

    passed = sum(result.status == "PASS" for result in results)
    failed = sum(result.status == "FAIL" for result in results)
    print(f"\nSummary: {passed} passed, {failed} failed, {len(results)} total")
    if failed:
        print("Hardware, teleoperation, rendering, datasets, and training were not exercised.")
    return 1 if failed else 0


def main() -> None:
    raise SystemExit(run(_parse_args()))


if __name__ == "__main__":
    main()
