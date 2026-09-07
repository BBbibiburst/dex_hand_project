"""Independent C-MuJoCo stability replay for transferred task trajectories."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from source.envs.manipulation import make_nut_assembly_env

NUT_STABILITY_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class NutStabilityReplayResult:
    success: bool
    frames: int
    successful_frames: int
    tail_success_fraction: float
    final_success: bool
    maximum_position_drift: float
    maximum_linear_speed: float
    maximum_angular_speed: float
    final_position: tuple[float, float, float]


def _manifest_path(path: str | Path) -> Path:
    path = Path(path)
    return path / "manifest.json" if path.is_dir() else path


def _signature(manifest: Path, *, hold_steps: int, tail_steps: int) -> str:
    arrays = manifest.parent / json.loads(manifest.read_text(encoding="utf-8"))["arrays"]
    payload = {
        "schema_version": NUT_STABILITY_SCHEMA_VERSION,
        "manifest": str(manifest.resolve()),
        "manifest_size": manifest.stat().st_size,
        "manifest_mtime_ns": manifest.stat().st_mtime_ns,
        "arrays_size": arrays.stat().st_size,
        "arrays_mtime_ns": arrays.stat().st_mtime_ns,
        "hold_steps": hold_steps,
        "tail_steps": tail_steps,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def replay_nut_terminal_stability(
    trajectory_or_manifest: str | Path,
    *,
    hold_steps: int = 100,
    tail_steps: int = 40,
    maximum_position_drift: float = 0.005,
    maximum_linear_speed: float = 0.03,
    maximum_angular_speed: float = 0.4,
    use_cache: bool = True,
) -> NutStabilityReplayResult:
    """Rebuild a NutAssembly terminal state and verify a sustained physical hold.

    The replay deliberately uses low-level position control. This avoids relying
    on hidden IK-controller history while preserving the recorded actuator
    targets and the complete robot/object dynamic state.
    """

    if hold_steps < 1 or not 1 <= tail_steps <= hold_steps:
        raise ValueError("hold_steps must be positive and tail_steps must lie within it.")
    if min(maximum_position_drift, maximum_linear_speed, maximum_angular_speed) <= 0.0:
        raise ValueError("Stability limits must be positive.")

    manifest = _manifest_path(trajectory_or_manifest)
    payload: dict[str, Any] = json.loads(manifest.read_text(encoding="utf-8"))
    metadata = dict(payload.get("metadata", {}))
    if metadata.get("task") != "nut_assembly":
        raise ValueError(f"Expected a nut_assembly trajectory, got {metadata.get('task')!r}.")
    if not payload.get("success"):
        raise ValueError("Stability replay requires a successful source trajectory.")

    signature = _signature(manifest, hold_steps=hold_steps, tail_steps=tail_steps)
    cache = manifest.parent / "nut_stability_replay.json"
    if use_cache and cache.is_file():
        try:
            cached = json.loads(cache.read_text(encoding="utf-8"))
            if cached.get("signature") == signature:
                return NutStabilityReplayResult(**cached["result"])
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            pass

    arrays_path = manifest.parent / str(payload["arrays"])
    with np.load(arrays_path) as arrays:
        final_qpos = np.asarray(arrays["qpos"][-1], dtype=np.float64)
        final_qvel = np.asarray(arrays["qvel"][-1], dtype=np.float64)
        final_ctrl = np.asarray(arrays["ctrl"][-1], dtype=np.float64)

    task_config = dict(metadata["task_config"])
    task_config["terminate_on_success"] = False
    env = make_nut_assembly_env(
        task_config=task_config,
        control_mode="position",
        enable_tactile_sensors=True,
        render_mode=None,
        episode_length=hold_steps + 20,
    )
    try:
        env.reset(seed=int(payload.get("seed", 0)))
        if final_qpos.shape != env.data.qpos.shape or final_qvel.shape != env.data.qvel.shape:
            raise ValueError("Recorded terminal state does not match the current Nut environment.")
        if final_ctrl.shape != env.data.ctrl.shape:
            raise ValueError("Recorded terminal controls do not match the current Nut environment.")
        env.data.qpos[:] = final_qpos
        env.data.qvel[:] = final_qvel
        env.data.ctrl[:] = final_ctrl
        mujoco.mj_forward(env.model, env.data)

        binding = env.task._require_bindings().objects["assembly_bolt"]
        start = env.data.xpos[binding.body_id].copy()
        successes: list[bool] = []
        drifts: list[float] = []
        linear_speeds: list[float] = []
        angular_speeds: list[float] = []
        for _ in range(hold_steps):
            _, _, _, truncated, info = env.step(final_ctrl.astype(np.float32))
            position = env.data.xpos[binding.body_id]
            velocity = env.data.qvel[binding.qvel_adr : binding.qvel_adr + 6]
            successes.append(bool(info.get("task_success", False)))
            drifts.append(float(np.linalg.norm(position - start)))
            linear_speeds.append(float(np.linalg.norm(velocity[:3])))
            angular_speeds.append(float(np.linalg.norm(velocity[3:])))
            if truncated:
                break

        tail = np.asarray(successes[-tail_steps:], dtype=bool)
        tail_success_fraction = float(np.mean(tail))
        drift_max = float(np.max(drifts))
        linear_max = float(np.max(linear_speeds[-tail_steps:]))
        angular_max = float(np.max(angular_speeds[-tail_steps:]))
        result = NutStabilityReplayResult(
            success=bool(
                len(successes) == hold_steps
                and bool(tail[-1])
                and tail_success_fraction == 1.0
                and drift_max <= maximum_position_drift
                and linear_max <= maximum_linear_speed
                and angular_max <= maximum_angular_speed
            ),
            frames=len(successes),
            successful_frames=int(np.sum(successes)),
            tail_success_fraction=tail_success_fraction,
            final_success=bool(tail[-1]),
            maximum_position_drift=drift_max,
            maximum_linear_speed=linear_max,
            maximum_angular_speed=angular_max,
            final_position=tuple(float(value) for value in env.data.xpos[binding.body_id]),
        )
    finally:
        env.close()

    if use_cache:
        temporary = cache.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(
                {
                    "schema_version": NUT_STABILITY_SCHEMA_VERSION,
                    "signature": signature,
                    "settings": {"hold_steps": hold_steps, "tail_steps": tail_steps},
                    "result": asdict(result),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        temporary.replace(cache)
    return result
