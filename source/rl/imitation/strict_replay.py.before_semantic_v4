"""Strict C-MuJoCo grasp replay and reference-quality scoring."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from source.envs.manipulation import make_lift_env
from source.rl.residual.reference import EpisodeRecord, ReferenceTrajectory, resolve_reference_manifest
from source.rl.residual.trajectory import ResidualTrajectory


STRICT_REPLAY_SCHEMA_VERSION = 3


@dataclass(frozen=True)
class StrictReplayResult:
    success: bool
    frames: int
    final_lift: float
    max_lift: float
    tail_min_lift: float
    tail_mean_lift: float
    tail_max_speed: float
    tail_contact_fraction: float
    tail_grasp_fraction: float
    tail_opposition_mean: float
    robot_table_contacts: int
    max_penetration: float
    tail_tactile_max: float
    quality_score: float


def _manifest_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_dir() and (path / "manifest.json").is_file():
        return path / "manifest.json"
    return path


def _payload(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _signature(path: Path, settings: dict[str, Any]) -> str:
    stat = path.stat()
    data = {
        "schema": STRICT_REPLAY_SCHEMA_VERSION,
        "path": str(path.resolve()),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "settings": settings,
    }
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode("utf-8")).hexdigest()


def _contact_sets(env):
    bindings = env.task._require_bindings()
    object_ids = {int(value) for value in bindings.objects["object"].geom_ids}
    robot_ids = {int(value) for value in bindings.robot_geom_ids}
    table_ids: set[int] = set()
    for geom_id in range(env.model.ngeom):
        name = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
        lower = name.lower()
        if "table" in lower or lower == "floor":
            table_ids.add(geom_id)
    return object_ids, robot_ids, table_ids


def _contact_snapshot(env, sets) -> tuple[int, bool, float, int, float]:
    object_ids, robot_ids, table_ids = sets
    digit_flags = [False] * 5
    palm_contact = False
    thumb_normals: list[np.ndarray] = []
    finger_normals: list[np.ndarray] = []
    robot_table = 0
    max_penetration = 0.0
    for index in range(env.data.ncon):
        contact = env.data.contact[index]
        first, second = int(contact.geom1), int(contact.geom2)
        pair = {first, second}
        if pair & robot_ids:
            max_penetration = max(max_penetration, max(0.0, -float(contact.dist)))
        if pair & table_ids:
            other = second if first in table_ids else first
            if other in robot_ids:
                robot_table += 1
        robot_geom = -1
        if first in robot_ids and second in object_ids:
            robot_geom = first
            normal = np.asarray(contact.frame[:3], dtype=np.float64)
        elif second in robot_ids and first in object_ids:
            robot_geom = second
            normal = -np.asarray(contact.frame[:3], dtype=np.float64)
        else:
            continue
        norm = float(np.linalg.norm(normal))
        if norm > 1e-8:
            normal = normal / norm
        name = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_GEOM, robot_geom) or ""
        if "skin_palm_p" in name:
            palm_contact = True
        for digit in range(5):
            if f"skin_{digit}_" not in name:
                continue
            digit_flags[digit] = True
            if digit == 4:
                thumb_normals.append(normal)
            else:
                finger_normals.append(normal)
            break
    opposition = 0.0
    for thumb in thumb_normals:
        for finger in finger_normals:
            opposition = max(opposition, float(np.clip(-np.dot(thumb, finger), 0.0, 1.0)))
    return sum(digit_flags), palm_contact, opposition, robot_table, max_penetration


def _load_reference_for_replay(path: Path, env):
    try:
        trajectory = ResidualTrajectory.load(path)
    except (OSError, KeyError, TypeError, ValueError):
        manifest = resolve_reference_manifest(path)
        episode = EpisodeRecord.load(manifest)
        reference = ReferenceTrajectory.from_episode(
            episode,
            env,
            source_manifest=manifest,
            start_stage="approach",
        )
        return (
            reference.controls,
            reference.initial_qpos,
            reference.initial_qvel,
            reference.initial_ctrl,
        )
    return trajectory.controls, trajectory.initial_qpos, trajectory.initial_qvel, None


def strict_replay_manifest(
    trajectory_or_manifest: str | Path,
    *,
    render_mode: str | None = None,
    success_lift_height: float = 0.055,
    maximum_object_speed: float = 0.10,
    opposition_threshold: float = 0.25,
    verify_tail: int = 20,
    extra_hold_steps: int = 12,
    maximum_penetration: float = 0.003,
    maximum_robot_table_contacts: int = 0,
    use_cache: bool = True,
) -> StrictReplayResult:
    manifest = _manifest_path(trajectory_or_manifest)
    payload = _payload(manifest)
    object_id = str(payload["object_id"])
    metadata = dict(payload.get("metadata", {}))
    control_dt = float(metadata.get("control_dt", 0.05))
    source_seed = int(metadata.get("source_seed", payload.get("seed", 0)))
    settings = {
        "success_lift_height": success_lift_height,
        "maximum_object_speed": maximum_object_speed,
        "opposition_threshold": opposition_threshold,
        "verify_tail": verify_tail,
        "extra_hold_steps": extra_hold_steps,
        "maximum_penetration": maximum_penetration,
        "maximum_robot_table_contacts": maximum_robot_table_contacts,
    }
    signature = _signature(manifest, settings)
    cache = manifest.parent / "strict_replay_v3.json"
    if use_cache and cache.is_file():
        try:
            cached = json.loads(cache.read_text(encoding="utf-8"))
            if cached.get("signature") == signature:
                return StrictReplayResult(**cached["result"])
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            pass

    env = make_lift_env(
        task_config={
            "object_id": object_id,
            "reward_shaping": False,
            "terminate_on_success": False,
        },
        control_mode="position",
        enable_tactile_sensors=True,
        render_mode=render_mode,
        control_dt=control_dt,
        episode_length=int(payload.get("frames", 300)) + extra_hold_steps + 20,
    )
    try:
        env.reset(seed=source_seed)
        controls, initial_qpos, initial_qvel, initial_ctrl = _load_reference_for_replay(manifest, env)
        controls = np.asarray(controls, dtype=np.float32)
        if controls.ndim != 2 or controls.shape[1:] != env.action_space.shape:
            raise ValueError(
                f"Strict replay control shape {controls.shape} does not match {env.action_space.shape}."
            )
        if np.asarray(initial_qpos).shape != env.data.qpos.shape:
            raise ValueError("Strict replay qpos does not match the current robot model.")
        env.data.qpos[:] = initial_qpos
        env.data.qvel[:] = initial_qvel
        if initial_ctrl is not None and np.asarray(initial_ctrl).shape == env.data.ctrl.shape:
            env.data.ctrl[:] = initial_ctrl
        mujoco.mj_forward(env.model, env.data)
        bindings = env.task._require_bindings()
        object_binding = bindings.objects["object"]
        qvel_adr = int(object_binding.qvel_adr)
        initial_z = float(env.data.xpos[object_binding.body_id, 2])
        sets = _contact_sets(env)

        lifts: list[float] = []
        speeds: list[float] = []
        contacts: list[float] = []
        grasps: list[float] = []
        oppositions: list[float] = []
        tactile_values: list[float] = []
        robot_table_contacts = 0
        max_penetration = 0.0

        sequence = list(controls)
        if len(controls):
            sequence.extend([controls[-1]] * max(0, int(extra_hold_steps)))
        for control in sequence:
            observation, _, terminated, truncated, _ = env.step(np.asarray(control, dtype=np.float32))
            z = float(env.data.xpos[object_binding.body_id, 2])
            lift = z - initial_z
            speed = float(np.linalg.norm(env.data.qvel[qvel_adr : qvel_adr + 3]))
            digit_count, palm, opposition, table_contacts, penetration = _contact_snapshot(env, sets)
            grasp_valid = (
                (digit_count >= 2 and opposition >= opposition_threshold)
                or (palm and digit_count >= 2)
            )
            lifts.append(lift)
            speeds.append(speed)
            contacts.append(float(digit_count > 0 or palm))
            grasps.append(float(grasp_valid))
            oppositions.append(opposition)
            robot_table_contacts += table_contacts
            max_penetration = max(max_penetration, penetration)
            tactile = observation.get("tactile") if isinstance(observation, dict) else None
            if tactile is not None:
                values = np.asarray(tactile, dtype=np.float32)
                tactile_values.append(float(np.max(np.abs(values))) if values.size else 0.0)
            else:
                tactile_values.append(0.0)
            if terminated or truncated:
                break

        if not lifts:
            raise RuntimeError("Strict replay produced no frames.")
        tail_count = min(max(1, int(verify_tail)), len(lifts))
        tail_lifts = np.asarray(lifts[-tail_count:], dtype=np.float64)
        tail_speeds = np.asarray(speeds[-tail_count:], dtype=np.float64)
        tail_contacts = np.asarray(contacts[-tail_count:], dtype=np.float64)
        tail_grasps = np.asarray(grasps[-tail_count:], dtype=np.float64)
        tail_oppositions = np.asarray(oppositions[-tail_count:], dtype=np.float64)
        tail_tactile = np.asarray(tactile_values[-tail_count:], dtype=np.float64)
        final_lift = float(lifts[-1])
        max_lift = float(np.max(lifts))
        tail_min = float(np.min(tail_lifts))
        tail_mean = float(np.mean(tail_lifts))
        tail_speed_max = float(np.max(tail_speeds))
        tail_contact_fraction = float(np.mean(tail_contacts))
        tail_grasp_fraction = float(np.mean(tail_grasps))
        tail_opposition_mean = float(np.mean(tail_oppositions))
        tail_tactile_max = float(np.max(tail_tactile))

        lift_scale = max(success_lift_height, 1e-6)
        quality = (
            2.0 * float(np.clip(tail_min / lift_scale, -1.0, 1.5))
            + 1.5 * float(np.clip(final_lift / lift_scale, -1.0, 1.5))
            + 0.50 * float(np.clip(max_lift / lift_scale, 0.0, 2.0))
            + 1.50 * tail_grasp_fraction
            + 0.75 * tail_opposition_mean
            - 0.75 * max(0.0, tail_speed_max / max(maximum_object_speed, 1e-6) - 1.0)
            - 0.10 * robot_table_contacts
            - 50.0 * max(0.0, max_penetration - maximum_penetration)
        )
        success = bool(
            tail_min >= success_lift_height
            and final_lift >= success_lift_height
            and tail_speed_max <= maximum_object_speed
            and tail_contact_fraction >= 0.80
            and tail_grasp_fraction >= 0.80
            and robot_table_contacts <= maximum_robot_table_contacts
            and max_penetration <= maximum_penetration
        )
        result = StrictReplayResult(
            success=success,
            frames=len(lifts),
            final_lift=final_lift,
            max_lift=max_lift,
            tail_min_lift=tail_min,
            tail_mean_lift=tail_mean,
            tail_max_speed=tail_speed_max,
            tail_contact_fraction=tail_contact_fraction,
            tail_grasp_fraction=tail_grasp_fraction,
            tail_opposition_mean=tail_opposition_mean,
            robot_table_contacts=int(robot_table_contacts),
            max_penetration=float(max_penetration),
            tail_tactile_max=tail_tactile_max,
            quality_score=float(quality),
        )
        if use_cache:
            temporary = cache.with_suffix(".json.tmp")
            temporary.write_text(
                json.dumps(
                    {
                        "schema_version": STRICT_REPLAY_SCHEMA_VERSION,
                        "signature": signature,
                        "result": asdict(result),
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            temporary.replace(cache)
        return result
    finally:
        env.close()
