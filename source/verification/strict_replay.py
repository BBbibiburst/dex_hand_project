"""Strict C-MuJoCo grasp replay and demonstration-quality scoring.

Only final-verified trajectories may be exported as automatic DP demonstrations.

The contact classifier does not rely exclusively on ``skin_*`` geoms.  Any
robot-object contacts contribute to physical contact and opposition; tactile
skin naming is kept only for digit/palm diagnostics.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from source.envs.manipulation import make_lift_env
from source.verification.profiles import FINAL_PROFILE, verification_status
from source.grasp_pipeline.reference import (
    EpisodeRecord,
    ReferenceTrajectory,
    resolve_reference_manifest,
)
from source.grasp_pipeline.trajectory import GraspTrajectory

STRICT_REPLAY_SCHEMA_VERSION = 6

_PROFILE_DEFAULTS = {
    FINAL_PROFILE: {
        "success_lift_height": 0.055,
        "maximum_object_speed": 0.10,
        "maximum_object_angular_speed": 0.10,
        "opposition_threshold": 0.25,
        "verify_tail": 20,
        "extra_hold_steps": 12,
        "maximum_penetration": 0.003,
        "minimum_tail_contact_fraction": 0.80,
        "minimum_tail_grasp_fraction": 0.80,
        "maximum_tail_table_contact_fraction": 0.0,
    },
}


@dataclass(frozen=True)
class StrictReplayResult:
    success: bool
    profile: str
    verification_status: str
    frames: int
    final_lift: float
    max_lift: float
    tail_min_lift: float
    tail_mean_lift: float
    tail_max_speed: float
    tail_max_angular_speed: float
    tail_contact_fraction: float
    tail_grasp_fraction: float
    tail_opposition_mean: float
    robot_table_contacts: int
    tail_robot_table_contact_fraction: float
    tail_robot_table_max: int
    max_penetration: float
    tail_max_penetration: float
    tail_robot_object_contact_mean: float
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
    hand_prefix = str(getattr(env.controller.hand_controller, "hand_prefix", "") or "")
    palm_geom = -1
    for candidate in (
        f"{hand_prefix}skin_palm_p" if hand_prefix else "",
        "skin_palm_p",
    ):
        if not candidate:
            continue
        palm_geom = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, candidate)
        if palm_geom >= 0:
            break
    if palm_geom < 0:
        matches = [
            geom_id
            for geom_id in range(env.model.ngeom)
            if (mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or "").endswith(
                "skin_palm_p"
            )
        ]
        if len(matches) == 1:
            palm_geom = int(matches[0])
    palm_body = int(env.model.geom_bodyid[palm_geom]) if palm_geom >= 0 else -1
    return object_ids, robot_ids, table_ids, palm_body


def _digit_from_names(geom_name: str, body_name: str) -> int:
    """Best-effort tactile/digit diagnostic classification.

    Grasp validity itself does not depend on this mapping.  That is deliberate:
    valid side-link contacts should not disappear just because their geom is not
    named ``skin_<digit>_*``.
    """
    text = f"{geom_name} {body_name}".lower()
    for digit in range(5):
        if f"skin_{digit}_" in text:
            return digit
    # Common descriptive aliases, only for diagnostics.
    aliases = {
        # The MJCF numbers the four fingers from the little-finger side
        # towards the thumb; normalized controller inputs reverse this order.
        0: ("little", "pinky", "finger_0", "finger0"),
        1: ("ring", "finger_1", "finger1"),
        2: ("middle", "finger_2", "finger2"),
        3: ("index", "finger_3", "finger3"),
        4: ("thumb",),
    }
    for digit, names in aliases.items():
        if any(name in text for name in names):
            return digit
    return -1


def _contact_snapshot(env, sets) -> tuple[int, int, bool, float, int, float]:
    """Return physical robot-object contact semantics for one C-MuJoCo frame.

    Opposition is computed from *all* robot-object contact normals, rather than
    only tactile skin geoms.  This makes side-link/palm-assisted grasps visible
    while still rejecting one-sided contact because the opposing-normal score
    remains small.
    """
    object_ids, robot_ids, table_ids, palm_body = sets
    digit_flags = [False] * 5
    palm_contact = False
    robot_object_normals: list[np.ndarray] = []
    robot_object_contacts = 0
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

        robot_object_contacts += 1
        norm = float(np.linalg.norm(normal))
        if norm > 1e-8:
            normal = normal / norm
            robot_object_normals.append(normal)

        geom_name = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_GEOM, robot_geom) or ""
        body_id = int(env.model.geom_bodyid[robot_geom])
        body_name = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_BODY, body_id) or ""
        lower = f"{geom_name} {body_name}".lower()
        if "palm" in lower or body_id == palm_body:
            palm_contact = True
        digit = _digit_from_names(geom_name, body_name)
        if 0 <= digit < 5:
            digit_flags[digit] = True

    opposition = 0.0
    for first_index in range(len(robot_object_normals)):
        for second_index in range(first_index + 1, len(robot_object_normals)):
            opposition = max(
                opposition,
                float(
                    np.clip(
                        -np.dot(
                            robot_object_normals[first_index],
                            robot_object_normals[second_index],
                        ),
                        0.0,
                        1.0,
                    )
                ),
            )

    return (
        robot_object_contacts,
        sum(digit_flags),
        palm_contact,
        opposition,
        robot_table,
        max_penetration,
    )


def load_replay_controls(path: str | Path, env):
    """Load controls and initial simulator state from an Ultra or PPO manifest."""
    path = _manifest_path(path)
    try:
        trajectory = GraspTrajectory.load(path)
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


def _resolved_settings(
    profile: str,
    *,
    success_lift_height: float | None,
    maximum_object_speed: float | None,
    maximum_object_angular_speed: float | None,
    opposition_threshold: float | None,
    verify_tail: int | None,
    extra_hold_steps: int | None,
    maximum_penetration: float | None,
    minimum_tail_contact_fraction: float | None,
    minimum_tail_grasp_fraction: float | None,
    maximum_tail_table_contact_fraction: float | None,
) -> dict[str, Any]:
    if profile not in _PROFILE_DEFAULTS:
        raise ValueError(
            f"Unknown strict replay profile {profile!r}; "
            f"choose {FINAL_PROFILE!r}."
        )
    defaults = dict(_PROFILE_DEFAULTS[profile])
    overrides = {
        "success_lift_height": success_lift_height,
        "maximum_object_speed": maximum_object_speed,
        "maximum_object_angular_speed": maximum_object_angular_speed,
        "opposition_threshold": opposition_threshold,
        "verify_tail": verify_tail,
        "extra_hold_steps": extra_hold_steps,
        "maximum_penetration": maximum_penetration,
        "minimum_tail_contact_fraction": minimum_tail_contact_fraction,
        "minimum_tail_grasp_fraction": minimum_tail_grasp_fraction,
        "maximum_tail_table_contact_fraction": maximum_tail_table_contact_fraction,
    }
    for key, value in overrides.items():
        if value is not None:
            defaults[key] = value
    if float(defaults["success_lift_height"]) <= 0.0:
        raise ValueError("success_lift_height must be positive.")
    if float(defaults["maximum_object_speed"]) <= 0.0:
        raise ValueError("maximum_object_speed must be positive.")
    if float(defaults["maximum_object_angular_speed"]) <= 0.0:
        raise ValueError("maximum_object_angular_speed must be positive.")
    if int(defaults["verify_tail"]) <= 0 or int(defaults["extra_hold_steps"]) < 0:
        raise ValueError("verify_tail must be positive and extra_hold_steps non-negative.")
    for key in (
        "opposition_threshold",
        "minimum_tail_contact_fraction",
        "minimum_tail_grasp_fraction",
        "maximum_tail_table_contact_fraction",
    ):
        if not 0.0 <= float(defaults[key]) <= 1.0:
            raise ValueError(f"{key} must lie in [0, 1].")
    return defaults


def strict_replay_manifest(
    trajectory_or_manifest: str | Path,
    *,
    render_mode: str | None = None,
    profile: str = FINAL_PROFILE,
    success_lift_height: float | None = None,
    maximum_object_speed: float | None = None,
    maximum_object_angular_speed: float | None = None,
    opposition_threshold: float | None = None,
    verify_tail: int | None = None,
    extra_hold_steps: int | None = None,
    maximum_penetration: float | None = None,
    minimum_tail_contact_fraction: float | None = None,
    minimum_tail_grasp_fraction: float | None = None,
    maximum_tail_table_contact_fraction: float | None = None,
    use_cache: bool = True,
) -> StrictReplayResult:
    manifest = _manifest_path(trajectory_or_manifest)
    payload = _payload(manifest)
    object_id = str(payload["object_id"])
    metadata = dict(payload.get("metadata", {}))
    control_dt = float(metadata.get("control_dt", 0.05))
    source_seed = int(metadata.get("source_seed", payload.get("seed", 0)))

    settings = _resolved_settings(
        profile,
        success_lift_height=success_lift_height,
        maximum_object_speed=maximum_object_speed,
        maximum_object_angular_speed=maximum_object_angular_speed,
        opposition_threshold=opposition_threshold,
        verify_tail=verify_tail,
        extra_hold_steps=extra_hold_steps,
        maximum_penetration=maximum_penetration,
        minimum_tail_contact_fraction=minimum_tail_contact_fraction,
        minimum_tail_grasp_fraction=minimum_tail_grasp_fraction,
        maximum_tail_table_contact_fraction=maximum_tail_table_contact_fraction,
    )
    lift_height = float(settings["success_lift_height"])
    speed_limit = float(settings["maximum_object_speed"])
    angular_speed_limit = float(settings["maximum_object_angular_speed"])
    opposition_limit = float(settings["opposition_threshold"])
    tail_size = int(settings["verify_tail"])
    hold_steps = int(settings["extra_hold_steps"])
    penetration_limit = float(settings["maximum_penetration"])
    contact_fraction_limit = float(settings["minimum_tail_contact_fraction"])
    grasp_fraction_limit = float(settings["minimum_tail_grasp_fraction"])
    table_fraction_limit = float(settings["maximum_tail_table_contact_fraction"])

    signature = _signature(manifest, {"profile": profile, **settings})
    cache = manifest.parent / f"strict_replay_{profile}.json"
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
        episode_length=int(payload.get("frames", 300)) + hold_steps + 20,
    )
    try:
        env.reset(seed=source_seed)
        controls, initial_qpos, initial_qvel, initial_ctrl = load_replay_controls(
            manifest, env
        )
        controls = np.asarray(controls, dtype=np.float32)
        if controls.ndim != 2 or controls.shape[1:] != env.action_space.shape:
            raise ValueError(
                f"Strict replay control shape {controls.shape} does not match "
                f"{env.action_space.shape}."
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
        angular_speeds: list[float] = []
        contacts: list[float] = []
        grasps: list[float] = []
        oppositions: list[float] = []
        robot_object_counts: list[float] = []
        table_counts: list[int] = []
        penetrations: list[float] = []
        tactile_values: list[float] = []
        robot_table_contacts = 0
        max_penetration_seen = 0.0

        sequence = list(controls)
        if len(controls):
            sequence.extend([controls[-1]] * hold_steps)

        for control in sequence:
            observation, _, terminated, truncated, _ = env.step(
                np.asarray(control, dtype=np.float32)
            )
            z = float(env.data.xpos[object_binding.body_id, 2])
            lift = z - initial_z
            speed = float(np.linalg.norm(env.data.qvel[qvel_adr : qvel_adr + 3]))
            angular_speed = float(
                np.linalg.norm(env.data.qvel[qvel_adr + 3 : qvel_adr + 6])
            )
            (
                robot_object_contacts,
                _digit_count,
                palm,
                opposition,
                table_contacts,
                penetration,
            ) = _contact_snapshot(env, sets)

            grasp_valid = (robot_object_contacts >= 2 and opposition >= opposition_limit) or (
                palm and robot_object_contacts >= 2
            )

            lifts.append(lift)
            speeds.append(speed)
            angular_speeds.append(angular_speed)
            contacts.append(float(robot_object_contacts > 0))
            grasps.append(float(grasp_valid))
            oppositions.append(opposition)
            robot_object_counts.append(float(robot_object_contacts))
            table_counts.append(int(table_contacts))
            penetrations.append(float(penetration))
            robot_table_contacts += int(table_contacts)
            max_penetration_seen = max(max_penetration_seen, float(penetration))

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

        tail_count = min(max(1, tail_size), len(lifts))
        tail_lifts = np.asarray(lifts[-tail_count:], dtype=np.float64)
        tail_speeds = np.asarray(speeds[-tail_count:], dtype=np.float64)
        tail_angular_speeds = np.asarray(
            angular_speeds[-tail_count:], dtype=np.float64
        )
        tail_contacts = np.asarray(contacts[-tail_count:], dtype=np.float64)
        tail_grasps = np.asarray(grasps[-tail_count:], dtype=np.float64)
        tail_oppositions = np.asarray(oppositions[-tail_count:], dtype=np.float64)
        tail_robot_objects = np.asarray(robot_object_counts[-tail_count:], dtype=np.float64)
        tail_tables = np.asarray(table_counts[-tail_count:], dtype=np.int32)
        tail_penetrations = np.asarray(penetrations[-tail_count:], dtype=np.float64)
        tail_tactile = np.asarray(tactile_values[-tail_count:], dtype=np.float64)

        final_lift = float(lifts[-1])
        max_lift = float(np.max(lifts))
        tail_min = float(np.min(tail_lifts))
        tail_mean = float(np.mean(tail_lifts))
        tail_speed_max = float(np.max(tail_speeds))
        tail_angular_speed_max = float(np.max(tail_angular_speeds))
        tail_contact_fraction = float(np.mean(tail_contacts))
        tail_grasp_fraction = float(np.mean(tail_grasps))
        tail_opposition_mean = float(np.mean(tail_oppositions))
        tail_robot_object_mean = float(np.mean(tail_robot_objects))
        tail_table_fraction = float(np.mean(tail_tables > 0))
        tail_table_max = int(np.max(tail_tables))
        tail_max_penetration = float(np.max(tail_penetrations))
        tail_tactile_max = float(np.max(tail_tactile))

        lift_scale = max(lift_height, 1e-6)
        angular_speed_ratio = tail_angular_speed_max / max(angular_speed_limit, 1e-6)
        quality = (
            2.0 * float(np.clip(tail_min / lift_scale, -1.0, 1.5))
            + 1.5 * float(np.clip(final_lift / lift_scale, -1.0, 1.5))
            + 0.25 * float(np.clip(max_lift / lift_scale, 0.0, 2.0))
            + 1.50 * tail_grasp_fraction
            + 0.75 * tail_opposition_mean
            + 0.25 * min(tail_robot_object_mean / 2.0, 1.0)
            - 0.75 * max(0.0, tail_speed_max / max(speed_limit, 1e-6) - 1.0)
            - 0.25 * min(angular_speed_ratio, 5.0)
            - 0.75 * max(0.0, angular_speed_ratio - 1.0)
            - 2.0 * tail_table_fraction
            - 50.0 * max(0.0, tail_max_penetration - penetration_limit)
        )

        success = bool(
            tail_min >= lift_height
            and final_lift >= lift_height
            and tail_speed_max <= speed_limit
            and tail_angular_speed_max <= angular_speed_limit
            and tail_contact_fraction >= contact_fraction_limit
            and tail_grasp_fraction >= grasp_fraction_limit
            and tail_table_fraction <= table_fraction_limit
            and tail_max_penetration <= penetration_limit
        )

        result = StrictReplayResult(
            success=success,
            profile=profile,
            verification_status=verification_status(profile, success),
            frames=len(lifts),
            final_lift=final_lift,
            max_lift=max_lift,
            tail_min_lift=tail_min,
            tail_mean_lift=tail_mean,
            tail_max_speed=tail_speed_max,
            tail_max_angular_speed=tail_angular_speed_max,
            tail_contact_fraction=tail_contact_fraction,
            tail_grasp_fraction=tail_grasp_fraction,
            tail_opposition_mean=tail_opposition_mean,
            robot_table_contacts=int(robot_table_contacts),
            tail_robot_table_contact_fraction=tail_table_fraction,
            tail_robot_table_max=tail_table_max,
            max_penetration=float(max_penetration_seen),
            tail_max_penetration=tail_max_penetration,
            tail_robot_object_contact_mean=tail_robot_object_mean,
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
                        "settings": {"profile": profile, **settings},
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
