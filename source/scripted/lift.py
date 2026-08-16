"""End-effector-independent lift policy using searched grasp configurations."""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
import trimesh

from source.geometry import mat_to_quat
from source.grasping.constants import (
    DEFAULT_GRIP_PRELOAD,
    GRASP_CONFIG_SCHEMA_VERSION,
    GRASP_SEARCH_STRATEGY,
    SUPPORTED_GRASP_CONFIG_SCHEMA_VERSIONS,
)
from source.scripted.base import ActionContext, PhaseContext, PhaseResult, TaskStrategy


def _quat_matrix(quaternion_wxyz: np.ndarray) -> np.ndarray:
    matrix = np.empty(9, dtype=np.float64)
    mujoco.mju_quat2Mat(matrix, np.asarray(quaternion_wxyz, dtype=np.float64))
    return matrix.reshape(3, 3)


def _mesh_symmetry_yaws_from_vertices(vertices: np.ndarray) -> tuple[float, ...]:
    """Return z rotations that preserve a centered, 12-cm object mesh.

    The thresholds tolerate scanned-mesh tessellation noise, but reject shape
    changes of several millimetres.  This makes the returned rotations safe
    grasp equivalents instead of arbitrary IK conveniences.
    """
    vertices = np.asarray(vertices, dtype=np.float64)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) < 4:
        return (0.0,)
    center = 0.5 * (vertices.min(axis=0) + vertices.max(axis=0))
    normalized = vertices - center
    normalized *= 0.12 / max(float(np.ptp(normalized, axis=0).max()), 1e-9)
    if len(normalized) > 12_000:
        indices = np.linspace(0, len(normalized) - 1, 12_000, dtype=np.int64)
        normalized = normalized[indices]
    tree = cKDTree(normalized)
    valid = [0.0]
    for yaw in np.linspace(0.0, 2.0 * np.pi, 16, endpoint=False)[1:]:
        cosine, sine = np.cos(yaw), np.sin(yaw)
        rotation = np.asarray(
            [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        distances = tree.query(normalized @ rotation.T, k=1)[0]
        if float(np.mean(distances)) <= 0.0025 and float(np.quantile(distances, 0.95)) <= 0.006:
            valid.append(float(yaw))
    return tuple(valid)


@lru_cache(maxsize=256)
def _object_symmetry_yaws(object_id: str) -> tuple[float, ...]:
    try:
        from source.grasping.search.catalog import resolve_object

        loaded = trimesh.load_mesh(resolve_object(object_id), process=True)
        mesh = loaded.to_geometry() if isinstance(loaded, trimesh.Scene) else loaded
        if not isinstance(mesh, trimesh.Trimesh):
            return (0.0,)
        return _mesh_symmetry_yaws_from_vertices(np.asarray(mesh.vertices))
    except Exception:
        # A missing optional source mesh must not prevent execution of a cached
        # grasp.  Zero yaw is always the exact configuration that was tested.
        return (0.0,)


@dataclass
class LiftStrategyState:
    lift_stable_steps: int = 0
    verify_success_steps: int = 0
    hold_wrist_position: np.ndarray | None = None
    strategy_verified_success: bool = False

    def reset(self) -> None:
        self.lift_stable_steps = 0
        self.verify_success_steps = 0
        self.hold_wrist_position = None
        self.strategy_verified_success = False


class LiftStrategy(TaskStrategy):
    """Approach along the searched path, grasp, then lift and verify."""

    phases = (
        "approach",
        "grasp",
        "lift",
        "verify",
    )
    PHASE_PROMPTS = {
        "approach": "1/4 APPROACH: follow the collision-free path to the grasp",
        "grasp": "2/4 GRASP: close and preload the fingers",
        "lift": "3/4 LIFT: raise the object",
        "verify": "4/4 VERIFY: hold and confirm a stable grasp",
    }

    LIFT_HEIGHT = 0.20
    # The position controller settles a few millimetres away from its internal
    # IK solution under gravity and hand inertia. These thresholds still keep
    # the wrist inside the searched local corridor while avoiding deadlocks at
    # physically converged waypoints (observed residuals: 9-16 mm, 3-5 deg).
    WAYPOINT_POSITION_TOLERANCE = 0.018
    ORIENTATION_TOLERANCE = np.deg2rad(5.0)
    CHECK_MAX_DISTANCE = 0.06
    MIN_PHASE_STEPS = 10
    GRASP_STEPS = 130
    PHASE_TIMEOUT = 400
    GRASP_QUATERNION = np.asarray([0.0, 1.0, 0.0, 0.0], dtype=np.float64)

    # Fractions of each actuator's physical range. Current-project order is
    # finger0..3, thumb_rotate, thumb_grasp.
    # Stable standalone grasp generated for ycb:002_master_chef_can. The pose
    # maps points from the Dex Hand root frame into the object frame.
    TEMPLATE_HAND_FRACTIONS = np.asarray(
        [0.1857142857, 0.1857142857, 0.1857142857, 0.1857142857, 1.0, 0.1857142857],
        dtype=np.float64,
    )
    TEMPLATE_HAND_TRANSLATION = np.asarray(
        [0.0474914941, -0.1880781858, -0.0005670715],
        dtype=np.float64,
    )
    TEMPLATE_HAND_ROTATION = np.eye(3, dtype=np.float64)
    HAND_ATTACH_ROTATION = Rotation.from_euler("xyz", [-90.0, -90.0, 0.0], degrees=True).as_matrix()
    GRIP_PRELOAD = DEFAULT_GRIP_PRELOAD
    phase_position_speeds = {
        "approach": 0.16,
        "grasp": 0.05,
        "lift": 0.10,
    }
    phase_orientation_speeds = {
        "approach": 1.00,
        "grasp": 0.40,
        "lift": 0.50,
    }

    def __init__(
        self,
        *,
        max_position_step: float = 0.03,
        reuse_grasp_config: bool = False,
        grasp_search_options: dict | None = None,
        grasp_config_path: str | Path | None = None,
        grasp_candidate_index: int | None = None,
    ) -> None:
        super().__init__(max_position_step=max_position_step, max_orientation_step=0.20)
        self.state = LiftStrategyState()
        self.reuse_grasp_config = bool(reuse_grasp_config)
        self.grasp_config_override = None if grasp_config_path is None else Path(grasp_config_path)
        if grasp_candidate_index is not None and grasp_candidate_index < 0:
            raise ValueError("grasp_candidate_index must be non-negative.")
        self.forced_candidate_index = grasp_candidate_index
        self.grasp_search_options = dict(grasp_search_options or {})
        reserved_options = {"object_id", "output", "end_effector_name"}
        invalid_options = reserved_options.intersection(self.grasp_search_options)
        if invalid_options:
            raise ValueError(f"grasp_search_options cannot override {sorted(invalid_options)}.")
        self.template_hand_fractions = self.TEMPLATE_HAND_FRACTIONS.copy()
        self.template_hand_translation = self.TEMPLATE_HAND_TRANSLATION.copy()
        self.template_hand_rotation = self.TEMPLATE_HAND_ROTATION.copy()
        self.template_preload_weights = np.asarray(
            [1.0, 1.0, 1.0, 1.0, 0.0, 1.0],
            dtype=np.float64,
        )
        self.template_preload_directions = np.ones(6, dtype=np.float64)
        self.end_effector_name = "dex_hand"
        self.hand_attach_rotation = self.HAND_ATTACH_ROTATION.copy()
        self.approach_hand_translations = np.empty((0, 3), dtype=np.float64)
        self.approach_hand_rotations = np.empty((0, 3, 3), dtype=np.float64)
        self.approach_hand_fractions = np.empty((0, 6), dtype=np.float64)
        self.grasp_hand_translations = np.empty((0, 3), dtype=np.float64)
        self.grasp_hand_rotations = np.empty((0, 3, 3), dtype=np.float64)
        self.grasp_hand_fractions = np.empty((0, 6), dtype=np.float64)
        self.grasp_template_path: Path | None = None
        self.grasp_template_object_id: str | None = None
        self.archive_candidate_index = grasp_candidate_index or 0
        self.restart_count = 0
        self.last_failed_phase: str | None = None
        self.last_grasp_hand_error = 0.0
        self.last_contact_fingers: tuple[int, ...] = ()
        self.last_wrist_position_error = 0.0
        self.last_wrist_orientation_error = 0.0

    def reset(self) -> None:
        super().reset()
        self.state.reset()
        # Randomized episodes may require a different member of the stable
        # grasp archive, so force reachability selection for the new pose.
        self.grasp_template_object_id = None
        self.archive_candidate_index = self.forced_candidate_index or 0
        self.restart_count = 0
        self.last_failed_phase = None
        self.last_grasp_hand_error = 0.0
        self.last_contact_fingers = ()
        self.last_wrist_position_error = 0.0
        self.last_wrist_orientation_error = 0.0

    def _advance_grasp_candidate(self) -> None:
        """Try the next reachability-ranked stable grasp after a phase failure."""
        self.restart_count += 1
        self.last_failed_phase = self.phase_name
        if self.forced_candidate_index is None:
            self.archive_candidate_index += 1
        self.grasp_template_object_id = None

    @staticmethod
    def _robot_table_collision(env) -> bool:
        """Return whether any robot geom currently touches the table or floor."""
        bindings = env.task._require_bindings()
        object_geoms = set().union(*(binding.geom_ids for binding in bindings.objects.values()))
        table_geoms = bindings.environment_geom_ids.difference(object_geoms)
        for index in range(env.data.ncon):
            geom1 = int(env.data.contact[index].geom1)
            geom2 = int(env.data.contact[index].geom2)
            if (
                geom1 in bindings.robot_geom_ids
                and geom2 in table_geoms
                or geom2 in bindings.robot_geom_ids
                and geom1 in table_geoms
            ):
                return True
        return False

    @staticmethod
    def _grasp_config_name(object_id: str) -> str:
        return "".join(
            character if character.isalnum() or character in "-_" else "_"
            for character in object_id
        )

    @staticmethod
    def _select_reachable_payload(
        env,
        payload: dict,
        object_position: np.ndarray,
        object_quaternion: np.ndarray,
        hand_attach_rotation: np.ndarray,
        candidate_index: int = 0,
    ) -> dict:
        """Select a trajectory-validated grasp using arm IK residual."""
        candidates = payload.get("lift_verified_candidates")
        if not isinstance(candidates, list) or not candidates:
            candidates = payload.get("trajectory_stable_candidates")
        if not isinstance(candidates, list) or len(candidates) < 2:
            return payload
        arm = env.controller.arm_controller
        saved_qpos = env.data.qpos.copy()
        saved_qvel = env.data.qvel.copy()
        saved_ctrl = env.data.ctrl.copy()
        previous_target_q = None if arm._prev_target_q is None else arm._prev_target_q.copy()
        previous_filtered_velocity = (
            None if arm._filtered_velocity is None else arm._filtered_velocity.copy()
        )
        object_rotation = _quat_matrix(object_quaternion)
        ranked: list[tuple[float, dict]] = []
        try:
            for candidate in candidates:
                try:
                    final_rotation = np.asarray(candidate["hand_rotation_matrix"], dtype=np.float64)
                    final_translation = np.asarray(candidate["hand_translation"], dtype=np.float64)
                    approach_translation = np.asarray(
                        candidate["approach_hand_translations"][0], dtype=np.float64
                    )
                    approach_rotation = np.asarray(
                        candidate["approach_hand_rotation_matrices"][0], dtype=np.float64
                    )
                    cached_final_rotation = np.asarray(
                        candidate.get("grasp_hand_rotation_matrices", [final_rotation])[-1],
                        dtype=np.float64,
                    )
                except (KeyError, IndexError, TypeError, ValueError):
                    continue
                approach_rotation = final_rotation @ cached_final_rotation.T @ approach_rotation
                residual = 0.0
                for index, (local_position, local_rotation) in enumerate(
                    ((approach_translation, approach_rotation), (final_translation, final_rotation))
                ):
                    target_position = object_position + object_rotation @ local_position
                    target_quaternion = mat_to_quat(
                        object_rotation @ local_rotation @ hand_attach_rotation.T
                    )
                    arm_qpos = arm._solve_ik(
                        env.model, env.data, target_position, target_quaternion
                    )
                    env.data.qpos[arm.qpos_addrs] = arm_qpos
                    mujoco.mj_forward(env.model, env.data)
                    actual_position = env.data.site_xpos[arm.site_id]
                    actual_quaternion = mat_to_quat(env.data.site_xmat[arm.site_id])
                    position_error = float(np.linalg.norm(actual_position - target_position))
                    orientation_error = 2.0 * np.arccos(
                        np.clip(abs(float(np.dot(actual_quaternion, target_quaternion))), 0.0, 1.0)
                    )
                    # Final contact pose matters more than the first approach
                    # point, which can later be reconnected by the planner.
                    residual += (1.0 if index == 0 else 2.0) * (
                        position_error + 0.08 * orientation_error
                    )
                    if LiftStrategy._robot_table_collision(env):
                        residual += 100.0
                ranked.append((residual, candidate))
                env.data.qpos[:] = saved_qpos
                mujoco.mj_forward(env.model, env.data)
        finally:
            arm._prev_target_q = previous_target_q
            arm._filtered_velocity = previous_filtered_velocity
            env.data.qpos[:] = saved_qpos
            env.data.qvel[:] = saved_qvel
            env.data.ctrl[:] = saved_ctrl
            mujoco.mj_forward(env.model, env.data)
        if not ranked:
            return payload
        ranked.sort(key=lambda item: item[0])
        return ranked[candidate_index % len(ranked)][1]

    def _ensure_grasp_template(
        self,
        env,
        object_position: np.ndarray,
        object_quaternion: np.ndarray,
    ) -> None:
        object_id = getattr(env.task, "object_id", None)
        if not isinstance(object_id, str) or not object_id:
            raise RuntimeError("Lift task does not expose a valid object_id.")
        if (
            self.grasp_template_object_id == object_id
            and self.end_effector_name == env.hand_descriptor.name
        ):
            return

        end_effector_name = env.hand_descriptor.name
        from source.grasping.search.catalog import grasp_config_directory

        config_dir = grasp_config_directory(end_effector_name)
        path = (
            self.grasp_config_override
            if self.grasp_config_override is not None
            else config_dir / f"{self._grasp_config_name(object_id)}.json"
        )
        should_generate = self.grasp_config_override is None and (
            not self.reuse_grasp_config or not path.is_file()
        )
        if self.grasp_config_override is not None and not path.is_file():
            raise FileNotFoundError(f"Explicit grasp config does not exist: {path}")
        if should_generate:
            # Import lazily so normal cached-policy startup does not pay the
            # visualization import cost of the search demo.
            from source.grasping.search.api import generate_validated_grasp_config

            reason = "default fresh search" if path.is_file() else "no cached config"
            print(
                f"Generating grasp with the production two-stage search for {object_id} "
                f"({reason}) -> {path} ..."
            )
            result = generate_validated_grasp_config(
                object_id,
                output=path,
                end_effector_name=end_effector_name,
                **self.grasp_search_options,
            )
            path = result.output_path
            print(
                f"Generated validated grasp config: {path} "
                f"(seed={result.selected_seed}, attempts={result.attempts_used})"
            )

        payload = json.loads(path.read_text(encoding="utf-8"))
        attach_degrees = (
            env.arm_descriptor.hand_attach_rot_xyz_deg
            if env.config.hand_attach_rot_xyz_deg is None
            else tuple(env.config.hand_attach_rot_xyz_deg)
        )
        hand_attach_rotation = Rotation.from_euler("xyz", attach_degrees, degrees=True).as_matrix()
        payload = self._select_reachable_payload(
            env,
            payload,
            object_position,
            object_quaternion,
            hand_attach_rotation,
            self.archive_candidate_index,
        )
        schema_version = payload.get("schema_version")
        if schema_version not in SUPPORTED_GRASP_CONFIG_SCHEMA_VERSIONS:
            raise ValueError(f"Unsupported or missing schema_version in {path}.")
        if (
            schema_version == GRASP_CONFIG_SCHEMA_VERSION
            and payload.get("search_strategy") != GRASP_SEARCH_STRATEGY
        ):
            raise ValueError(
                f"Grasp {path} uses stale strategy "
                f"{payload.get('search_strategy')!r}; expected "
                f"{GRASP_SEARCH_STRATEGY!r}. Regenerate it without "
                "--reuse-grasp-config."
            )
        payload_object_id = payload.get("object_id")
        if payload_object_id != object_id:
            raise ValueError(
                f"Grasp {path} belongs to {payload_object_id!r}, "
                f"not the active object {object_id!r}."
            )
        if payload.get("trajectory_hold_stable") is not True or payload.get(
            "validation_stage"
        ) not in {"trajectory_hold_stable", "robot_lift_verified"}:
            raise ValueError(
                f"Grasp {path} has not passed executable approach/closure/hold "
                "validation. Regenerate it with the current benchmark pipeline."
            )
        payload_end_effector = payload.get("end_effector_name", "dex_hand")
        if payload_end_effector != end_effector_name:
            raise ValueError(
                f"Grasp {path} belongs to {payload_end_effector!r}, "
                f"not active end effector {end_effector_name!r}."
            )

        fractions = np.asarray(payload["hand_actuator_fractions"], dtype=np.float64)
        translation = np.asarray(payload["hand_translation"], dtype=np.float64)
        rotation = np.asarray(payload["hand_rotation_matrix"], dtype=np.float64)
        preload_weights = np.asarray(
            payload.get("hand_preload_weights", []),
            dtype=np.float64,
        )
        preload_directions = np.asarray(
            payload.get(
                "hand_preload_directions",
                np.ones_like(preload_weights),
            ),
            dtype=np.float64,
        )
        approach_translations = np.asarray(
            payload.get("approach_hand_translations", []),
            dtype=np.float64,
        )
        approach_rotations = np.asarray(
            payload.get("approach_hand_rotation_matrices", []),
            dtype=np.float64,
        )
        approach_fractions = np.asarray(
            payload.get("approach_hand_actuator_fractions", []),
            dtype=np.float64,
        )
        grasp_translations = np.asarray(
            payload.get("grasp_hand_translations", [translation]),
            dtype=np.float64,
        )
        grasp_rotations = np.asarray(
            payload.get("grasp_hand_rotation_matrices", [rotation]),
            dtype=np.float64,
        )
        grasp_fractions = np.asarray(
            payload.get("grasp_hand_actuator_fractions", [fractions]),
            dtype=np.float64,
        )
        # Configs produced before trajectory-aware evolution kept the seed
        # trajectory while mutating the final rotation and finger state. Repair
        # those cached configs in memory so the executed final waypoint matches
        # the state that passed standalone dynamics validation.
        rotation_delta = rotation @ grasp_rotations[-1].T
        approach_rotations = np.einsum("ij,njk->nik", rotation_delta, approach_rotations)
        grasp_rotations = np.einsum("ij,njk->nik", rotation_delta, grasp_rotations)
        fraction_delta = fractions - grasp_fractions[-1]
        grasp_progress = (
            np.ones((1, 1), dtype=np.float64)
            if len(grasp_fractions) == 1
            else np.linspace(0.0, 1.0, len(grasp_fractions))[:, None]
        )
        grasp_fractions = np.clip(
            grasp_fractions + grasp_progress * fraction_delta,
            0.0,
            1.0,
        )
        actuator_count = env.controller.hand_controller.action_size
        if fractions.shape != (actuator_count,) or np.any((fractions < 0.0) | (fractions > 1.0)):
            raise ValueError(f"Invalid hand_actuator_fractions in {path}.")
        if translation.shape != (3,) or not np.all(np.isfinite(translation)):
            raise ValueError(f"Invalid hand_translation in {path}.")
        if rotation.shape != (3, 3) or not np.all(np.isfinite(rotation)):
            raise ValueError(f"Invalid hand_rotation_matrix in {path}.")
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-4):
            raise ValueError(f"Non-orthonormal hand_rotation_matrix in {path}.")
        if preload_weights.shape != (actuator_count,) or np.any(
            (preload_weights < 0.0) | (preload_weights > 1.0)
        ):
            raise ValueError(f"Invalid hand_preload_weights in {path}.")
        if preload_directions.shape != (actuator_count,) or np.any(
            ~np.isin(preload_directions, (-1.0, 1.0))
        ):
            raise ValueError(f"Invalid hand_preload_directions in {path}.")
        waypoint_count = approach_translations.shape[0]
        if (
            waypoint_count < 2
            or approach_translations.shape != (waypoint_count, 3)
            or approach_rotations.shape != (waypoint_count, 3, 3)
            or approach_fractions.shape != (waypoint_count, actuator_count)
            or not np.all(np.isfinite(approach_translations))
            or not np.all(np.isfinite(approach_rotations))
            or np.any((approach_fractions < 0.0) | (approach_fractions > 1.0))
        ):
            raise ValueError(
                f"Grasp {path} has no valid point-cloud approach waypoints. "
                "Regenerate it with search_mesh_force_closure."
            )
        if not np.allclose(
            approach_rotations.transpose(0, 2, 1) @ approach_rotations,
            np.eye(3)[None, :, :],
            atol=1e-4,
        ):
            raise ValueError(f"Invalid approach rotations in {path}.")
        grasp_waypoint_count = grasp_translations.shape[0]
        if (
            grasp_waypoint_count < 1
            or grasp_translations.shape != (grasp_waypoint_count, 3)
            or grasp_rotations.shape != (grasp_waypoint_count, 3, 3)
            or grasp_fractions.shape != (grasp_waypoint_count, actuator_count)
            or not np.all(np.isfinite(grasp_translations))
            or not np.all(np.isfinite(grasp_rotations))
            or np.any((grasp_fractions < 0.0) | (grasp_fractions > 1.0))
        ):
            raise ValueError(f"Grasp {path} has no valid closing trajectory.")
        if not np.allclose(
            grasp_rotations.transpose(0, 2, 1) @ grasp_rotations,
            np.eye(3)[None, :, :],
            atol=1e-4,
        ):
            raise ValueError(f"Invalid grasp trajectory rotations in {path}.")
        self.template_hand_fractions = fractions
        self.template_hand_translation = translation
        self.template_hand_rotation = rotation
        self.template_preload_weights = preload_weights
        self.template_preload_directions = preload_directions
        self.approach_hand_translations = approach_translations
        self.approach_hand_rotations = approach_rotations
        self.approach_hand_fractions = approach_fractions
        self.grasp_hand_translations = grasp_translations
        self.grasp_hand_rotations = grasp_rotations
        self.grasp_hand_fractions = grasp_fractions
        self.grasp_template_path = path
        self.grasp_template_object_id = object_id
        self.end_effector_name = end_effector_name
        self.hand_attach_rotation = hand_attach_rotation

    @property
    def phase_prompt(self) -> str:
        if self.finished:
            return "3/3 VERIFIED: object is lifted and remains inside the hand"
        return self.PHASE_PROMPTS[self.phase_name]

    @staticmethod
    def _hand_target(env, fractions: np.ndarray) -> np.ndarray:
        hand = env.controller.hand_controller
        count = hand.action_size
        fractions = np.asarray(fractions, dtype=np.float64)
        if fractions.shape != (count,):
            raise ValueError(f"Expected {count} end-effector fractions, got {fractions.shape}.")
        low = np.asarray(env.action_space.low[-count:], dtype=np.float64)
        high = np.asarray(env.action_space.high[-count:], dtype=np.float64)
        return low + fractions * (high - low)

    @staticmethod
    def _hand_qpos(env) -> np.ndarray:
        hand = env.controller.hand_controller
        joint_positions = np.asarray(env.data.qpos[hand.qpos_addrs], dtype=np.float64)
        if env.hand_descriptor.name == "pika_gripper":
            return np.asarray(
                hand._joint_target_to_opening(joint_positions),
                dtype=np.float64,
            )
        return joint_positions

    @staticmethod
    def _distal_centers(env) -> tuple[np.ndarray, np.ndarray]:
        """Return middle-finger and thumb distal tactile-patch centers."""
        prefix = env.controller.hand_controller.hand_prefix
        groups = (f"{prefix}taxel_skin_2_2_p_", f"{prefix}taxel_skin_4_2_p_")
        centers: list[np.ndarray] = []
        for group in groups:
            ids = [
                site_id
                for site_id in range(env.model.nsite)
                if (
                    mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_SITE, site_id) or ""
                ).startswith(group)
            ]
            if not ids:
                raise RuntimeError(f"No Dex Hand distal tactile sites match {group!r}.")
            centers.append(np.mean(env.data.site_xpos[ids], axis=0))
        return centers[0], centers[1]

    @classmethod
    def _grasp_midpoint(cls, env) -> np.ndarray:
        """Midpoint of middle-finger and thumb distal tactile patches."""
        if env.hand_descriptor.name == "pika_gripper":
            prefix = env.controller.hand_controller.hand_prefix
            ids = [
                site_id
                for site_id in range(env.model.nsite)
                if (
                    mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_SITE, site_id) or ""
                ).startswith(f"{prefix}taxel_pika_")
            ]
            if ids:
                return np.mean(env.data.site_xpos[ids], axis=0)
            site_name = f"{prefix}gripper_tcp"
            site_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_SITE, site_name)
            if site_id < 0:
                raise RuntimeError(f"Pika grasp midpoint site {site_name!r} missing.")
            return env.data.site_xpos[site_id].copy()
        finger, thumb = cls._distal_centers(env)
        return 0.5 * (finger + thumb)

    def grasp_midpoint(self, env) -> np.ndarray:
        """Expose the measured grasp midpoint for interactive visualization."""
        return self._grasp_midpoint(env)

    @staticmethod
    def _stable(
        context: PhaseContext,
        key: str,
        condition: bool,
        *,
        required_steps: int = 5,
    ) -> bool:
        """Require a condition to remain true for consecutive control steps."""
        count = int(context.memory.get(key, 0)) + 1 if condition else 0
        context.memory[key] = count
        return count >= required_steps

    @staticmethod
    def _contact_fingers(env) -> tuple[int, ...]:
        object_geoms = env.task._require_bindings().objects["object"].geom_ids
        fingers: set[int] = set()
        for index in range(env.data.ncon):
            contact = env.data.contact[index]
            geom1, geom2 = int(contact.geom1), int(contact.geom2)
            if geom1 in object_geoms:
                robot_geom = geom2
            elif geom2 in object_geoms:
                robot_geom = geom1
            else:
                continue
            name = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_GEOM, robot_geom) or ""
            if "gripper_left_link" in name:
                fingers.add(0)
            if "gripper_right_link" in name:
                fingers.add(1)
            for finger in range(5):
                if (
                    f"skin_{finger}_" in name
                    or f"finger_first_{finger}" in name
                    or f"finger_second_{finger}" in name
                ):
                    fingers.add(finger)
            if "thumb_" in name:
                fingers.add(4)
        return tuple(sorted(fingers))

    @staticmethod
    def _select_grasp_quaternion(
        object_quaternion: np.ndarray,
        midpoint_target: np.ndarray,
        offset_local: np.ndarray,
        finger_axis_local: np.ndarray,
        current_ee_position: np.ndarray,
    ) -> np.ndarray:
        """Construct a palm-down pose and align its closing axis to the object."""
        object_rotation = _quat_matrix(object_quaternion)
        object_axes = (object_rotation[:, 0].copy(), object_rotation[:, 1].copy())
        for axis in object_axes:
            axis[2] = 0.0
            axis /= np.linalg.norm(axis)

        finger_axis_local = np.asarray(finger_axis_local, dtype=np.float64)
        finger_axis_local /= np.linalg.norm(finger_axis_local)
        palm_normal_local = np.asarray([0.0, 0.0, -1.0], dtype=np.float64)
        palm_normal_local -= np.dot(palm_normal_local, finger_axis_local) * finger_axis_local
        palm_normal_local /= np.linalg.norm(palm_normal_local)
        hand_side_local = np.cross(finger_axis_local, palm_normal_local)
        hand_side_local /= np.linalg.norm(hand_side_local)
        source_basis = np.column_stack([finger_axis_local, palm_normal_local, hand_side_local])
        target_palm_normal = np.asarray([0.0, 0.0, -1.0], dtype=np.float64)

        best_score = float("inf")
        best_quaternion = LiftStrategy.GRASP_QUATERNION.copy()
        for yaw in np.linspace(-np.pi, np.pi, 73, endpoint=False):
            target_finger_axis = np.asarray(
                [np.cos(yaw), np.sin(yaw), 0.0],
                dtype=np.float64,
            )
            target_hand_side = np.cross(target_finger_axis, target_palm_normal)
            target_hand_side /= np.linalg.norm(target_hand_side)
            target_finger_axis = np.cross(
                target_palm_normal,
                target_hand_side,
            )
            target_basis = np.column_stack(
                [target_finger_axis, target_palm_normal, target_hand_side]
            )
            candidate_rotation = target_basis @ source_basis.T
            candidate = np.empty(4, dtype=np.float64)
            mujoco.mju_mat2Quat(candidate, candidate_rotation.reshape(9))
            wrist = LiftStrategy._wrist_from_midpoint(
                midpoint_target,
                candidate,
                offset_local,
            )
            closing_axis = -(candidate_rotation @ finger_axis_local)
            closing_axis[2] = 0.0
            closing_axis /= np.linalg.norm(closing_axis)
            alignment = max(
                abs(float(np.dot(closing_axis, object_axes[0]))),
                abs(float(np.dot(closing_axis, object_axes[1]))),
            )
            score = 5.0 * (1.0 - alignment) + float(np.linalg.norm(wrist - current_ee_position))
            if score < best_score:
                best_score = score
                best_quaternion = candidate
        return best_quaternion

    @staticmethod
    def _wrist_from_midpoint(
        target_midpoint: np.ndarray,
        target_quaternion: np.ndarray,
        offset_local: np.ndarray,
    ) -> np.ndarray:
        return target_midpoint - _quat_matrix(target_quaternion) @ offset_local

    def _template_wrist_pose(
        self,
        object_position: np.ndarray,
        object_quaternion: np.ndarray,
        yaw: float = 0.0,
        tool_roll: float = 0.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        object_rotation = _quat_matrix(object_quaternion)
        symmetry_rotation = Rotation.from_euler("z", yaw).as_matrix()
        tool_symmetry = Rotation.from_euler("x", tool_roll).as_matrix()
        pivot = (
            np.asarray([0.16, 0.0, 0.0064182], dtype=np.float64)
            if self.end_effector_name == "pika_gripper"
            else np.zeros(3, dtype=np.float64)
        )
        local_translation = (
            self.template_hand_translation
            + self.template_hand_rotation @ pivot
            - self.template_hand_rotation @ tool_symmetry @ pivot
        )
        hand_position = object_position + object_rotation @ symmetry_rotation @ local_translation
        hand_rotation = (
            object_rotation @ symmetry_rotation @ self.template_hand_rotation @ tool_symmetry
        )
        ee_rotation = hand_rotation @ self.hand_attach_rotation.T
        return hand_position, mat_to_quat(ee_rotation)

    def _select_reachable_template_pose(
        self,
        env,
        object_position: np.ndarray,
        object_quaternion: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
        """Evaluate the configured object-relative grasp with actual IK residual.

        A grasp configuration is tied to a specific object mesh. Rotating it
        around the object to improve reachability is only valid for a proven
        rotational symmetry; doing so for boxes and EGAD shapes destroys the
        contact geometry that passed standalone validation. The object's
        randomized world yaw is already included in ``object_quaternion``.
        """
        arm = env.controller.arm_controller
        saved_qpos = env.data.qpos.copy()
        saved_qvel = env.data.qvel.copy()
        saved_ctrl = env.data.ctrl.copy()
        previous_velocity = arm.max_joint_velocity
        previous_filter = arm.velocity_filter_alpha
        previous_target_q = None if arm._prev_target_q is None else arm._prev_target_q.copy()
        previous_filtered_velocity = (
            None if arm._filtered_velocity is None else arm._filtered_velocity.copy()
        )
        arm.max_joint_velocity = 100.0
        arm.velocity_filter_alpha = 1.0
        best = None
        try:
            # Pika's two-finger tool has a known 180-degree roll symmetry. An
            # object-relative yaw is considered only when its actual mesh
            # passes the geometric symmetry check above.
            rolls = (0.0, np.pi) if self.end_effector_name == "pika_gripper" else (0.0,)
            yaws = _object_symmetry_yaws(self.grasp_template_object_id or "")
            for yaw in yaws:
                for tool_roll in rolls:
                    grasp_position, quaternion = self._template_wrist_pose(
                        object_position,
                        object_quaternion,
                        yaw,
                        tool_roll,
                    )
                    approach_positions, approach_quaternions = self._world_approach_waypoints(
                        object_position,
                        object_quaternion,
                        yaw,
                        tool_roll,
                    )
                    residual = 0.0
                    grasp_arm_qpos = None
                    for position, waypoint_quaternion in (
                        (approach_positions[0], approach_quaternions[0]),
                        (grasp_position, quaternion),
                    ):
                        arm_qpos = arm._solve_ik(
                            env.model,
                            env.data,
                            position,
                            waypoint_quaternion,
                        )
                        env.data.qpos[arm.qpos_addrs] = arm_qpos
                        mujoco.mj_forward(env.model, env.data)
                        actual_position = env.data.site_xpos[arm.site_id]
                        actual_quaternion = mat_to_quat(env.data.site_xmat[arm.site_id])
                        position_error = np.linalg.norm(actual_position - position)
                        orientation_error = 2.0 * np.arccos(
                            np.clip(
                                abs(float(np.dot(actual_quaternion, waypoint_quaternion))),
                                0.0,
                                1.0,
                            )
                        )
                        residual += float(position_error + 0.08 * orientation_error)
                        if self._robot_table_collision(env):
                            residual += 100.0
                        grasp_arm_qpos = arm_qpos.copy()
                    env.data.qpos[:] = saved_qpos
                    mujoco.mj_forward(env.model, env.data)
                    if best is None or residual < best[0]:
                        best = (
                            residual,
                            grasp_position,
                            quaternion,
                            grasp_arm_qpos,
                            yaw,
                            tool_roll,
                        )
        finally:
            arm.max_joint_velocity = previous_velocity
            arm.velocity_filter_alpha = previous_filter
            arm._prev_target_q = previous_target_q
            arm._filtered_velocity = previous_filtered_velocity
            env.data.qpos[:] = saved_qpos
            env.data.qvel[:] = saved_qvel
            env.data.ctrl[:] = saved_ctrl
            mujoco.mj_forward(env.model, env.data)
        if best is None:
            raise RuntimeError("No symmetry-equivalent template pose was evaluated.")
        return best[1], best[2], best[3], best[4], best[5]

    def _world_hand_waypoints(
        self,
        local_translations: np.ndarray,
        local_rotations: np.ndarray,
        object_position: np.ndarray,
        object_quaternion: np.ndarray,
        yaw: float,
        tool_roll: float = 0.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        object_rotation = _quat_matrix(object_quaternion)
        symmetry_rotation = Rotation.from_euler("z", yaw).as_matrix()
        tool_symmetry = Rotation.from_euler("x", tool_roll).as_matrix()
        pivot = (
            np.asarray([0.16, 0.0, 0.0064182], dtype=np.float64)
            if self.end_effector_name == "pika_gripper"
            else np.zeros(3, dtype=np.float64)
        )
        local_positions = (
            local_translations
            + (local_rotations @ pivot)
            - (local_rotations @ tool_symmetry @ pivot)
        )
        positions = (
            object_position[None, :] + (object_rotation @ symmetry_rotation @ local_positions.T).T
        )
        quaternions = []
        for hand_rotation_local in local_rotations:
            hand_rotation = (
                object_rotation @ symmetry_rotation @ hand_rotation_local @ tool_symmetry
            )
            quaternions.append(mat_to_quat(hand_rotation @ self.hand_attach_rotation.T))
        return positions, np.asarray(quaternions)

    def _world_approach_waypoints(
        self,
        object_position: np.ndarray,
        object_quaternion: np.ndarray,
        yaw: float,
        tool_roll: float = 0.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        return self._world_hand_waypoints(
            self.approach_hand_translations,
            self.approach_hand_rotations,
            object_position,
            object_quaternion,
            yaw,
            tool_roll,
        )

    def _world_grasp_waypoints(
        self,
        object_position: np.ndarray,
        object_quaternion: np.ndarray,
        yaw: float,
        tool_roll: float = 0.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        return self._world_hand_waypoints(
            self.grasp_hand_translations,
            self.grasp_hand_rotations,
            object_position,
            object_quaternion,
            yaw,
            tool_roll,
        )

    def execute_phase(
        self, phase_index: int, context: PhaseContext
    ) -> tuple[PhaseResult, ActionContext]:
        phase = self.phases[phase_index]
        env = context.env
        current = env.controller.current_ik_action(env.model, env.data).astype(np.float64)
        ee_position = current[:3]
        object_position = np.asarray(context.observation["object_pos"], dtype=np.float64)
        object_quaternion = np.asarray(context.observation["object_quat"], dtype=np.float64)
        self._ensure_grasp_template(env, object_position, object_quaternion)
        gripper_fractions = np.asarray(
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0]
            if self.end_effector_name == "dex_hand"
            else np.ones_like(self.template_hand_fractions),
            dtype=np.float64,
        )
        gripper = self._hand_target(env, gripper_fractions)
        grasp = self._hand_target(env, self.template_hand_fractions)
        preload_fractions = self.template_hand_fractions.copy()
        preload_endpoints = np.where(
            self.template_preload_directions > 0.0,
            1.0,
            0.0,
        )
        preload_fractions += (
            self.GRIP_PRELOAD
            * self.template_preload_weights
            * (preload_endpoints - preload_fractions)
        )
        preload = self._hand_target(env, preload_fractions)

        if phase == "approach" and "target_quaternion" not in context.memory:
            open_error = float(np.max(np.abs(self._hand_qpos(env) - gripper)))
            open_ready = context.phase_step >= self.MIN_PHASE_STEPS and self._stable(
                context,
                "open_hand_stable_steps",
                open_error < 0.0007,
            )
            if not open_ready:
                return PhaseResult.CONTINUE, ActionContext(hand_target=gripper)
            (
                grasp_wrist,
                target_quaternion,
                grasp_arm_qpos,
                template_yaw,
                template_tool_roll,
            ) = self._select_reachable_template_pose(env, object_position, object_quaternion)
            approach_positions, approach_quaternions = self._world_approach_waypoints(
                object_position,
                object_quaternion,
                template_yaw,
                template_tool_roll,
            )
            grasp_positions, grasp_quaternions = self._world_grasp_waypoints(
                object_position,
                object_quaternion,
                template_yaw,
                template_tool_roll,
            )
            grasp_wrist = grasp_positions[-1]
            target_quaternion = grasp_quaternions[-1]
            approach_wrist = approach_positions[0]
            contact_wrist = grasp_positions[0]
            engaged_wrist = grasp_positions[0]
            context.memory["target_quaternion"] = approach_quaternions[0]
            context.memory["grasp_target_quaternion"] = target_quaternion
            context.memory["approach_positions"] = approach_positions
            context.memory["approach_quaternions"] = approach_quaternions
            context.memory["approach_waypoint_index"] = 0
            context.memory["grasp_positions"] = grasp_positions
            context.memory["grasp_quaternions"] = grasp_quaternions
            context.memory["grasp_waypoint_index"] = 0
            context.memory["grasp_wrist_position"] = grasp_wrist
            context.memory["approach_wrist_position"] = approach_wrist
            context.memory["contact_wrist_position"] = contact_wrist
            context.memory["engaged_wrist_position"] = engaged_wrist
            context.memory["template_object_position"] = object_position.copy()
            context.memory["grasp_arm_qpos"] = grasp_arm_qpos

        target_quaternion = np.asarray(context.memory["target_quaternion"], dtype=np.float64)

        if phase == "approach":
            waypoint_index = int(context.memory["approach_waypoint_index"])
            positions = np.asarray(
                context.memory["approach_positions"],
                dtype=np.float64,
            )
            quaternions = np.asarray(
                context.memory["approach_quaternions"],
                dtype=np.float64,
            )
            target = positions[waypoint_index]
            target_quaternion = quaternions[waypoint_index]
            waypoint_hand = self._hand_target(
                env,
                self.approach_hand_fractions[waypoint_index],
            )
            position_error = float(np.linalg.norm(target - ee_position))
            quaternion_dot = abs(float(np.dot(current[3:7], target_quaternion)))
            orientation_error = 2.0 * np.arccos(np.clip(quaternion_dot, 0.0, 1.0))
            converged = (
                context.phase_step >= self.MIN_PHASE_STEPS
                and position_error < self.WAYPOINT_POSITION_TOLERANCE
                and orientation_error < self.ORIENTATION_TOLERANCE
                and float(np.max(np.abs(self._hand_qpos(env) - waypoint_hand))) < 0.001
            )
            if converged and waypoint_index + 1 < len(positions):
                # Intermediate points describe a collision-free polyline, not
                # places where the arm should stop. Look ahead as soon as the
                # measured wrist enters the waypoint tolerance so the IK
                # target moves continuously instead of settling and restarting
                # at every point.
                waypoint_index += 1
                context.memory["approach_waypoint_index"] = waypoint_index
                target = positions[waypoint_index]
                target_quaternion = quaternions[waypoint_index]
                waypoint_hand = self._hand_target(
                    env,
                    self.approach_hand_fractions[waypoint_index],
                )
                return PhaseResult.CONTINUE, ActionContext(
                    target,
                    target_quaternion,
                    waypoint_hand,
                )
            if self._stable(
                context,
                "approach_final_stable_steps",
                converged and waypoint_index + 1 == len(positions),
            ):
                context.memory["target_quaternion"] = target_quaternion
                context.memory["template_object_position"] = object_position.copy()
                context.memory["object_position_at_grasp"] = object_position.copy()
                return PhaseResult.NEXT, ActionContext(
                    target,
                    target_quaternion,
                    waypoint_hand,
                )
            if context.phase_step >= self.PHASE_TIMEOUT and not converged:
                self._advance_grasp_candidate()
                return PhaseResult.RESTART, ActionContext(hand_target=gripper)
            return PhaseResult.CONTINUE, ActionContext(
                target,
                target_quaternion,
                waypoint_hand,
            )

        grasp_wrist = np.asarray(context.memory["grasp_wrist_position"], dtype=np.float64)
        if phase == "grasp":
            positions = np.asarray(context.memory["grasp_positions"], dtype=np.float64)
            quaternions = np.asarray(context.memory["grasp_quaternions"], dtype=np.float64)
            waypoint_index = int(context.memory["grasp_waypoint_index"])
            if waypoint_index < len(positions):
                wrist_target = positions[waypoint_index]
                target_quaternion = quaternions[waypoint_index]
                hand_target = self._hand_target(
                    env,
                    self.grasp_hand_fractions[waypoint_index],
                )
                position_error = float(np.linalg.norm(wrist_target - ee_position))
                quaternion_dot = abs(float(np.dot(current[3:7], target_quaternion)))
                orientation_error = 2.0 * np.arccos(np.clip(quaternion_dot, 0.0, 1.0))
                self.last_wrist_position_error = position_error
                self.last_wrist_orientation_error = orientation_error
                hand_error = float(np.max(np.abs(self._hand_qpos(env) - hand_target)))
                contact_fingers = self._contact_fingers(env)
                self.last_grasp_hand_error = hand_error
                self.last_contact_fingers = contact_fingers
                converged = (
                    context.phase_step >= self.MIN_PHASE_STEPS
                    and position_error < self.WAYPOINT_POSITION_TOLERANCE
                    and orientation_error < self.ORIENTATION_TOLERANCE
                    and hand_error
                    < (0.015 if self.end_effector_name == "pika_gripper" else 0.0015)
                )
                if self._stable(
                    context,
                    "grasp_path_stable_steps",
                    converged,
                    required_steps=3,
                ):
                    context.memory["grasp_waypoint_index"] = waypoint_index + 1
                    context.memory["grasp_path_stable_steps"] = 0
                    if waypoint_index + 1 == len(positions):
                        context.memory["target_quaternion"] = target_quaternion
                        context.memory["grasp_wrist_position"] = wrist_target.copy()
                        context.memory["grasp_preload_step"] = 0
                    return PhaseResult.CONTINUE, ActionContext(
                        wrist_target,
                        target_quaternion,
                        hand_target,
                    )
                if context.phase_step >= self.PHASE_TIMEOUT and not converged:
                    self._advance_grasp_candidate()
                    return PhaseResult.RESTART, ActionContext(hand_target=gripper)
                return PhaseResult.CONTINUE, ActionContext(
                    wrist_target,
                    target_quaternion,
                    hand_target,
                )

            preload_step = int(context.memory.get("grasp_preload_step", 0)) + 1
            context.memory["grasp_preload_step"] = preload_step
            preload_alpha = np.clip(preload_step / 40.0, 0.0, 1.0)
            wrist_target = grasp_wrist
            hand_target = (1.0 - preload_alpha) * grasp + preload_alpha * preload
            hand_error = float(np.max(np.abs(self._hand_qpos(env) - hand_target)))
            contact_fingers = self._contact_fingers(env)
            self.last_grasp_hand_error = hand_error
            self.last_contact_fingers = contact_fingers
            antagonistic_contact = (
                (4 in contact_fingers and any(finger < 4 for finger in contact_fingers))
                if self.end_effector_name == "dex_hand"
                else 0 in contact_fingers and 1 in contact_fingers
            )
            grasp_stable = (
                preload_alpha >= 1.0
                and hand_error
                < (0.015 if self.end_effector_name == "pika_gripper" else 0.0015)
                and antagonistic_contact
            )
            if self._stable(
                context,
                "grasp_stable_steps",
                grasp_stable,
                required_steps=10,
            ):
                return PhaseResult.NEXT, ActionContext(grasp_wrist, target_quaternion, preload)
            if context.phase_step >= self.PHASE_TIMEOUT:
                self._advance_grasp_candidate()
                return PhaseResult.RESTART, ActionContext(hand_target=gripper)
            return PhaseResult.CONTINUE, ActionContext(wrist_target, target_quaternion, hand_target)

        if phase == "lift":
            target = grasp_wrist + np.asarray([0.0, 0.0, self.LIFT_HEIGHT])
            position_error = float(np.linalg.norm(target - ee_position))
            lifted = bool(context.info.get("task_success", False))
            converged = (
                context.phase_step >= self.MIN_PHASE_STEPS and position_error < 0.015 and lifted
            )
            self.state.lift_stable_steps = self.state.lift_stable_steps + 1 if converged else 0
            if self.state.lift_stable_steps >= 5:
                self.state.hold_wrist_position = ee_position.copy()
                return PhaseResult.NEXT, ActionContext(target, target_quaternion, preload)
            if context.phase_step >= self.PHASE_TIMEOUT:
                self.state.reset()
                self._advance_grasp_candidate()
                return PhaseResult.RESTART, ActionContext(hand_target=gripper)
            return PhaseResult.CONTINUE, ActionContext(target, target_quaternion, preload)

        if phase != "verify":
            raise RuntimeError(f"Unsupported lift phase {phase!r}.")
        hold = np.asarray(self.state.hold_wrist_position, dtype=np.float64)
        midpoint_error = float(np.linalg.norm(object_position - self._grasp_midpoint(env)))
        valid = bool(context.info.get("task_success", False)) and (
            midpoint_error <= self.CHECK_MAX_DISTANCE
        )
        self.state.verify_success_steps = self.state.verify_success_steps + 1 if valid else 0
        if self.state.verify_success_steps >= 10:
            self.state.strategy_verified_success = True
            return PhaseResult.NEXT, ActionContext(hold, target_quaternion, preload)
        if context.phase_step >= 40:
            self.state.reset()
            self._advance_grasp_candidate()
            return PhaseResult.RESTART, ActionContext(hand_target=gripper)
        return PhaseResult.CONTINUE, ActionContext(hold, target_quaternion, preload)
