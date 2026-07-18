"""End-effector-independent lift policy using searched grasp configurations."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from source.assets import PROJECT_ROOT
from source.scripted.base import ActionContext, PhaseContext, PhaseResult, TaskStrategy
from source.geometry import mat_to_quat


def _quat_matrix(quaternion_wxyz: np.ndarray) -> np.ndarray:
    matrix = np.empty(9, dtype=np.float64)
    mujoco.mju_quat2Mat(matrix, np.asarray(quaternion_wxyz, dtype=np.float64))
    return matrix.reshape(3, 3)


@dataclass
class LiftStrategyState:
    lift_stable_steps: int = 0
    verify_success_steps: int = 0
    hold_wrist_position: np.ndarray | None = None
    verified_success: bool = False

    def reset(self) -> None:
        self.lift_stable_steps = 0
        self.verify_success_steps = 0
        self.hold_wrist_position = None
        self.verified_success = False


class LiftStrategy(TaskStrategy):
    """Approach along the searched path, grasp, then lift and verify."""

    phases = (
        "approach",
        "grasp",
        "lift",
    )
    PHASE_PROMPTS = {
        "approach": "1/3 APPROACH: follow the collision-free path to the grasp",
        "grasp": "2/3 GRASP: close and preload the fingers",
        "lift": "3/3 LIFT: raise and verify the object",
    }

    PRE_GRASP_HEIGHT = 0.12
    LIFT_HEIGHT = 0.20
    POSITION_TOLERANCE = 0.01
    WAYPOINT_POSITION_TOLERANCE = 0.012
    XY_TOLERANCE = 0.005
    ORIENTATION_TOLERANCE = 0.05
    CHECK_MAX_DISTANCE = 0.06
    MIN_PHASE_STEPS = 10
    GRASP_STEPS = 130
    PHASE_TIMEOUT = 400
    GRASP_QUATERNION = np.asarray([0.0, 1.0, 0.0, 0.0], dtype=np.float64)

    # Fractions of each actuator's physical range. Current-project order is
    # finger0..3, thumb_rotate, thumb_grasp.
    HAND_GRIPPER = np.asarray([1.0, 1.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float64)
    HAND_GRASP = np.asarray([1.0, 1.0, 0.7, 0.7, 0.3, 1.0], dtype=np.float64)
    HAND_CLOSE = np.ones(6, dtype=np.float64)
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
    HAND_ATTACH_ROTATION = Rotation.from_euler(
        "xyz", [-90.0, -90.0, 0.0], degrees=True
    ).as_matrix()
    GRIP_PRELOAD = 0.40
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
    ) -> None:
        super().__init__(max_position_step=max_position_step, max_orientation_step=0.20)
        self.state = LiftStrategyState()
        self.reuse_grasp_config = bool(reuse_grasp_config)
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
        self.grasp_template_path: Path | None = None
        self.grasp_template_object_id: str | None = None

    def reset(self) -> None:
        super().reset()
        self.state.reset()

    @staticmethod
    def _grasp_config_name(object_id: str) -> str:
        return "".join(
            character if character.isalnum() or character in "-_" else "_"
            for character in object_id
        )

    def _ensure_grasp_template(self, env) -> None:
        object_id = getattr(env.task, "object_id", None)
        if not isinstance(object_id, str) or not object_id:
            raise RuntimeError("Lift task does not expose a valid object_id.")
        if (
            self.grasp_template_object_id == object_id
            and self.end_effector_name == env.hand_descriptor.name
        ):
            return

        end_effector_name = env.hand_descriptor.name
        config_dir = PROJECT_ROOT / "configs" / "grasps"
        if end_effector_name != "dex_hand":
            config_dir = config_dir / end_effector_name
        path = config_dir / f"{self._grasp_config_name(object_id)}.json"
        should_generate = not self.reuse_grasp_config or not path.is_file()
        if should_generate:
            # Import lazily so normal cached-policy startup does not pay the
            # visualization import cost of the search demo.
            from source.grasping.grasp_config_search import (
                generate_grasp_config,
            )

            reason = (
                "default fresh search"
                if path.is_file()
                else "no cached config"
            )
            print(f"Generating grasp for {object_id} ({reason}) -> {path} ...")
            path = generate_grasp_config(
                object_id,
                output=path,
                end_effector_name=end_effector_name,
            )
            print(f"Generated grasp config: {path}")

        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("schema_version") != 1:
            raise ValueError(f"Unsupported or missing schema_version in {path}.")
        payload_object_id = payload.get("object_id")
        if payload_object_id != object_id:
            raise ValueError(
                f"Grasp {path} belongs to {payload_object_id!r}, "
                f"not the active object {object_id!r}."
            )
        if payload.get("hand_fit_success") is not True:
            raise ValueError(f"Grasp {path} did not pass mesh fitting.")
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
        actuator_count = env.controller.hand_controller.action_size
        if fractions.shape != (actuator_count,) or np.any(
            (fractions < 0.0) | (fractions > 1.0)
        ):
            raise ValueError(f"Invalid hand_actuator_fractions in {path}.")
        if translation.shape != (3,) or not np.all(np.isfinite(translation)):
            raise ValueError(f"Invalid hand_translation in {path}.")
        if rotation.shape != (3, 3) or not np.all(np.isfinite(rotation)):
            raise ValueError(f"Invalid hand_rotation_matrix in {path}.")
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-4):
            raise ValueError(f"Non-orthonormal hand_rotation_matrix in {path}.")
        if (
            preload_weights.shape != (actuator_count,)
            or np.any((preload_weights < 0.0) | (preload_weights > 1.0))
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
        palmward_force = payload.get("hand_palmward_force_component")
        if palmward_force is not None and float(palmward_force) < 0.0:
            raise ValueError(f"Grasp {path} has an outward preload resultant.")

        self.template_hand_fractions = fractions
        self.template_hand_translation = translation
        self.template_hand_rotation = rotation
        self.template_preload_weights = preload_weights
        self.template_preload_directions = preload_directions
        self.approach_hand_translations = approach_translations
        self.approach_hand_rotations = approach_rotations
        self.approach_hand_fractions = approach_fractions
        self.grasp_template_path = path
        self.grasp_template_object_id = object_id
        self.end_effector_name = end_effector_name
        attach_degrees = (
            env.arm_descriptor.hand_attach_rot_xyz_deg
            if env.config.hand_attach_rot_xyz_deg is None
            else tuple(env.config.hand_attach_rot_xyz_deg)
        )
        self.hand_attach_rotation = Rotation.from_euler(
            "xyz", attach_degrees, degrees=True
        ).as_matrix()

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
            raise ValueError(
                f"Expected {count} end-effector fractions, got {fractions.shape}."
            )
        low = np.asarray(env.action_space.low[-count:], dtype=np.float64)
        high = np.asarray(env.action_space.high[-count:], dtype=np.float64)
        return low + fractions * (high - low)

    @staticmethod
    def _hand_qpos(env) -> np.ndarray:
        hand = env.controller.hand_controller
        joint_positions = np.asarray(
            env.data.qpos[hand.qpos_addrs], dtype=np.float64
        )
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
                if (mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_SITE, site_id) or "").startswith(
                    group
                )
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
                    mujoco.mj_id2name(
                        env.model, mujoco.mjtObj.mjOBJ_SITE, site_id
                    )
                    or ""
                ).startswith(f"{prefix}taxel_pika_")
            ]
            if ids:
                return np.mean(env.data.site_xpos[ids], axis=0)
            site_name = f"{prefix}gripper_tcp"
            site_id = mujoco.mj_name2id(
                env.model, mujoco.mjtObj.mjOBJ_SITE, site_name
            )
            if site_id < 0:
                raise RuntimeError(
                    f"Pika grasp midpoint site {site_name!r} missing."
                )
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
            name = (
                mujoco.mj_id2name(
                    env.model, mujoco.mjtObj.mjOBJ_GEOM, robot_geom
                )
                or ""
            )
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
        palm_normal_local -= (
            np.dot(palm_normal_local, finger_axis_local) * finger_axis_local
        )
        palm_normal_local /= np.linalg.norm(palm_normal_local)
        hand_side_local = np.cross(finger_axis_local, palm_normal_local)
        hand_side_local /= np.linalg.norm(hand_side_local)
        source_basis = np.column_stack(
            [finger_axis_local, palm_normal_local, hand_side_local]
        )
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
            score = 5.0 * (1.0 - alignment) + float(
                np.linalg.norm(wrist - current_ee_position)
            )
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
        hand_position = (
            object_position
            + object_rotation
            @ symmetry_rotation
            @ local_translation
        )
        hand_rotation = (
            object_rotation
            @ symmetry_rotation
            @ self.template_hand_rotation
            @ tool_symmetry
        )
        ee_rotation = hand_rotation @ self.hand_attach_rotation.T
        return hand_position, mat_to_quat(ee_rotation)

    def _select_reachable_template_pose(
        self,
        env,
        object_position: np.ndarray,
        object_quaternion: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
        """Choose a symmetry-equivalent can grasp using actual IK residual."""
        arm = env.controller.arm_controller
        saved_qpos = env.data.qpos.copy()
        saved_qvel = env.data.qvel.copy()
        saved_ctrl = env.data.ctrl.copy()
        previous_velocity = arm.max_joint_velocity
        previous_filter = arm.velocity_filter_alpha
        previous_target_q = (
            None if arm._prev_target_q is None else arm._prev_target_q.copy()
        )
        previous_filtered_velocity = (
            None
            if arm._filtered_velocity is None
            else arm._filtered_velocity.copy()
        )
        arm.max_joint_velocity = 100.0
        arm.velocity_filter_alpha = 1.0
        best = None
        try:
            rolls = (
                (0.0, np.pi)
                if self.end_effector_name == "pika_gripper"
                else (0.0,)
            )
            for yaw, tool_roll in (
                (float(yaw), float(tool_roll))
                for yaw in np.linspace(-np.pi, np.pi, 16, endpoint=False)
                for tool_roll in rolls
            ):
                grasp_position, quaternion = self._template_wrist_pose(
                    object_position,
                    object_quaternion,
                    yaw,
                    tool_roll,
                )
                approach_positions, approach_quaternions = (
                    self._world_approach_waypoints(
                        object_position,
                        object_quaternion,
                        yaw,
                        tool_roll,
                    )
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
                    actual_quaternion = mat_to_quat(
                        env.data.site_xmat[arm.site_id]
                    )
                    position_error = np.linalg.norm(actual_position - position)
                    orientation_error = 2.0 * np.arccos(
                        np.clip(
                            abs(
                                float(
                                    np.dot(
                                        actual_quaternion,
                                        waypoint_quaternion,
                                    )
                                )
                            ),
                            0.0,
                            1.0,
                        )
                    )
                    residual += float(position_error + 0.08 * orientation_error)
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

    def _world_approach_waypoints(
        self,
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
            self.approach_hand_translations
            + (self.approach_hand_rotations @ pivot)
            - (
                self.approach_hand_rotations
                @ tool_symmetry
                @ pivot
            )
        )
        positions = (
            object_position[None, :]
            + (
                object_rotation
                @ symmetry_rotation
                @ local_positions.T
            ).T
        )
        quaternions = []
        for hand_rotation_local in self.approach_hand_rotations:
            hand_rotation = (
                object_rotation
                @ symmetry_rotation
                @ hand_rotation_local
                @ tool_symmetry
            )
            quaternions.append(
                mat_to_quat(hand_rotation @ self.hand_attach_rotation.T)
            )
        return positions, np.asarray(quaternions)

    def execute_phase(
        self, phase_index: int, context: PhaseContext
    ) -> tuple[PhaseResult, ActionContext]:
        phase = self.phases[phase_index]
        env = context.env
        self._ensure_grasp_template(env)
        current = env.controller.current_ik_action(env.model, env.data).astype(np.float64)
        ee_position = current[:3]
        object_position = np.asarray(context.observation["object_pos"], dtype=np.float64)
        object_quaternion = np.asarray(context.observation["object_quat"], dtype=np.float64)
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
            open_ready = (
                context.phase_step >= self.MIN_PHASE_STEPS
                and self._stable(
                    context,
                    "open_hand_stable_steps",
                    open_error < 0.0007,
                )
            )
            if not open_ready:
                return PhaseResult.CONTINUE, ActionContext(hand_target=gripper)
            (
                grasp_wrist,
                target_quaternion,
                grasp_arm_qpos,
                template_yaw,
                template_tool_roll,
            ) = self._select_reachable_template_pose(
                env, object_position, object_quaternion
            )
            approach_positions, approach_quaternions = (
                self._world_approach_waypoints(
                    object_position,
                    object_quaternion,
                    template_yaw,
                    template_tool_roll,
                )
            )
            approach_wrist = approach_positions[0]
            contact_wrist = grasp_wrist.copy()
            engaged_wrist = grasp_wrist.copy()
            context.memory["target_quaternion"] = approach_quaternions[0]
            context.memory["grasp_target_quaternion"] = target_quaternion
            context.memory["approach_positions"] = approach_positions
            context.memory["approach_quaternions"] = approach_quaternions
            context.memory["approach_waypoint_index"] = 0
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
            orientation_error = 2.0 * np.arccos(
                np.clip(quaternion_dot, 0.0, 1.0)
            )
            converged = (
                context.phase_step >= self.MIN_PHASE_STEPS
                and position_error < self.WAYPOINT_POSITION_TOLERANCE
                and orientation_error < self.ORIENTATION_TOLERANCE
                and float(
                    np.max(np.abs(self._hand_qpos(env) - waypoint_hand))
                )
                < 0.001
            )
            if self._stable(context, "approach_stable_steps", converged):
                if waypoint_index + 1 < len(positions):
                    context.memory["approach_waypoint_index"] = waypoint_index + 1
                    context.memory["approach_stable_steps"] = 0
                    return PhaseResult.CONTINUE, ActionContext(
                        target,
                        target_quaternion,
                        waypoint_hand,
                    )
                reached_wrist = ee_position.copy()
                for key in (
                    "contact_wrist_position",
                    "engaged_wrist_position",
                    "grasp_wrist_position",
                ):
                    context.memory[key] = reached_wrist.copy()
                context.memory["target_quaternion"] = target_quaternion
                context.memory["template_object_position"] = object_position.copy()
                context.memory["object_position_at_grasp"] = object_position.copy()
                return PhaseResult.NEXT, ActionContext(
                    target,
                    target_quaternion,
                    waypoint_hand,
                )
            if context.phase_step >= self.PHASE_TIMEOUT:
                return PhaseResult.RESTART, ActionContext(hand_target=gripper)
            return PhaseResult.CONTINUE, ActionContext(
                target,
                target_quaternion,
                waypoint_hand,
            )

        if phase == "descend":
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
            position_error = float(np.linalg.norm(ee_position - target))
            quaternion_dot = abs(float(np.dot(current[3:7], target_quaternion)))
            orientation_error = 2.0 * np.arccos(
                np.clip(quaternion_dot, 0.0, 1.0)
            )
            converged = (
                context.phase_step >= self.MIN_PHASE_STEPS
                and position_error < self.WAYPOINT_POSITION_TOLERANCE
                and orientation_error < self.ORIENTATION_TOLERANCE
                and float(
                    np.max(np.abs(self._hand_qpos(env) - waypoint_hand))
                )
                < 0.001
            )
            if self._stable(context, "descend_stable_steps", converged):
                if waypoint_index + 1 < len(positions):
                    context.memory["approach_waypoint_index"] = waypoint_index + 1
                    context.memory["descend_stable_steps"] = 0
                    return PhaseResult.CONTINUE, ActionContext(
                        target,
                        target_quaternion,
                        waypoint_hand,
                    )
                reached_wrist = ee_position.copy()
                for key in (
                    "contact_wrist_position",
                    "engaged_wrist_position",
                    "grasp_wrist_position",
                ):
                    context.memory[key] = reached_wrist.copy()
                context.memory["target_quaternion"] = target_quaternion
                return PhaseResult.NEXT, ActionContext(
                    target,
                    target_quaternion,
                    waypoint_hand,
                )
            if context.phase_step >= self.PHASE_TIMEOUT:
                return PhaseResult.RESTART, ActionContext(hand_target=gripper)
            return PhaseResult.CONTINUE, ActionContext(
                target,
                target_quaternion,
                waypoint_hand,
            )

        if phase == "adjust":
            contact_target = np.asarray(
                context.memory["contact_wrist_position"], dtype=np.float64
            )
            template_object_position = np.asarray(
                context.memory["template_object_position"], dtype=np.float64
            )
            object_delta = object_position - template_object_position
            target = contact_target + object_delta
            position_error = float(np.linalg.norm(ee_position - target))
            quaternion_dot = abs(float(np.dot(current[3:7], target_quaternion)))
            orientation_error = 2.0 * np.arccos(
                np.clip(quaternion_dot, 0.0, 1.0)
            )
            converged = (
                context.phase_step >= self.MIN_PHASE_STEPS
                and position_error < self.XY_TOLERANCE
                and orientation_error < self.ORIENTATION_TOLERANCE
            )
            if self._stable(context, "adjust_stable_steps", converged):
                for key in (
                    "approach_wrist_position",
                    "contact_wrist_position",
                    "engaged_wrist_position",
                    "grasp_wrist_position",
                ):
                    context.memory[key] = (
                        np.asarray(context.memory[key], dtype=np.float64)
                        + object_delta
                    )
                context.memory["template_object_position"] = object_position.copy()
                context.memory["object_position_at_grasp"] = object_position.copy()
                return PhaseResult.NEXT, ActionContext(target, target_quaternion, gripper)
            if context.phase_step >= self.PHASE_TIMEOUT:
                return PhaseResult.RESTART, ActionContext(hand_target=gripper)
            return PhaseResult.CONTINUE, ActionContext(target, target_quaternion, gripper)

        if phase == "make_gripper_hand_form":
            contact_wrist = np.asarray(
                context.memory["contact_wrist_position"], dtype=np.float64
            )
            error = float(np.max(np.abs(self._hand_qpos(env) - grasp)))
            ready = (
                context.phase_step >= self.MIN_PHASE_STEPS
                and self._stable(
                    context,
                    "hand_form_stable_steps",
                    error < 0.0007,
                )
            )
            if ready:
                return PhaseResult.NEXT, ActionContext(
                    contact_wrist, target_quaternion, grasp
                )
            if context.phase_step >= self.PHASE_TIMEOUT:
                return PhaseResult.RESTART, ActionContext(hand_target=gripper)
            return PhaseResult.CONTINUE, ActionContext(
                contact_wrist, target_quaternion, grasp
            )

        grasp_wrist = np.asarray(context.memory["grasp_wrist_position"], dtype=np.float64)
        if phase == "grasp":
            engaged_wrist = np.asarray(
                context.memory["engaged_wrist_position"], dtype=np.float64
            )
            close_alpha = np.clip(context.phase_step / 100.0, 0.0, 1.0)
            wrist_target = engaged_wrist
            hand_target = (
                (1.0 - close_alpha) * grasp + close_alpha * preload
            )
            hand_error = float(np.max(np.abs(self._hand_qpos(env) - hand_target)))
            contact_fingers = self._contact_fingers(env)
            antagonistic_contact = (
                (
                    4 in contact_fingers
                    and any(finger < 4 for finger in contact_fingers)
                )
                if self.end_effector_name == "dex_hand"
                else 0 in contact_fingers and 1 in contact_fingers
            )
            grasp_stable = (
                context.phase_step >= self.GRASP_STEPS
                and hand_error
                < (
                    0.015
                    if self.end_effector_name == "pika_gripper"
                    else 0.0015
                )
                and antagonistic_contact
            )
            if self._stable(
                context,
                    "grasp_stable_steps",
                grasp_stable,
                required_steps=10,
            ):
                context.memory["grasp_wrist_position"] = engaged_wrist
                return PhaseResult.NEXT, ActionContext(
                    engaged_wrist, target_quaternion, preload
                )
            if context.phase_step >= self.PHASE_TIMEOUT:
                return PhaseResult.RESTART, ActionContext(hand_target=gripper)
            return PhaseResult.CONTINUE, ActionContext(
                wrist_target, target_quaternion, hand_target
            )

        if phase == "lift":
            target = grasp_wrist + np.asarray([0.0, 0.0, self.LIFT_HEIGHT])
            position_error = float(np.linalg.norm(target - ee_position))
            lifted = bool(context.info.get("task_success", False))
            converged = (
                context.phase_step >= self.MIN_PHASE_STEPS
                and position_error < 0.015
                and lifted
            )
            self.state.lift_stable_steps = (
                self.state.lift_stable_steps + 1 if converged else 0
            )
            if self.state.lift_stable_steps >= 5:
                self.state.hold_wrist_position = ee_position.copy()
                self.state.verified_success = True
                return PhaseResult.NEXT, ActionContext(target, target_quaternion, preload)
            if context.phase_step >= self.PHASE_TIMEOUT:
                self.state.reset()
                return PhaseResult.RESTART, ActionContext(hand_target=gripper)
            return PhaseResult.CONTINUE, ActionContext(target, target_quaternion, preload)

        hold = np.asarray(self.state.hold_wrist_position, dtype=np.float64)
        midpoint_error = float(np.linalg.norm(object_position - self._grasp_midpoint(env)))
        valid = bool(context.info.get("task_success", False)) and (
            midpoint_error <= self.CHECK_MAX_DISTANCE
        )
        self.state.verify_success_steps = (
            self.state.verify_success_steps + 1 if valid else 0
        )
        if self.state.verify_success_steps >= 10:
            self.state.verified_success = True
            return PhaseResult.NEXT, ActionContext(hold, target_quaternion, preload)
        if context.phase_step >= 40:
            self.state.reset()
            return PhaseResult.RESTART, ActionContext(hand_target=gripper)
        return PhaseResult.CONTINUE, ActionContext(hold, target_quaternion, preload)
