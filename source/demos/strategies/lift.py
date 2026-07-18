"""Dex-hand lift policy adapted from ``mujoco_project`` BlockLiftingStrategy."""

from __future__ import annotations

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from source.demos.strategies.base import ActionContext, PhaseContext, PhaseResult, TaskStrategy
from source.geometry import mat_to_quat


def _quat_matrix(quaternion_wxyz: np.ndarray) -> np.ndarray:
    matrix = np.empty(9, dtype=np.float64)
    mujoco.mju_quat2Mat(matrix, np.asarray(quaternion_wxyz, dtype=np.float64))
    return matrix.reshape(3, 3)


class LiftStrategy(TaskStrategy):
    """Pre-form, approach, descend, adjust, grasp, lift, and check."""

    phases = (
        "approach",
        "descend",
        "adjust",
        "make_gripper_hand_form",
        "grasp",
        "lift",
        "check",
    )
    PHASE_PROMPTS = {
        "approach": "1/7 APPROACH: approach with fingers open and thumb opposed",
        "descend": "2/7 ADVANCE: move toward the collision-free pregrasp",
        "adjust": "3/7 ADJUST: realign the pregrasp to the current object pose",
        "make_gripper_hand_form": "4/7 HAND FORM: form the optimized grasp shape",
        "grasp": "5/7 GRASP: close fingers while holding the wrist pose",
        "lift": "6/7 LIFT: raise the grasp center vertically",
        "check": "7/7 CHECK: verify object height and grasp-center distance",
    }

    PRE_GRASP_HEIGHT = 0.12
    LIFT_HEIGHT = 0.20
    POSITION_TOLERANCE = 0.01
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
        "descend": 0.07,
        "adjust": 0.05,
        "grasp": 0.05,
        "lift": 0.10,
        "check": 0.05,
    }
    phase_orientation_speeds = {
        "approach": 1.00,
        "descend": 0.60,
        "adjust": 0.40,
        "grasp": 0.40,
        "lift": 0.50,
        "check": 0.40,
    }

    def __init__(self, *, max_position_step: float = 0.03) -> None:
        super().__init__(max_position_step=max_position_step, max_orientation_step=0.20)

    @property
    def phase_prompt(self) -> str:
        if self.finished:
            return "7/7 VERIFIED: object is lifted and remains inside the hand"
        return self.PHASE_PROMPTS[self.phase_name]

    @staticmethod
    def _hand_target(env, fractions: np.ndarray) -> np.ndarray:
        hand = env.controller.hand_controller
        if hand.action_size != 6:
            raise ValueError(
                "LiftStrategy requires the six-actuator Dex Hand; "
                f"got {hand.action_size} actuators."
            )
        low = np.asarray(env.action_space.low[-6:], dtype=np.float64)
        high = np.asarray(env.action_space.high[-6:], dtype=np.float64)
        return low + fractions * (high - low)

    @staticmethod
    def _hand_qpos(env) -> np.ndarray:
        hand = env.controller.hand_controller
        return np.asarray(env.data.qpos[hand.qpos_addrs], dtype=np.float64)

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

    @classmethod
    def _template_wrist_pose(
        cls,
        object_position: np.ndarray,
        object_quaternion: np.ndarray,
        yaw: float = 0.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        object_rotation = _quat_matrix(object_quaternion)
        symmetry_rotation = Rotation.from_euler("z", yaw).as_matrix()
        hand_position = (
            object_position
            + object_rotation
            @ symmetry_rotation
            @ cls.TEMPLATE_HAND_TRANSLATION
        )
        hand_rotation = (
            object_rotation @ symmetry_rotation @ cls.TEMPLATE_HAND_ROTATION
        )
        ee_rotation = hand_rotation @ cls.HAND_ATTACH_ROTATION.T
        return hand_position, mat_to_quat(ee_rotation)

    @classmethod
    def _select_reachable_template_pose(
        cls,
        env,
        object_position: np.ndarray,
        object_quaternion: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
            for yaw in np.linspace(-np.pi, np.pi, 16, endpoint=False):
                grasp_position, quaternion = cls._template_wrist_pose(
                    object_position,
                    object_quaternion,
                    float(yaw),
                )
                residual = 0.0
                grasp_arm_qpos = None
                for position in (
                    grasp_position
                    + np.asarray([0.0, 0.0, cls.PRE_GRASP_HEIGHT]),
                    grasp_position,
                ):
                    arm_qpos = arm._solve_ik(
                        env.model, env.data, position, quaternion
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
                            abs(float(np.dot(actual_quaternion, quaternion))),
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
        return best[1], best[2], best[3]

    def execute_phase(
        self, phase_index: int, context: PhaseContext
    ) -> tuple[PhaseResult, ActionContext]:
        phase = self.phases[phase_index]
        env = context.env
        current = env.controller.current_ik_action(env.model, env.data).astype(np.float64)
        ee_position = current[:3]
        object_position = np.asarray(context.observation["object_pos"], dtype=np.float64)
        object_quaternion = np.asarray(context.observation["object_quat"], dtype=np.float64)
        gripper_fractions = np.asarray(
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float64
        )
        gripper = self._hand_target(env, gripper_fractions)
        grasp = self._hand_target(env, self.TEMPLATE_HAND_FRACTIONS)
        preload_fractions = self.TEMPLATE_HAND_FRACTIONS.copy()
        preload_indices = np.asarray([0, 1, 2, 3, 5])
        preload_fractions[preload_indices] += self.GRIP_PRELOAD * (
            1.0 - preload_fractions[preload_indices]
        )
        preload_fractions[5] = 1.0
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
            ) = self._select_reachable_template_pose(
                env, object_position, object_quaternion
            )
            approach_wrist = grasp_wrist + np.asarray(
                [0.0, 0.0, self.PRE_GRASP_HEIGHT]
            )
            contact_wrist = grasp_wrist.copy()
            engaged_wrist = grasp_wrist.copy()
            context.memory["target_quaternion"] = target_quaternion
            context.memory["grasp_wrist_position"] = grasp_wrist
            context.memory["approach_wrist_position"] = approach_wrist
            context.memory["contact_wrist_position"] = contact_wrist
            context.memory["engaged_wrist_position"] = engaged_wrist
            context.memory["template_object_position"] = object_position.copy()
            context.memory["grasp_arm_qpos"] = grasp_arm_qpos

        target_quaternion = np.asarray(context.memory["target_quaternion"], dtype=np.float64)

        if phase == "approach":
            target = np.asarray(
                context.memory["approach_wrist_position"], dtype=np.float64
            )
            position_error = float(np.linalg.norm(target - ee_position))
            quaternion_dot = abs(
                float(np.dot(current[3:7], target_quaternion))
            )
            orientation_error = 2.0 * np.arccos(
                np.clip(quaternion_dot, 0.0, 1.0)
            )
            converged = (
                context.phase_step >= self.MIN_PHASE_STEPS
                and position_error < self.POSITION_TOLERANCE
                and orientation_error < self.ORIENTATION_TOLERANCE
            )
            if self._stable(context, "approach_stable_steps", converged):
                return PhaseResult.NEXT, ActionContext(target, target_quaternion, gripper)
            if context.phase_step >= self.PHASE_TIMEOUT:
                return PhaseResult.RESTART, ActionContext(hand_target=gripper)
            return PhaseResult.CONTINUE, ActionContext(target, target_quaternion, gripper)

        if phase == "descend":
            target = np.asarray(
                context.memory["contact_wrist_position"], dtype=np.float64
            )
            position_error = float(np.linalg.norm(ee_position - target))
            quaternion_dot = abs(float(np.dot(current[3:7], target_quaternion)))
            orientation_error = 2.0 * np.arccos(
                np.clip(quaternion_dot, 0.0, 1.0)
            )
            converged = (
                context.phase_step >= self.MIN_PHASE_STEPS
                and position_error < self.POSITION_TOLERANCE
                and orientation_error < self.ORIENTATION_TOLERANCE
            )
            if self._stable(context, "descend_stable_steps", converged):
                return PhaseResult.NEXT, ActionContext(target, target_quaternion, gripper)
            if context.phase_step >= self.PHASE_TIMEOUT:
                return PhaseResult.RESTART, ActionContext(hand_target=gripper)
            return PhaseResult.CONTINUE, ActionContext(target, target_quaternion, gripper)

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
            wrist_target = engaged_wrist
            close_alpha = np.clip(context.phase_step / 100.0, 0.0, 1.0)
            hand_target = (
                (1.0 - close_alpha) * grasp + close_alpha * preload
            )
            hand_error = float(np.max(np.abs(self._hand_qpos(env) - hand_target)))
            contact_fingers = self._contact_fingers(env)
            antagonistic_contact = (
                4 in contact_fingers
                and any(finger < 4 for finger in contact_fingers)
            )
            grasp_stable = (
                context.phase_step >= self.GRASP_STEPS
                and hand_error < 0.0015
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
            if self._stable(context, "lift_stable_steps", converged):
                context.memory["hold_wrist_position"] = ee_position.copy()
                return PhaseResult.NEXT, ActionContext(target, target_quaternion, preload)
            if context.phase_step >= self.PHASE_TIMEOUT:
                return PhaseResult.RESTART, ActionContext(hand_target=gripper)
            return PhaseResult.CONTINUE, ActionContext(target, target_quaternion, preload)

        hold = np.asarray(context.memory["hold_wrist_position"], dtype=np.float64)
        midpoint_error = float(np.linalg.norm(object_position - self._grasp_midpoint(env)))
        valid = bool(context.info.get("task_success", False)) and (
            midpoint_error <= self.CHECK_MAX_DISTANCE
        )
        consecutive = int(context.memory.get("verify_success_steps", 0))
        context.memory["verify_success_steps"] = consecutive + 1 if valid else 0
        if context.memory["verify_success_steps"] >= 10:
            context.memory["verified_success"] = True
            return PhaseResult.NEXT, ActionContext(hold, target_quaternion, preload)
        if context.phase_step >= 40:
            return PhaseResult.RESTART, ActionContext(hand_target=gripper)
        return PhaseResult.CONTINUE, ActionContext(hold, target_quaternion, preload)
