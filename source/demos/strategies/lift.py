"""Dex-hand lift policy adapted from ``mujoco_project`` BlockLiftingStrategy."""

from __future__ import annotations

import mujoco
import numpy as np

from source.demos.strategies.base import ActionContext, PhaseContext, PhaseResult, TaskStrategy


def _quat_matrix(quaternion_wxyz: np.ndarray) -> np.ndarray:
    matrix = np.empty(9, dtype=np.float64)
    mujoco.mju_quat2Mat(matrix, np.asarray(quaternion_wxyz, dtype=np.float64))
    return matrix.reshape(3, 3)


class LiftStrategy(TaskStrategy):
    """Pre-form, approach, descend, adjust, grasp, lift, and check."""

    phases = (
        "make_gripper_hand_form",
        "approach",
        "descend",
        "adjust",
        "grasp",
        "lift",
        "check",
    )
    PHASE_PROMPTS = {
        "make_gripper_hand_form": "1/7 HAND FORM: shape thumb and fingers as a gripper",
        "approach": "2/7 APPROACH: move the calibrated grasp center above the object",
        "descend": "3/7 DESCEND: lower the grasp center to the object center height",
        "adjust": "4/7 ADJUST: correct grasp-center XY alignment",
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
    GRASP_STEPS = 100
    PHASE_TIMEOUT = 400
    GRASP_QUATERNION = np.asarray([0.0, 1.0, 0.0, 0.0], dtype=np.float64)

    # Fractions of each actuator's physical range. Current-project order is
    # finger0..3, thumb_rotate, thumb_grasp.
    HAND_GRIPPER = np.asarray([1.0, 1.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float64)
    HAND_GRASP = np.asarray([1.0, 1.0, 0.7, 0.7, 0.3, 1.0], dtype=np.float64)
    HAND_CLOSE = np.ones(6, dtype=np.float64)
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

    def execute_phase(
        self, phase_index: int, context: PhaseContext
    ) -> tuple[PhaseResult, ActionContext]:
        phase = self.phases[phase_index]
        env = context.env
        current = env.controller.current_ik_action(env.model, env.data).astype(np.float64)
        ee_position = current[:3]
        object_position = np.asarray(context.observation["object_pos"], dtype=np.float64)
        object_quaternion = np.asarray(context.observation["object_quat"], dtype=np.float64)
        table_z = float(env.task.table_top_z)
        gripper = self._hand_target(env, self.HAND_GRIPPER)
        grasp = self._hand_target(env, self.HAND_GRASP)
        closed = self._hand_target(env, self.HAND_CLOSE)

        if phase == "make_gripper_hand_form":
            error = float(np.max(np.abs(self._hand_qpos(env) - gripper)))
            ready = (
                context.phase_step >= self.MIN_PHASE_STEPS
                and self._stable(
                    context,
                    "hand_form_stable_steps",
                    error < 0.0007,
                )
            )
            if ready:
                current_quaternion = current[3:7].copy()
                finger_center, thumb_center = self._distal_centers(env)
                midpoint = 0.5 * (finger_center + thumb_center)
                current_rotation = _quat_matrix(current_quaternion)
                context.memory["grasp_offset_local"] = (
                    current_rotation.T @ (midpoint - ee_position)
                )
                finger_axis_world = finger_center - thumb_center
                context.memory["finger_axis_local"] = (
                    current_rotation.T
                    @ (finger_axis_world / np.linalg.norm(finger_axis_world))
                )
                approach_midpoint = np.asarray(
                    [object_position[0], object_position[1], table_z + self.PRE_GRASP_HEIGHT],
                    dtype=np.float64,
                )
                target_quaternion = self._select_grasp_quaternion(
                    object_quaternion,
                    approach_midpoint,
                    context.memory["grasp_offset_local"],
                    context.memory["finger_axis_local"],
                    ee_position,
                )
                context.memory["target_quaternion"] = target_quaternion
                context.memory["approach_midpoint"] = approach_midpoint
                return PhaseResult.NEXT, ActionContext(hand_target=gripper)
            return PhaseResult.CONTINUE, ActionContext(hand_target=gripper)

        offset_local = np.asarray(context.memory["grasp_offset_local"], dtype=np.float64)
        target_quaternion = np.asarray(context.memory["target_quaternion"], dtype=np.float64)

        if phase == "approach":
            midpoint = np.asarray(context.memory["approach_midpoint"], dtype=np.float64)
            target = self._wrist_from_midpoint(midpoint, target_quaternion, offset_local)
            position_error = float(np.linalg.norm(target - ee_position))
            current_midpoint = self._grasp_midpoint(env)
            midpoint_xy_error = float(
                np.linalg.norm(current_midpoint[:2] - midpoint[:2])
            )
            midpoint_z_error = abs(float(current_midpoint[2] - midpoint[2]))
            quaternion_dot = abs(
                float(np.dot(current[3:7], target_quaternion))
            )
            orientation_error = 2.0 * np.arccos(
                np.clip(quaternion_dot, 0.0, 1.0)
            )
            converged = (
                context.phase_step >= self.MIN_PHASE_STEPS
                and position_error < self.POSITION_TOLERANCE
                and midpoint_xy_error < 0.01
                and midpoint_z_error < 0.01
                and orientation_error < self.ORIENTATION_TOLERANCE
            )
            if self._stable(context, "approach_stable_steps", converged):
                grasp_midpoint = np.asarray(
                    [
                        object_position[0],
                        object_position[1],
                        object_position[2],
                    ],
                    dtype=np.float64,
                )
                context.memory["descend_midpoint"] = grasp_midpoint
                return PhaseResult.NEXT, ActionContext(target, target_quaternion, gripper)
            if context.phase_step >= self.PHASE_TIMEOUT:
                return PhaseResult.RESTART, ActionContext(hand_target=gripper)
            return PhaseResult.CONTINUE, ActionContext(target, target_quaternion, gripper)

        if phase == "descend":
            midpoint = np.asarray(context.memory["descend_midpoint"], dtype=np.float64)
            target = self._wrist_from_midpoint(midpoint, target_quaternion, offset_local)
            current_midpoint = self._grasp_midpoint(env)
            midpoint_error = float(np.linalg.norm(current_midpoint - midpoint))
            quaternion_dot = abs(float(np.dot(current[3:7], target_quaternion)))
            orientation_error = 2.0 * np.arccos(
                np.clip(quaternion_dot, 0.0, 1.0)
            )
            converged = (
                context.phase_step >= self.MIN_PHASE_STEPS
                and midpoint_error < self.POSITION_TOLERANCE
                and orientation_error < self.ORIENTATION_TOLERANCE
            )
            if self._stable(context, "descend_stable_steps", converged):
                context.memory["adjust_midpoint"] = np.asarray(
                    [object_position[0], object_position[1], current_midpoint[2]],
                    dtype=np.float64,
                )
                return PhaseResult.NEXT, ActionContext(target, target_quaternion, gripper)
            if context.phase_step >= self.PHASE_TIMEOUT:
                return PhaseResult.RESTART, ActionContext(hand_target=gripper)
            return PhaseResult.CONTINUE, ActionContext(target, target_quaternion, gripper)

        if phase == "adjust":
            midpoint = np.asarray(context.memory["adjust_midpoint"], dtype=np.float64)
            target = self._wrist_from_midpoint(midpoint, target_quaternion, offset_local)
            current_midpoint = self._grasp_midpoint(env)
            xy_error = float(np.linalg.norm(current_midpoint[:2] - midpoint[:2]))
            quaternion_dot = abs(float(np.dot(current[3:7], target_quaternion)))
            orientation_error = 2.0 * np.arccos(
                np.clip(quaternion_dot, 0.0, 1.0)
            )
            converged = (
                context.phase_step >= self.MIN_PHASE_STEPS
                and xy_error < self.XY_TOLERANCE
                and orientation_error < self.ORIENTATION_TOLERANCE
            )
            if self._stable(context, "adjust_stable_steps", converged):
                context.memory["grasp_wrist_position"] = ee_position.copy()
                context.memory["object_position_at_grasp"] = object_position.copy()
                return PhaseResult.NEXT, ActionContext(target, target_quaternion, gripper)
            if context.phase_step >= self.PHASE_TIMEOUT:
                return PhaseResult.RESTART, ActionContext(hand_target=gripper)
            return PhaseResult.CONTINUE, ActionContext(target, target_quaternion, gripper)

        grasp_wrist = np.asarray(context.memory["grasp_wrist_position"], dtype=np.float64)
        if phase == "grasp":
            hand_error = float(np.max(np.abs(self._hand_qpos(env) - grasp)))
            grasp_stable = (
                context.phase_step >= self.GRASP_STEPS
                and hand_error < 0.0015
            )
            if self._stable(
                context,
                "grasp_stable_steps",
                grasp_stable,
                required_steps=10,
            ):
                return PhaseResult.NEXT, ActionContext(
                    grasp_wrist, target_quaternion, grasp
                )
            if context.phase_step >= self.PHASE_TIMEOUT:
                return PhaseResult.RESTART, ActionContext(hand_target=gripper)
            return PhaseResult.CONTINUE, ActionContext(grasp_wrist, target_quaternion, grasp)

        if phase == "lift":
            object_at_grasp = np.asarray(
                context.memory["object_position_at_grasp"], dtype=np.float64
            )
            lift_midpoint = np.asarray(
                [
                    object_at_grasp[0],
                    object_at_grasp[1],
                    table_z + self.PRE_GRASP_HEIGHT + self.LIFT_HEIGHT,
                ],
                dtype=np.float64,
            )
            target = self._wrist_from_midpoint(lift_midpoint, target_quaternion, offset_local)
            position_error = float(np.linalg.norm(target - ee_position))
            lifted = bool(context.info.get("task_success", False))
            converged = (
                context.phase_step >= self.MIN_PHASE_STEPS
                and position_error < 0.015
                and lifted
            )
            if self._stable(context, "lift_stable_steps", converged):
                context.memory["hold_wrist_position"] = ee_position.copy()
                return PhaseResult.NEXT, ActionContext(target, target_quaternion, closed)
            if context.phase_step >= self.PHASE_TIMEOUT:
                return PhaseResult.RESTART, ActionContext(hand_target=gripper)
            return PhaseResult.CONTINUE, ActionContext(target, target_quaternion, closed)

        hold = np.asarray(context.memory["hold_wrist_position"], dtype=np.float64)
        midpoint_error = float(np.linalg.norm(object_position - self._grasp_midpoint(env)))
        valid = bool(context.info.get("task_success", False)) and (
            midpoint_error <= self.CHECK_MAX_DISTANCE
        )
        consecutive = int(context.memory.get("verify_success_steps", 0))
        context.memory["verify_success_steps"] = consecutive + 1 if valid else 0
        if context.memory["verify_success_steps"] >= 10:
            context.memory["verified_success"] = True
            return PhaseResult.NEXT, ActionContext(hold, target_quaternion, closed)
        if context.phase_step >= 40:
            return PhaseResult.RESTART, ActionContext(hand_target=gripper)
        return PhaseResult.CONTINUE, ActionContext(hold, target_quaternion, closed)
