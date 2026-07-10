# -*- coding: utf-8 -*-
"""Arm controllers."""

from __future__ import annotations

from typing import Any, Dict, Optional

from gymnasium import spaces
import mujoco
import numpy as np

from source.control.common import IK_ACTION_LAYOUT, _actuator_ids_or_raise, _validate_mode
from source.geometry import (
    mat_to_quat,
    normalize_quat,
    quat_conjugate,
    quat_multiply,
    quat_to_rotvec,
)
from source.robots.descriptors import ArmDescriptor

class ArmPositionIkController:
    """Position/IK controller for a single arm descriptor."""

    def __init__(
        self,
        *,
        arm_descriptor: ArmDescriptor,
        control_mode: str = "position",
        ee_site_name: str | None = None,
        normalized_position: bool = False,
        reset_to_current_position: bool = True,
        ik_iterations: int = 80,
        damping: float = 1e-3,
        damping_adaptive: bool = True,
        damping_min: float = 1e-4,
        damping_max: float = 1e-1,
        damping_singular_threshold: float = 0.05,
        max_joint_step: float = 0.15,
        max_joint_velocity: float = 2.0,
        velocity_filter_alpha: float = 0.3,
        target_filter_alpha: float = 0.7,
        position_tolerance: float = 1e-4,
        orientation_tolerance: float = 1e-3,
        position_weight: float = 1.0,
        orientation_weight: float = 0.35,
        use_nullspace: bool = True,
        nullspace_gain: float = 0.1,
        nullspace_posture: Optional[np.ndarray] = None,
    ) -> None:
        self.arm_descriptor = arm_descriptor
        self.actuator_names = tuple(arm_descriptor.position_actuator_names)
        self.ee_site_name = ee_site_name or arm_descriptor.ee_site_name
        self.normalized_position = normalized_position
        self.reset_to_current_position = reset_to_current_position
        self.control_mode = _validate_mode(control_mode)

        self.ik_iterations = ik_iterations
        self.damping = damping
        self.damping_adaptive = damping_adaptive
        self.damping_min = damping_min
        self.damping_max = damping_max
        self.damping_singular_threshold = damping_singular_threshold
        self.max_joint_step = max_joint_step
        self.max_joint_velocity = max_joint_velocity
        self.velocity_filter_alpha = velocity_filter_alpha
        self.target_filter_alpha = target_filter_alpha
        self.position_tolerance = position_tolerance
        self.orientation_tolerance = orientation_tolerance
        self.position_weight = position_weight
        self.orientation_weight = orientation_weight
        self.use_nullspace = use_nullspace
        self.nullspace_gain = nullspace_gain
        self.nullspace_posture = nullspace_posture

        self._actuator_ids: Optional[np.ndarray] = None
        self._joint_ids: Optional[np.ndarray] = None
        self._qpos_addrs: Optional[np.ndarray] = None
        self._ctrl_low: Optional[np.ndarray] = None
        self._ctrl_high: Optional[np.ndarray] = None
        self._site_id: Optional[int] = None
        self._arm_dof_addrs: Optional[np.ndarray] = None
        self._ik_data: Optional[mujoco.MjData] = None
        self._last_ik_error = np.zeros(6, dtype=np.float32)
        self._last_ik_iterations = 0
        self._prev_target_q: Optional[np.ndarray] = None
        self._filtered_velocity: Optional[np.ndarray] = None
        self._prev_ee_target: Optional[np.ndarray] = None
        self._dt = 0.002
        self._jacp: Optional[np.ndarray] = None
        self._jacr: Optional[np.ndarray] = None
        self._action_space = self._unbound_action_space()

    @property
    def action_size(self) -> int:
        return 7 if self.control_mode == "ik" else len(self.actuator_names)

    @property
    def position_action_size(self) -> int:
        return len(self.actuator_names)

    @property
    def action_space(self) -> spaces.Box:
        return self._action_space

    @property
    def actuator_ids(self) -> np.ndarray:
        if self._actuator_ids is None:
            raise RuntimeError("ArmPositionIkController.bind() must be called first.")
        return self._actuator_ids

    @property
    def qpos_addrs(self) -> np.ndarray:
        if self._qpos_addrs is None:
            raise RuntimeError("ArmPositionIkController.bind() must be called first.")
        return self._qpos_addrs

    @property
    def ctrl_low(self) -> np.ndarray:
        if self._ctrl_low is None:
            raise RuntimeError("ArmPositionIkController.bind() must be called first.")
        return self._ctrl_low

    @property
    def ctrl_high(self) -> np.ndarray:
        if self._ctrl_high is None:
            raise RuntimeError("ArmPositionIkController.bind() must be called first.")
        return self._ctrl_high

    @property
    def site_id(self) -> int:
        if self._site_id is None:
            raise RuntimeError("ArmPositionIkController.bind() must be called first.")
        return self._site_id

    @property
    def arm_dof_addrs(self) -> np.ndarray:
        if self._arm_dof_addrs is None:
            raise RuntimeError("ArmPositionIkController.bind() must be called first.")
        return self._arm_dof_addrs

    @property
    def ik_data(self) -> mujoco.MjData:
        if self._ik_data is None:
            raise RuntimeError("ArmPositionIkController.bind() must be called first.")
        return self._ik_data

    def set_timestep(self, dt: float) -> None:
        if dt <= 0.0:
            raise ValueError(f"Controller timestep must be positive, got {dt}.")
        self._dt = dt

    def set_control_mode(self, control_mode: str) -> None:
        self.control_mode = _validate_mode(control_mode)
        self._action_space = (
            self._bound_action_space()
            if self._ctrl_low is not None
            else self._unbound_action_space()
        )

    def bind(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        self._actuator_ids = _actuator_ids_or_raise(
            model,
            self.actuator_names,
            owner=f"{self.arm_descriptor.name} arm controller",
        )
        self._joint_ids = model.actuator_trnid[self._actuator_ids, 0].astype(np.int32)
        self._qpos_addrs = model.jnt_qposadr[self._joint_ids].astype(np.int32)
        ctrlrange = model.actuator_ctrlrange[self._actuator_ids].astype(np.float32)
        self._ctrl_low = ctrlrange[:, 0].copy()
        self._ctrl_high = ctrlrange[:, 1].copy()

        site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, self.ee_site_name)
        if site_id < 0:
            available = [
                mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, idx)
                for idx in range(model.nsite)
            ]
            raise ValueError(
                f"Missing end-effector site {self.ee_site_name!r}. Available sites: {available}"
            )
        self._site_id = site_id
        self._arm_dof_addrs = model.jnt_dofadr[self._joint_ids].astype(np.int32)
        self._ik_data = mujoco.MjData(model)
        self._jacp = np.zeros((3, model.nv), dtype=np.float64)
        self._jacr = np.zeros((3, model.nv), dtype=np.float64)
        self._prev_target_q = data.qpos[self.qpos_addrs].copy().astype(np.float64)
        self._filtered_velocity = np.zeros(self.position_action_size, dtype=np.float64)
        if self.nullspace_posture is None:
            self.nullspace_posture = 0.5 * (self.ctrl_low + self.ctrl_high).astype(np.float64)
        self._action_space = self._bound_action_space()

    def reset(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        *,
        rng: np.random.Generator,
        options: Optional[dict],
    ) -> Dict[str, Any]:
        _ = rng
        _ = options
        if self.reset_to_current_position:
            target = np.clip(data.qpos[self.qpos_addrs], self.ctrl_low, self.ctrl_high)
        else:
            target = np.clip(
                np.zeros(self.position_action_size, dtype=np.float32),
                self.ctrl_low,
                self.ctrl_high,
            )
        data.ctrl[self.actuator_ids] = target
        mujoco.mj_forward(model, data)

        self._prev_target_q = data.qpos[self.qpos_addrs].copy().astype(np.float64)
        if self._filtered_velocity is None:
            self._filtered_velocity = np.zeros(self.position_action_size, dtype=np.float64)
        else:
            self._filtered_velocity.fill(0.0)
        self._prev_ee_target = None

        return {
            "arm_controller": self.arm_descriptor.name,
            "control_mode": self.control_mode,
            "arm_position_target": target.astype(np.float32).copy(),
            "arm_position_actuators": self.actuator_names,
            "ik_site": self.ee_site_name,
            "ee_position": data.site_xpos[self.site_id].astype(np.float32).copy(),
            "ee_quat": mat_to_quat(data.site_xmat[self.site_id]).astype(np.float32),
        }

    def apply_action(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        action: Any,
    ) -> Dict[str, Any]:
        if self.control_mode == "position":
            return self.apply_position_action(model, data, action)
        return self.apply_ik_action(model, data, action)

    def apply_position_action(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        action: Any,
    ) -> Dict[str, Any]:
        _ = model
        target = self._coerce_position_action(action)
        data.ctrl[self.actuator_ids] = target
        return {
            "arm_controller": self.arm_descriptor.name,
            "control_mode": "position",
            "arm_position_target": target.astype(np.float32).copy(),
        }

    def apply_ik_action(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        action: Any,
    ) -> Dict[str, Any]:
        action_arr = np.asarray(action, dtype=np.float32)
        if action_arr.shape != (7,):
            raise ValueError(f"Arm IK action must have shape (7,), got {action_arr.shape}.")

        target_pos = action_arr[:3].astype(np.float64)
        target_quat = normalize_quat(action_arr[3:7].astype(np.float64))
        if self._prev_ee_target is not None:
            alpha = np.clip(self.target_filter_alpha, 0.0, 1.0)
            target_pos = alpha * target_pos + (1.0 - alpha) * self._prev_ee_target[:3]
            target_quat = self._slerp(self._prev_ee_target[3:7], target_quat, alpha)
        self._prev_ee_target = np.concatenate([target_pos, target_quat])

        arm_target = self._solve_ik(model, data, target_pos, target_quat)
        data.ctrl[self.actuator_ids] = arm_target
        mujoco.mj_forward(model, data)
        return {
            "arm_controller": self.arm_descriptor.name,
            "control_mode": "ik",
            "ik_site": self.ee_site_name,
            "ik_target_position": target_pos.astype(np.float32),
            "ik_target_quat": target_quat.astype(np.float32),
            "ik_error": self._last_ik_error.copy(),
            "ik_iterations": self._last_ik_iterations,
            "arm_position_target": arm_target.copy(),
        }

    def current_action(self, model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
        if self.control_mode == "ik":
            return self.current_ik_action(model, data)
        _ = model
        return data.ctrl[self.actuator_ids].astype(np.float32).copy()

    def current_ik_action(self, model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
        mujoco.mj_forward(model, data)
        ee_pos = data.site_xpos[self.site_id].astype(np.float32)
        ee_quat = mat_to_quat(data.site_xmat[self.site_id]).astype(np.float32)
        return np.concatenate([ee_pos, ee_quat]).astype(np.float32)

    def ik_action_layout(self) -> tuple[str, ...]:
        return IK_ACTION_LAYOUT

    def _solve_ik(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        target_pos: np.ndarray,
        target_quat: np.ndarray,
    ) -> np.ndarray:
        tmp = self.ik_data
        tmp.qpos[:] = data.qpos
        tmp.qvel[:] = data.qvel
        tmp.ctrl[:] = data.ctrl
        tmp.time = data.time

        qpos_addrs = self.qpos_addrs
        current_q = data.qpos[qpos_addrs].copy().astype(np.float64)
        if self._prev_target_q is not None:
            tmp.qpos[qpos_addrs] = self._prev_target_q
            mujoco.mj_forward(model, tmp)

        q = tmp.qpos[qpos_addrs].copy()
        low = self.ctrl_low.astype(np.float64)
        high = self.ctrl_high.astype(np.float64)
        jacp, jacr = self._jacp, self._jacr
        if jacp is None or jacr is None:
            raise RuntimeError("ArmPositionIkController.bind() must be called first.")
        eye = np.eye(self.position_action_size, dtype=np.float64)

        for iteration in range(1, self.ik_iterations + 1):
            tmp.qpos[qpos_addrs] = q
            mujoco.mj_forward(model, tmp)

            pos_error = target_pos - tmp.site_xpos[self.site_id]
            current_quat = mat_to_quat(tmp.site_xmat[self.site_id])
            quat_error = quat_multiply(target_quat, quat_conjugate(current_quat))
            rot_error = quat_to_rotvec(quat_error)
            self._last_ik_error = np.concatenate([pos_error, rot_error]).astype(np.float32)
            self._last_ik_iterations = iteration

            if (
                np.linalg.norm(pos_error) <= self.position_tolerance
                and np.linalg.norm(rot_error) <= self.orientation_tolerance
            ):
                break

            mujoco.mj_jacSite(model, tmp, jacp, jacr, self.site_id)
            jac = np.vstack(
                [
                    self.position_weight * jacp[:, self.arm_dof_addrs],
                    self.orientation_weight * jacr[:, self.arm_dof_addrs],
                ]
            )
            error = np.concatenate(
                [self.position_weight * pos_error, self.orientation_weight * rot_error]
            )

            try:
                u, s, vh = np.linalg.svd(jac, full_matrices=False)
            except np.linalg.LinAlgError:
                dq = jac.T @ np.linalg.solve(
                    jac @ jac.T + (self.damping**2) * np.eye(6, dtype=np.float64),
                    error,
                )
            else:
                adaptive_damp = self._adaptive_damping(s)
                s_damped = s / (s**2 + adaptive_damp**2)
                dq_primary = vh.T @ (s_damped * (u.T @ error))
                if self.use_nullspace:
                    j_inv = vh.T @ np.diag(s_damped) @ u.T
                    null_proj = eye - j_inv @ jac
                    posture_error = self.nullspace_posture - q
                    dq = dq_primary + null_proj @ (self.nullspace_gain * posture_error)
                else:
                    dq = dq_primary

            q = np.clip(q + np.clip(dq, -self.max_joint_step, self.max_joint_step), low, high)

        q = self._smooth_joint_target(current_q, q, low, high)
        tmp.qpos[qpos_addrs] = q
        mujoco.mj_forward(model, tmp)

        pos_error = target_pos - tmp.site_xpos[self.site_id]
        current_quat = mat_to_quat(tmp.site_xmat[self.site_id])
        quat_error = quat_multiply(target_quat, quat_conjugate(current_quat))
        rot_error = quat_to_rotvec(quat_error)
        self._last_ik_error = np.concatenate([pos_error, rot_error]).astype(np.float32)
        self._prev_target_q = q.copy()
        return q.astype(np.float32)

    def _adaptive_damping(self, singular_values: np.ndarray) -> float:
        if not self.damping_adaptive:
            return self.damping
        cond = singular_values[0] / (singular_values[-1] + 1e-10)
        if cond > 1e3 or singular_values[-1] < self.damping_singular_threshold:
            return min(self.damping_max, self.damping * (1.0 + np.log(cond) / 10.0))
        return max(self.damping_min, self.damping / (1.0 + 1.0 / cond))

    def _smooth_joint_target(
        self,
        current_q: np.ndarray,
        solved_q: np.ndarray,
        low: np.ndarray,
        high: np.ndarray,
    ) -> np.ndarray:
        if self._prev_target_q is None:
            self._prev_target_q = current_q.copy()

        desired_velocity = (solved_q - self._prev_target_q) / self._dt
        if self._filtered_velocity is None:
            self._filtered_velocity = desired_velocity
        else:
            alpha = np.clip(self.velocity_filter_alpha, 0.0, 1.0)
            self._filtered_velocity = (
                alpha * desired_velocity + (1.0 - alpha) * self._filtered_velocity
            )

        max_velocity = abs(self.max_joint_velocity)
        filtered_velocity = np.clip(self._filtered_velocity, -max_velocity, max_velocity)
        limited_q = self._prev_target_q + filtered_velocity * self._dt
        return np.clip(limited_q, low, high)

    @staticmethod
    def _slerp(q1: np.ndarray, q2: np.ndarray, t: float) -> np.ndarray:
        dot = np.clip(np.dot(q1, q2), -1.0, 1.0)
        if dot < 0.0:
            q2 = -q2
            dot = -dot
        if dot > 0.9995:
            return normalize_quat(q1 + t * (q2 - q1))

        theta_0 = np.arccos(dot)
        theta = theta_0 * t
        sin_theta = np.sin(theta)
        sin_theta_0 = np.sin(theta_0)
        s1 = np.cos(theta) - dot * sin_theta / sin_theta_0
        s2 = sin_theta / sin_theta_0
        return normalize_quat(s1 * q1 + s2 * q2)

    def _coerce_position_action(self, action: Any) -> np.ndarray:
        action_arr = np.asarray(action, dtype=np.float32)
        expected_shape = (self.position_action_size,)
        if action_arr.shape != expected_shape:
            raise ValueError(
                f"{self.arm_descriptor.name} arm position action must have shape "
                f"{expected_shape}, got {action_arr.shape}."
            )
        if self.normalized_position:
            clipped = np.clip(action_arr, -1.0, 1.0)
            midpoint = 0.5 * (self.ctrl_low + self.ctrl_high)
            half_range = 0.5 * (self.ctrl_high - self.ctrl_low)
            return (midpoint + clipped * half_range).astype(np.float32)
        return np.clip(action_arr, self.ctrl_low, self.ctrl_high).astype(np.float32)

    def _bound_action_space(self) -> spaces.Box:
        if self.control_mode == "ik":
            return spaces.Box(
                low=np.concatenate(
                    [
                        np.full(3, -np.inf, dtype=np.float32),
                        np.full(4, -1.0, dtype=np.float32),
                    ]
                ),
                high=np.concatenate(
                    [
                        np.full(3, np.inf, dtype=np.float32),
                        np.full(4, 1.0, dtype=np.float32),
                    ]
                ),
                dtype=np.float32,
            )
        if self.normalized_position:
            return spaces.Box(
                low=-np.ones(self.position_action_size, dtype=np.float32),
                high=np.ones(self.position_action_size, dtype=np.float32),
                dtype=np.float32,
            )
        return spaces.Box(
            low=self.ctrl_low.astype(np.float32),
            high=self.ctrl_high.astype(np.float32),
            dtype=np.float32,
        )

    def _unbound_action_space(self) -> spaces.Box:
        if self.control_mode == "ik":
            return spaces.Box(
                low=np.concatenate(
                    [
                        np.full(3, -np.inf, dtype=np.float32),
                        np.full(4, -1.0, dtype=np.float32),
                    ]
                ),
                high=np.concatenate(
                    [
                        np.full(3, np.inf, dtype=np.float32),
                        np.full(4, 1.0, dtype=np.float32),
                    ]
                ),
                dtype=np.float32,
            )
        if self.normalized_position:
            return spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(self.position_action_size,),
                dtype=np.float32,
            )
        return spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.position_action_size,),
            dtype=np.float32,
        )
