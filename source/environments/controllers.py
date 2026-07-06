# -*- coding: utf-8 -*-
"""Controller for the RM75B arm with dex hand."""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

from gymnasium import spaces
import mujoco
import numpy as np


DEFAULT_EE_SITE_NAME = "right_hand_site"
CONTROL_MODES = ("position", "ik")

ARM_POSITION_ACTUATORS = (
    "pos_joint1",
    "pos_joint2",
    "pos_joint3",
    "pos_joint4",
    "pos_joint5",
    "pos_joint6",
    "pos_joint7",
)

DEX_HAND_POSITION_ACTUATORS = (
    "act_push_0_j",
    "act_push_1_j",
    "act_push_2_j",
    "act_push_3_j",
    "thumb_rotate_act_push_j",
    "thumb_grasp_act_push_j",
)


def prefixed_names(names: Sequence[str], prefix: str = "") -> tuple[str, ...]:
    return tuple(f"{prefix}{name}" for name in names)


def _normalize_quat(quat: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(quat)
    if norm <= 1e-8:
        raise ValueError("Quaternion norm is too small.")
    return quat / norm


def _quat_conjugate(quat: np.ndarray) -> np.ndarray:
    return np.asarray([quat[0], -quat[1], -quat[2], -quat[3]], dtype=np.float64)


def _quat_multiply(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    lw, lx, ly, lz = lhs
    rw, rx, ry, rz = rhs
    return np.asarray(
        [
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ],
        dtype=np.float64,
    )


def _quat_to_rotvec(quat: np.ndarray) -> np.ndarray:
    quat = _normalize_quat(quat)
    if quat[0] < 0.0:
        quat = -quat
    vector = quat[1:]
    vector_norm = np.linalg.norm(vector)
    if vector_norm <= 1e-8:
        return 2.0 * vector
    angle = 2.0 * np.arctan2(vector_norm, quat[0])
    return angle * vector / vector_norm


def _mat_to_quat(mat: np.ndarray) -> np.ndarray:
    quat = np.empty(4, dtype=np.float64)
    mujoco.mju_mat2Quat(quat, mat.reshape(9))
    return _normalize_quat(quat)


class Rm75bDexHandController:
    """Single controller supporting joint position and end-effector IK modes.

    ``position`` action:
        13 actuator targets: 7 arm joint targets + 6 hand actuator targets.

    ``ik`` action:
        ``[x, y, z, qw, qx, qy, qz, hand0, ..., hand5]``.
        The pose controls ``ee_site_name`` and hand entries remain direct
        position targets.
    """

    def __init__(
        self,
        *,
        hand_prefix: str = "",
        control_mode: str = "position",
        ee_site_name: str = DEFAULT_EE_SITE_NAME,
        include_hand_action: bool = True,
        normalized_position: bool = False,
        reset_to_current_position: bool = True,
        ik_iterations: int = 80,
        damping: float = 1e-3,
        max_joint_step: float = 0.15,
        position_tolerance: float = 1e-4,
        orientation_tolerance: float = 1e-3,
        position_weight: float = 1.0,
        orientation_weight: float = 0.35,
    ) -> None:
        self.hand_prefix = hand_prefix
        self.ee_site_name = ee_site_name
        self.include_hand_action = include_hand_action
        self.normalized_position = normalized_position
        self.reset_to_current_position = reset_to_current_position
        self.ik_iterations = ik_iterations
        self.damping = damping
        self.max_joint_step = max_joint_step
        self.position_tolerance = position_tolerance
        self.orientation_tolerance = orientation_tolerance
        self.position_weight = position_weight
        self.orientation_weight = orientation_weight

        self.actuator_names = (
            ARM_POSITION_ACTUATORS
            + prefixed_names(DEX_HAND_POSITION_ACTUATORS, hand_prefix)
        )
        self.control_mode = self._validate_mode(control_mode)

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
        self._action_space = self._unbound_action_space()

    @property
    def action_space(self) -> spaces.Box:
        return self._action_space

    @property
    def actuator_ids(self) -> np.ndarray:
        if self._actuator_ids is None:
            raise RuntimeError("Rm75bDexHandController.bind() must be called first.")
        return self._actuator_ids

    @property
    def joint_ids(self) -> np.ndarray:
        if self._joint_ids is None:
            raise RuntimeError("Rm75bDexHandController.bind() must be called first.")
        return self._joint_ids

    @property
    def qpos_addrs(self) -> np.ndarray:
        if self._qpos_addrs is None:
            raise RuntimeError("Rm75bDexHandController.bind() must be called first.")
        return self._qpos_addrs

    @property
    def ctrl_low(self) -> np.ndarray:
        if self._ctrl_low is None:
            raise RuntimeError("Rm75bDexHandController.bind() must be called first.")
        return self._ctrl_low

    @property
    def ctrl_high(self) -> np.ndarray:
        if self._ctrl_high is None:
            raise RuntimeError("Rm75bDexHandController.bind() must be called first.")
        return self._ctrl_high

    @property
    def site_id(self) -> int:
        if self._site_id is None:
            raise RuntimeError("Rm75bDexHandController.bind() must be called first.")
        return self._site_id

    @property
    def arm_dof_addrs(self) -> np.ndarray:
        if self._arm_dof_addrs is None:
            raise RuntimeError("Rm75bDexHandController.bind() must be called first.")
        return self._arm_dof_addrs

    @property
    def ik_data(self) -> mujoco.MjData:
        if self._ik_data is None:
            raise RuntimeError("Rm75bDexHandController.bind() must be called first.")
        return self._ik_data

    def set_control_mode(self, control_mode: str) -> None:
        self.control_mode = self._validate_mode(control_mode)
        self._action_space = self._bound_action_space() if self._ctrl_low is not None else self._unbound_action_space()

    def bind(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        _ = data
        actuator_ids = []
        missing = []
        for name in self.actuator_names:
            actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            if actuator_id < 0:
                missing.append(name)
            else:
                actuator_ids.append(actuator_id)
        if missing:
            available = [
                mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, idx)
                for idx in range(model.nu)
            ]
            raise ValueError(f"Missing actuator(s): {missing}. Available actuators: {available}")

        self._actuator_ids = np.asarray(actuator_ids, dtype=np.int32)
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
            raise ValueError(f"Missing end-effector site {self.ee_site_name!r}. Available sites: {available}")
        self._site_id = site_id
        self._arm_dof_addrs = model.jnt_dofadr[self.joint_ids[: len(ARM_POSITION_ACTUATORS)]]
        self._ik_data = mujoco.MjData(model)
        self._action_space = self._bound_action_space()

    def reset(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        *,
        rng: np.random.Generator,
        options: Optional[dict],
    ) -> Dict[str, Any]:
        _ = model
        _ = rng
        _ = options
        if self.reset_to_current_position:
            target = np.clip(data.qpos[self.qpos_addrs], self.ctrl_low, self.ctrl_high)
        else:
            target = np.clip(
                np.zeros(len(self.actuator_names), dtype=np.float32),
                self.ctrl_low,
                self.ctrl_high,
            )
        data.ctrl[self.actuator_ids] = target
        mujoco.mj_forward(model, data)
        return {
            "controller": "rm75b_dex_hand",
            "control_mode": self.control_mode,
            "position_target": target.astype(np.float32).copy(),
            "position_actuators": self.actuator_names,
            "ik_site": self.ee_site_name,
            "ik_action_layout": self.ik_action_layout(),
            "ee_position": data.site_xpos[self.site_id].astype(np.float32).copy(),
            "ee_quat": _mat_to_quat(data.site_xmat[self.site_id]).astype(np.float32),
        }

    def apply_action(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        action: Any,
    ) -> Dict[str, Any]:
        if self.control_mode == "position":
            return self._apply_position_action(model, data, action)
        return self._apply_ik_action(model, data, action)

    def current_action(self, model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
        if self.control_mode == "position":
            return data.ctrl[self.actuator_ids].astype(np.float32).copy()
        return self.current_ik_action(model, data)

    def current_ik_action(self, model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
        _ = model
        mujoco.mj_forward(model, data)
        ee_pos = data.site_xpos[self.site_id].astype(np.float32)
        ee_quat = _mat_to_quat(data.site_xmat[self.site_id]).astype(np.float32)
        if not self.include_hand_action:
            return np.concatenate([ee_pos, ee_quat]).astype(np.float32)
        hand_ctrl = data.ctrl[self.actuator_ids[len(ARM_POSITION_ACTUATORS) :]].astype(np.float32)
        return np.concatenate([ee_pos, ee_quat, hand_ctrl]).astype(np.float32)

    def ik_action_layout(self) -> tuple[str, ...]:
        layout = ("x", "y", "z", "qw", "qx", "qy", "qz")
        if self.include_hand_action:
            layout += DEX_HAND_POSITION_ACTUATORS
        return layout

    def _apply_position_action(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        action: Any,
    ) -> Dict[str, Any]:
        _ = model
        target = self._coerce_position_action(action)
        data.ctrl[self.actuator_ids] = target
        return {
            "controller": "rm75b_dex_hand",
            "control_mode": "position",
            "position_target": target.astype(np.float32).copy(),
        }

    def _apply_ik_action(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        action: Any,
    ) -> Dict[str, Any]:
        action_arr = np.asarray(action, dtype=np.float32)
        expected_shape = (7 + self._hand_action_size(),)
        if action_arr.shape != expected_shape:
            raise ValueError(f"IK action must have shape {expected_shape}, got {action_arr.shape}.")

        target_pos = action_arr[:3].astype(np.float64)
        target_quat = _normalize_quat(action_arr[3:7].astype(np.float64))
        arm_target = self._solve_ik(model, data, target_pos, target_quat)
        ctrl_target = data.ctrl[self.actuator_ids].astype(np.float32).copy()
        ctrl_target[: len(ARM_POSITION_ACTUATORS)] = arm_target

        if self.include_hand_action:
            hand_low = self.ctrl_low[len(ARM_POSITION_ACTUATORS) :]
            hand_high = self.ctrl_high[len(ARM_POSITION_ACTUATORS) :]
            ctrl_target[len(ARM_POSITION_ACTUATORS) :] = np.clip(
                action_arr[7:],
                hand_low,
                hand_high,
            )
        data.ctrl[self.actuator_ids] = ctrl_target
        mujoco.mj_forward(model, data)
        return {
            "controller": "rm75b_dex_hand",
            "control_mode": "ik",
            "ik_site": self.ee_site_name,
            "ik_target_position": target_pos.astype(np.float32),
            "ik_target_quat": target_quat.astype(np.float32),
            "ik_error": self._last_ik_error.copy(),
            "ik_iterations": self._last_ik_iterations,
            "position_target": ctrl_target.copy(),
        }

    def _solve_ik(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        target_pos: np.ndarray,
        target_quat: np.ndarray,
    ) -> np.ndarray:
        tmp = self.ik_data
        tmp.qpos[:] = data.qpos
        tmp.qvel[:] = 0.0
        tmp.ctrl[:] = data.ctrl

        arm_count = len(ARM_POSITION_ACTUATORS)
        qpos_addrs = self.qpos_addrs[:arm_count]
        q = tmp.qpos[qpos_addrs].copy()
        low = self.ctrl_low[:arm_count].astype(np.float64)
        high = self.ctrl_high[:arm_count].astype(np.float64)
        jacp = np.zeros((3, model.nv), dtype=np.float64)
        jacr = np.zeros((3, model.nv), dtype=np.float64)
        eye6 = np.eye(6, dtype=np.float64)

        for iteration in range(1, self.ik_iterations + 1):
            tmp.qpos[qpos_addrs] = q
            mujoco.mj_forward(model, tmp)
            pos_error = target_pos - tmp.site_xpos[self.site_id]
            current_quat = _mat_to_quat(tmp.site_xmat[self.site_id])
            quat_error = _quat_multiply(target_quat, _quat_conjugate(current_quat))
            rot_error = _quat_to_rotvec(quat_error)
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
                [
                    self.position_weight * pos_error,
                    self.orientation_weight * rot_error,
                ]
            )
            dq = jac.T @ np.linalg.solve(jac @ jac.T + (self.damping**2) * eye6, error)
            q = np.clip(q + np.clip(dq, -self.max_joint_step, self.max_joint_step), low, high)

        return q.astype(np.float32)

    def _coerce_position_action(self, action: Any) -> np.ndarray:
        action_arr = np.asarray(action, dtype=np.float32)
        expected_shape = (len(self.actuator_names),)
        if action_arr.shape != expected_shape:
            raise ValueError(f"Position action must have shape {expected_shape}, got {action_arr.shape}.")
        if self.normalized_position:
            clipped = np.clip(action_arr, -1.0, 1.0)
            midpoint = 0.5 * (self.ctrl_low + self.ctrl_high)
            half_range = 0.5 * (self.ctrl_high - self.ctrl_low)
            return (midpoint + clipped * half_range).astype(np.float32)
        return np.clip(action_arr, self.ctrl_low, self.ctrl_high).astype(np.float32)

    def _bound_action_space(self) -> spaces.Box:
        if self.control_mode == "ik":
            return self._ik_action_space(bound_hand=True)
        if self.normalized_position:
            return spaces.Box(
                low=-np.ones(len(self.actuator_names), dtype=np.float32),
                high=np.ones(len(self.actuator_names), dtype=np.float32),
                dtype=np.float32,
            )
        return spaces.Box(
            low=self.ctrl_low.astype(np.float32),
            high=self.ctrl_high.astype(np.float32),
            dtype=np.float32,
        )

    def _unbound_action_space(self) -> spaces.Box:
        if self.control_mode == "ik":
            return self._ik_action_space(bound_hand=False)
        shape = (len(self.actuator_names),)
        if self.normalized_position:
            return spaces.Box(low=-1.0, high=1.0, shape=shape, dtype=np.float32)
        return spaces.Box(low=-np.inf, high=np.inf, shape=shape, dtype=np.float32)

    def _ik_action_space(self, *, bound_hand: bool) -> spaces.Box:
        hand_size = self._hand_action_size()
        if bound_hand and hand_size:
            hand_low = self.ctrl_low[len(ARM_POSITION_ACTUATORS) :].astype(np.float32)
            hand_high = self.ctrl_high[len(ARM_POSITION_ACTUATORS) :].astype(np.float32)
        else:
            hand_low = np.full(hand_size, -np.inf, dtype=np.float32)
            hand_high = np.full(hand_size, np.inf, dtype=np.float32)
        return spaces.Box(
            low=np.concatenate(
                [
                    np.full(3, -np.inf, dtype=np.float32),
                    np.full(4, -1.0, dtype=np.float32),
                    hand_low,
                ]
            ),
            high=np.concatenate(
                [
                    np.full(3, np.inf, dtype=np.float32),
                    np.full(4, 1.0, dtype=np.float32),
                    hand_high,
                ]
            ),
            dtype=np.float32,
        )

    def _hand_action_size(self) -> int:
        return len(DEX_HAND_POSITION_ACTUATORS) if self.include_hand_action else 0

    @staticmethod
    def _validate_mode(control_mode: str) -> str:
        if control_mode not in CONTROL_MODES:
            raise ValueError(f"control_mode must be one of {CONTROL_MODES}, got {control_mode!r}.")
        return control_mode
