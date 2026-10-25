"""Deployable Cartesian policy-action contracts and continuous RM75B IK."""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from source.geometry import (
    mat_to_quat,
    normalize_quat,
    quat_conjugate,
    quat_multiply,
    quat_to_rotvec,
)

JOINT13 = "joint13"
XYZ9 = "xyz9"
POSE12 = "pose12"


def quat_from_rotvec(rotation: np.ndarray) -> np.ndarray:
    """Convert a radians rotation vector to a wxyz quaternion."""
    vector = np.asarray(rotation, dtype=np.float64).reshape(3)
    angle = float(np.linalg.norm(vector))
    if angle < 1e-10:
        return normalize_quat(np.r_[1.0, 0.5 * vector])
    half = 0.5 * angle
    return np.r_[np.cos(half), np.sin(half) * vector / angle]


def left_rotation_delta(anchor_wxyz: np.ndarray, target_wxyz: np.ndarray) -> np.ndarray:
    """Return r where target = exp(r) * anchor (world/base-frame left delta)."""
    return quat_to_rotvec(quat_multiply(target_wxyz, quat_conjugate(anchor_wxyz)))


@dataclass(frozen=True)
class CartesianActionContract:
    name: str
    frame: str = "robot_base"
    quaternion_order: str = "wxyz"
    position_unit: str = "metre"
    rotation_unit: str = "radian"
    rotation_composition: str = "left: target = exp(delta) * anchor"
    chunk_anchor: str = "one measured pre-inference pose shared by the complete chunk"

    @property
    def action_dim(self) -> int:
        return {JOINT13: 13, XYZ9: 9, POSE12: 12}[self.name]


@dataclass
class IKResult:
    joint_target: np.ndarray
    position_error: float
    orientation_error: float
    joint_delta: float
    minimum_singular_value: float
    iterations: int
    success: bool
    limit_violation: bool


class ContinuousCartesianAdapter:
    """Translate one anchored Cartesian action into a continuous arm target.

    The online seed is the previous commanded joint target (or current measured
    arm qpos on reset). No expert future joint state is accepted by this API.
    """

    def __init__(self, env, *, max_joint_delta: float = 0.20, max_xyz_delta: float = 0.05):
        self.env = env
        self.arm = env.controller.arm_controller
        self.max_joint_delta = float(max_joint_delta)
        self.max_xyz_delta = float(max_xyz_delta)
        self.data = mujoco.MjData(env.model)
        self.previous_joint_target: np.ndarray | None = None
        self._jacp = np.zeros((3, env.model.nv), dtype=np.float64)
        self._jacr = np.zeros((3, env.model.nv), dtype=np.float64)

    def reset(self, measured_qpos: np.ndarray | None = None) -> None:
        q = self.env.data.qpos[self.arm.qpos_addrs] if measured_qpos is None else measured_qpos
        self.previous_joint_target = np.asarray(q, dtype=np.float64).copy()

    def pose_for_joint_target(self, arm_target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        self.data.qpos[:] = self.env.data.qpos
        self.data.qvel[:] = 0.0
        self.data.qpos[self.arm.qpos_addrs] = np.asarray(arm_target, dtype=np.float64)
        mujoco.mj_forward(self.env.model, self.data)
        return (
            self.data.site_xpos[self.arm.site_id].copy(),
            mat_to_quat(self.data.site_xmat[self.arm.site_id]),
        )

    def measured_pose(self) -> tuple[np.ndarray, np.ndarray]:
        mujoco.mj_forward(self.env.model, self.env.data)
        return (
            self.env.data.site_xpos[self.arm.site_id].copy(),
            mat_to_quat(self.env.data.site_xmat[self.arm.site_id]),
        )

    def encode(
        self, joint13: np.ndarray, anchor_pose: tuple[np.ndarray, np.ndarray], mode: str
    ) -> np.ndarray:
        action = np.asarray(joint13, dtype=np.float64)
        target_position, target_quat = self.pose_for_joint_target(action[:7])
        anchor_position, anchor_quat = anchor_pose
        if mode == XYZ9:
            return np.r_[target_position - anchor_position, action[7:]].astype(np.float32)
        if mode == POSE12:
            return np.r_[
                target_position - anchor_position,
                left_rotation_delta(anchor_quat, target_quat),
                action[7:],
            ].astype(np.float32)
        if mode == JOINT13:
            return action.astype(np.float32)
        raise ValueError(f"Unknown action contract: {mode}")

    def decode(
        self,
        action: np.ndarray,
        anchor_pose: tuple[np.ndarray, np.ndarray],
        mode: str,
        *,
        fixed_orientation_wxyz: np.ndarray | None = None,
    ) -> tuple[np.ndarray, IKResult]:
        value = np.asarray(action, dtype=np.float64)
        if mode == JOINT13:
            q = value[:7].copy()
            return value.astype(np.float32), IKResult(q, 0, 0, 0, np.nan, 0, True, False)
        anchor_position, anchor_quat = anchor_pose
        delta = value[:3]
        if np.linalg.norm(delta) > self.max_xyz_delta:
            delta = delta * (self.max_xyz_delta / np.linalg.norm(delta))
        target_position = anchor_position + delta
        if mode == XYZ9:
            target_quat = anchor_quat if fixed_orientation_wxyz is None else fixed_orientation_wxyz
            hand = value[3:]
        elif mode == POSE12:
            target_quat = quat_multiply(quat_from_rotvec(value[3:6]), anchor_quat)
            hand = value[6:]
        else:
            raise ValueError(f"Unknown action contract: {mode}")
        result = self.solve(target_position, target_quat)
        return np.r_[result.joint_target, hand].astype(np.float32), result

    def solve(self, target_position: np.ndarray, target_quat: np.ndarray) -> IKResult:
        seed = (
            self.env.data.qpos[self.arm.qpos_addrs].copy()
            if self.previous_joint_target is None
            else self.previous_joint_target.copy()
        )
        q = seed.copy()
        low = self.arm.ctrl_low.astype(np.float64)
        high = self.arm.ctrl_high.astype(np.float64)
        minimum_singular = np.inf
        position_error = orientation_error = np.inf
        iterations = 0
        for iterations in range(1, 81):
            self.data.qpos[:] = self.env.data.qpos
            self.data.qvel[:] = 0.0
            self.data.qpos[self.arm.qpos_addrs] = q
            mujoco.mj_forward(self.env.model, self.data)
            pos_error = np.asarray(target_position) - self.data.site_xpos[self.arm.site_id]
            current_quat = mat_to_quat(self.data.site_xmat[self.arm.site_id])
            rot_error = quat_to_rotvec(quat_multiply(target_quat, quat_conjugate(current_quat)))
            position_error = float(np.linalg.norm(pos_error))
            orientation_error = float(np.linalg.norm(rot_error))
            if position_error <= 2e-4 and orientation_error <= 2e-3:
                break
            mujoco.mj_jacSite(self.env.model, self.data, self._jacp, self._jacr, self.arm.site_id)
            jac = np.vstack(
                [self._jacp[:, self.arm.arm_dof_addrs], 0.5 * self._jacr[:, self.arm.arm_dof_addrs]]
            )
            error = np.r_[pos_error, 0.5 * rot_error]
            u, singular, vh = np.linalg.svd(jac, full_matrices=False)
            minimum_singular = min(minimum_singular, float(singular[-1]))
            damping = 1e-3 if singular[-1] >= 0.03 else 1e-2
            dq = vh.T @ ((singular / (singular**2 + damping**2)) * (u.T @ error))
            # The redundant null direction stays near the online seed, not an oracle future q.
            inverse = vh.T @ np.diag(singular / (singular**2 + damping**2)) @ u.T
            dq += (np.eye(7) - inverse @ jac) @ (0.05 * (seed - q))
            q = np.clip(q + np.clip(dq, -0.08, 0.08), low, high)
        raw_delta = float(np.max(np.abs(q - seed)))
        if raw_delta > self.max_joint_delta:
            q = seed + (q - seed) * (self.max_joint_delta / raw_delta)
        q = np.clip(q, low, high)
        limit_violation = bool(np.any(q < low - 1e-8) or np.any(q > high + 1e-8))
        success = position_error <= 1e-3 and orientation_error <= 1e-2 and not limit_violation
        if success:
            self.previous_joint_target = q.copy()
        return IKResult(
            joint_target=q.astype(np.float32),
            position_error=position_error,
            orientation_error=orientation_error,
            joint_delta=float(np.max(np.abs(q - seed))),
            minimum_singular_value=float(minimum_singular),
            iterations=iterations,
            success=success,
            limit_violation=limit_violation,
        )


class CartesianChunkDataset:
    """Wrap a Joint13 LeRobot episode with shared-anchor Cartesian chunks."""

    def __init__(self, base_dataset, env, mode: str, *, fixed_orientation_wxyz=None):
        import torch

        self.base = base_dataset
        self.mode = mode
        adapter = ContinuousCartesianAdapter(env)
        fixed = None if fixed_orientation_wxyz is None else np.asarray(fixed_orientation_wxyz)
        labels = []
        # Build labels once. Each horizon shares the measured pose at sample t.
        episode_rows = base_dataset.episode_indices()
        episode_end = {index: rows[-1] for rows in episode_rows.values() for index in rows}
        raw = base_dataset.dataset.hf_dataset
        horizon = len(base_dataset.dataset.delta_timestamps["action"])
        for index in range(len(base_dataset)):
            state = np.asarray(raw[index]["observation.state"], dtype=np.float32)
            env.data.qpos[adapter.arm.qpos_addrs] = state[:7]
            mujoco.mj_forward(env.model, env.data)
            anchor = adapter.measured_pose()
            chunk = []
            final = episode_end[index]
            future_actions = [
                np.asarray(raw[min(index + offset, final)]["action"], dtype=np.float32)
                for offset in range(horizon)
            ]
            for action in future_actions:
                if mode == XYZ9:
                    target_position, _ = adapter.pose_for_joint_target(action[:7])
                    orientation = anchor[1] if fixed is None else fixed
                    # XYZ has no rotation label; orientation is recorded in metadata/executor.
                    _ = orientation
                    chunk.append(np.r_[target_position - anchor[0], action[7:]])
                elif mode == POSE12:
                    target_position, target_quat = adapter.pose_for_joint_target(action[:7])
                    chunk.append(
                        np.r_[
                            target_position - anchor[0],
                            left_rotation_delta(anchor[1], target_quat),
                            action[7:],
                        ]
                    )
                else:
                    raise ValueError(f"Cartesian dataset requires xyz9 or pose12, got {mode}")
            labels.append(np.asarray(chunk, dtype=np.float32))
        self.labels = torch.from_numpy(np.asarray(labels, dtype=np.float32))

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        sample = self.base[index]
        sample["action"] = self.labels[index]
        return sample

    def episode_indices(self):
        return self.base.episode_indices()
