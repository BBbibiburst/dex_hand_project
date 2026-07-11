# -*- coding: utf-8 -*-
"""Shared spatial math utilities (quaternions, rotations, etc.)."""

from __future__ import annotations

import mujoco
import numpy as np


def normalize_quat(quat: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(quat)
    if norm <= 1e-8:
        raise ValueError("Quaternion norm is too small.")
    return quat / norm


def quat_conjugate(quat: np.ndarray) -> np.ndarray:
    return np.asarray([quat[0], -quat[1], -quat[2], -quat[3]], dtype=np.float64)


def quat_multiply(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
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


def quat_to_rotvec(quat: np.ndarray) -> np.ndarray:
    quat = normalize_quat(quat)
    if quat[0] < 0.0:
        quat = -quat
    vector = quat[1:]
    vector_norm = np.linalg.norm(vector)
    if vector_norm <= 1e-8:
        return 2.0 * vector
    angle = 2.0 * np.arctan2(vector_norm, quat[0])
    return angle * vector / vector_norm


def mat_to_quat(mat: np.ndarray) -> np.ndarray:
    quat = np.empty(4, dtype=np.float64)
    mujoco.mju_mat2Quat(quat, mat.reshape(9))
    return normalize_quat(quat)
