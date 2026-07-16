"""Coordinate transforms shared by Vive visualization and robot control."""

from __future__ import annotations

import math

import numpy as np

from source.geometry import normalize_quat


# OpenVR/SteamVR -> physical workspace: right (+X), forward (+Y), up (+Z).
STEAMVR_TO_WORKSPACE = np.asarray(
    [[-1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]]
)

# Fixed tracker mounting extrinsic: tracker frame -> physical hand frame.
TRACKER_TO_HAND_ROTATION = np.asarray(
    [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]]
)


def quaternion_to_rotation_matrix(quaternion_wxyz) -> np.ndarray:
    w, x, y, z = normalize_quat(np.asarray(quaternion_wxyz, dtype=float))
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def rotation_matrix_to_quaternion_wxyz(rotation: np.ndarray) -> np.ndarray:
    rotation = np.asarray(rotation, dtype=float).reshape(3, 3)
    trace = float(np.trace(rotation))
    if trace > 0:
        scale = math.sqrt(trace + 1.0) * 2.0
        quaternion = np.asarray(
            [0.25 * scale, (rotation[2, 1] - rotation[1, 2]) / scale,
             (rotation[0, 2] - rotation[2, 0]) / scale,
             (rotation[1, 0] - rotation[0, 1]) / scale]
        )
    else:
        index = int(np.argmax(np.diag(rotation)))
        j, k = (index + 1) % 3, (index + 2) % 3
        scale = math.sqrt(1 + rotation[index, index] - rotation[j, j] - rotation[k, k]) * 2
        xyz = np.zeros(3)
        xyz[index] = 0.25 * scale
        xyz[j] = (rotation[j, index] + rotation[index, j]) / scale
        xyz[k] = (rotation[k, index] + rotation[index, k]) / scale
        quaternion = np.asarray([(rotation[k, j] - rotation[j, k]) / scale, *xyz])
    return normalize_quat(quaternion)


def rotation_matrix_to_rpy_degrees(rotation: np.ndarray) -> tuple[float, float, float]:
    sin_pitch = max(-1.0, min(1.0, -float(rotation[2, 0])))
    pitch = math.asin(sin_pitch)
    if abs(math.cos(pitch)) > 1e-7:
        roll = math.atan2(rotation[2, 1], rotation[2, 2])
        yaw = math.atan2(rotation[1, 0], rotation[0, 0])
    else:
        roll = math.atan2(-rotation[1, 2], rotation[1, 1])
        yaw = 0.0
    return tuple(math.degrees(angle) for angle in (roll, pitch, yaw))


def remap_pose(position, quaternion_wxyz) -> tuple[np.ndarray, np.ndarray]:
    position = STEAMVR_TO_WORKSPACE @ np.asarray(position, dtype=float)
    steamvr_rotation = quaternion_to_rotation_matrix(quaternion_wxyz)
    tracker_rotation = STEAMVR_TO_WORKSPACE @ steamvr_rotation @ STEAMVR_TO_WORKSPACE.T
    return position, tracker_rotation @ TRACKER_TO_HAND_ROTATION
