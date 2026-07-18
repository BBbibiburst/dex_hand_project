"""Shared semantic right-hand skeleton for Vive visualizations."""

from __future__ import annotations

import math

import numpy as np


FINGER_NAMES = ("index", "middle", "ring", "pinky", "thumb")


def _bend_finger(base, lengths, flex) -> np.ndarray:
    points = [np.asarray(base, dtype=float)]
    direction = np.asarray([0.0, 1.0, 0.0])
    angle = 0.0
    flex = float(np.clip(flex, 0.0, 1.0))
    flex = flex * flex * (3.0 - 2.0 * flex)
    joint_angles = np.radians((55.0, 65.0, 45.0)) * flex
    for length, joint_angle in zip(lengths, joint_angles):
        angle += joint_angle
        step = length * (
            math.cos(angle) * direction + math.sin(angle) * np.asarray([0.0, 0.0, 1.0])
        )
        points.append(points[-1] + step)
    return np.asarray(points)


def _bend_thumb(opposition, flex) -> np.ndarray:
    opposition = float(np.clip(opposition, 0.0, 1.0))
    flex = float(np.clip(flex, 0.0, 1.0))
    flex = flex * flex * (3.0 - 2.0 * flex)
    base = np.asarray([0.045, 0.035, 0.000])
    # At opposition=0 the thumb points along local +X (0 degrees). It then
    # rotates toward local +Y, reaching exactly 90 degrees at opposition=1.
    extended_relative = np.asarray(
        [[0.000, 0.000, 0.000], [0.034, 0.000, 0.000], [0.065, 0.000, 0.002], [0.090, 0.000, 0.003]]
    )
    # Flexion shortens the thumb in the palm plane and raises it out of that
    # plane, but deliberately preserves its azimuth. Opposition is handled by
    # the explicit rotation below, so it cannot accidentally exceed 90 deg.
    flexed_relative = np.asarray(
        [[0.000, 0.000, 0.000], [0.025, 0.000, 0.018], [0.035, 0.000, 0.035], [0.040, 0.000, 0.043]]
    )
    relative = extended_relative + flex * (flexed_relative - extended_relative)
    angle = opposition * (0.5 * math.pi)
    rotation = np.asarray(
        [
            [math.cos(angle), -math.sin(angle), 0.0],
            [math.sin(angle), math.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    return base + relative @ rotation.T


def make_hand_lines(flex_values=None) -> list[np.ndarray]:
    """Return palm, wrist, index, middle, ring, pinky and thumb lines."""
    values = np.zeros(5) if flex_values is None else np.asarray(flex_values, dtype=float)
    values = np.clip(values.reshape(-1), 0.0, 1.0)
    if values.shape not in ((5,), (6,)):
        raise ValueError(f"Hand flex values must have five or six channels, got {values.shape}.")
    palm = np.asarray(
        [
            [-0.045, 0.000, 0.000],
            [-0.045, 0.090, 0.000],
            [0.045, 0.090, 0.000],
            [0.045, 0.000, 0.000],
            [-0.045, 0.000, 0.000],
        ]
    )
    wrist = np.asarray([[-0.025, 0.000, 0.000], [0.000, -0.045, 0.000], [0.025, 0.000, 0.000]])
    specs = (
        ((0.020, 0.090, 0.000), (0.036, 0.028, 0.023), values[0]),
        ((0.000, 0.090, 0.000), (0.040, 0.032, 0.026), values[1]),
        ((-0.020, 0.090, 0.000), (0.038, 0.030, 0.024), values[2]),
        ((-0.040, 0.090, 0.000), (0.032, 0.025, 0.020), values[3]),
    )
    fingers = [_bend_finger(base, lengths, flex) for base, lengths, flex in specs]
    thumb_opposition = values[4]
    thumb_flex = values[4] if values.shape == (5,) else values[5]
    return [palm, wrist, *fingers, _bend_thumb(thumb_opposition, thumb_flex)]
