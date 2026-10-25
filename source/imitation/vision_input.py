"""Deployable RGB preprocessing shared by vision-resolution experiments."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from PIL import Image


def resize_rgb_uint8(image: np.ndarray, *, width: int, height: int) -> np.ndarray:
    """Resize one RGB uint8 frame with an explicit, reproducible filter."""

    array = np.asarray(image)
    if array.ndim != 3 or array.shape[2] != 3 or array.dtype != np.uint8:
        raise ValueError("image must be an HxWx3 uint8 RGB array")
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive")
    if array.shape[:2] == (height, width):
        return array.copy()
    return np.asarray(
        Image.fromarray(array, mode="RGB").resize((width, height), Image.Resampling.BILINEAR),
        dtype=np.uint8,
    )


def crop_normalized_rgb(
    image: np.ndarray,
    box_xyxy: Sequence[float],
    *,
    output_width: int,
    output_height: int,
) -> np.ndarray:
    """Crop a fixed normalized camera ROI and resize it for policy input.

    The ROI is a camera-calibration constant, not an online simulator-state
    query. Coordinates are ``(x0, y0, x1, y1)`` in the range ``[0, 1]``.
    """

    array = np.asarray(image)
    if array.ndim != 3 or array.shape[2] != 3 or array.dtype != np.uint8:
        raise ValueError("image must be an HxWx3 uint8 RGB array")
    values = np.asarray(box_xyxy, dtype=np.float64)
    if values.shape != (4,) or not np.all(np.isfinite(values)):
        raise ValueError("box_xyxy must contain four finite values")
    x0, y0, x1, y1 = values
    if not (0.0 <= x0 < x1 <= 1.0 and 0.0 <= y0 < y1 <= 1.0):
        raise ValueError("normalized ROI must satisfy 0 <= min < max <= 1")
    height, width = array.shape[:2]
    left = int(np.floor(x0 * width))
    top = int(np.floor(y0 * height))
    right = min(width, int(np.ceil(x1 * width)))
    bottom = min(height, int(np.ceil(y1 * height)))
    return resize_rgb_uint8(
        array[top:bottom, left:right], width=output_width, height=output_height
    )
