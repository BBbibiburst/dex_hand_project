"""Shared low-latency processing for normalized glove samples."""

from __future__ import annotations

import numpy as np

from source.teleop.devices import GloveSample


class GloveValueFilter:
    def __init__(self, smoothing: float = 0.70, deadzone: float = 0.03):
        if not 0 < smoothing <= 1:
            raise ValueError("smoothing must be in (0, 1].")
        if not 0 <= deadzone < 0.5:
            raise ValueError("deadzone must be in [0, 0.5).")
        self.smoothing = float(smoothing)
        self.deadzone = float(deadzone)
        self.value = None

    def reset(self) -> None:
        self.value = None

    def update(self, values) -> np.ndarray:
        values = np.clip(np.asarray(values, dtype=np.float32).reshape(-1), 0, 1)
        if values.shape != (6,):
            raise ValueError(f"Glove sample must have six channels, got {values.shape}.")
        if self.deadzone:
            values = np.clip((values - self.deadzone) / (1 - 2 * self.deadzone), 0, 1)
        if self.value is None:
            self.value = values.copy()
        else:
            self.value += self.smoothing * (values - self.value)
        return self.value.copy()


def read_latest_glove(device) -> GloveSample:
    """Discard queued hardware packets and return the freshest available sample."""
    discard = getattr(device, "discard_pending", None)
    if discard is not None:
        discard()
    return device.read()
