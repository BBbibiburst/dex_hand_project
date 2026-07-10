"""Hardware abstraction points for a 6-channel glove and Vive tracker.

Replace only ``connect`` / ``read`` / ``close`` in the real drivers. The
collector and action mapping deliberately depend on these small protocols.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from source.geometry import normalize_quat, quat_multiply


@dataclass(frozen=True)
class GloveSample:
    stretch: np.ndarray  # normalized six channels, 0=open and 1=flexed
    timestamp: float


@dataclass(frozen=True)
class ViveSample:
    position: np.ndarray
    quaternion_wxyz: np.ndarray
    timestamp: float
    valid: bool = True


class StretchGlove(Protocol):
    def connect(self) -> None: ...
    def read(self) -> GloveSample: ...
    def close(self) -> None: ...


class ViveTracker(Protocol):
    def connect(self) -> None: ...
    def read(self) -> ViveSample: ...
    def close(self) -> None: ...


class MockStretchGlove:
    """Stationary open-hand source used before the hardware API is available."""

    def connect(self) -> None:
        pass

    def read(self) -> GloveSample:
        return GloveSample(np.zeros(6, dtype=np.float32), time.monotonic())

    def close(self) -> None:
        pass


class MockViveTracker:
    """Stationary pose source; call ``set_pose`` from tests or a GUI adapter."""

    def __init__(self) -> None:
        self.position = np.zeros(3, dtype=np.float32)
        self.quaternion_wxyz = np.asarray([1, 0, 0, 0], dtype=np.float32)

    def connect(self) -> None:
        pass

    def set_pose(self, position, quaternion_wxyz) -> None:
        self.position = np.asarray(position, dtype=np.float32).copy()
        self.quaternion_wxyz = np.asarray(quaternion_wxyz, dtype=np.float32).copy()

    def read(self) -> ViveSample:
        return ViveSample(self.position.copy(), self.quaternion_wxyz.copy(), time.monotonic())

    def close(self) -> None:
        pass


class SineStretchGlove:
    """Fake glove API producing six smooth, phase-shifted flexion channels."""

    def __init__(self, *, frequency_hz: float = 0.15, amplitude: float = 0.45) -> None:
        if frequency_hz <= 0:
            raise ValueError("frequency_hz must be positive.")
        if not 0 <= amplitude <= 0.5:
            raise ValueError("amplitude must be in [0, 0.5].")
        self.frequency_hz = float(frequency_hz)
        self.amplitude = float(amplitude)
        self._start = None

    def connect(self) -> None:
        self._start = time.monotonic()

    def read(self) -> GloveSample:
        if self._start is None:
            raise RuntimeError("SineStretchGlove.connect() must be called first.")
        now = time.monotonic()
        phase = 2 * np.pi * self.frequency_hz * (now - self._start)
        offsets = np.linspace(0, np.pi, 6, dtype=np.float32)
        stretch = 0.5 + self.amplitude * np.sin(phase + offsets)
        return GloveSample(np.clip(stretch, 0, 1).astype(np.float32), now)

    def close(self) -> None:
        self._start = None


class SineViveTracker:
    """Fake Vive API producing bounded translation and xyz orientation motion."""

    def __init__(
        self,
        *,
        frequency_hz: float = 0.10,
        translation_amplitude=(0.04, 0.04, 0.03),
        rotation_amplitude_deg=(8.0, 8.0, 12.0),
    ) -> None:
        if frequency_hz <= 0:
            raise ValueError("frequency_hz must be positive.")
        self.frequency_hz = float(frequency_hz)
        self.translation_amplitude = np.asarray(translation_amplitude, dtype=np.float64)
        self.rotation_amplitude = np.deg2rad(rotation_amplitude_deg)
        self._base_position = np.zeros(3, dtype=np.float64)
        self._base_quaternion = np.asarray([1, 0, 0, 0], dtype=np.float64)
        self._start = None

    def connect(self) -> None:
        self._start = time.monotonic()

    def set_pose(self, position, quaternion_wxyz) -> None:
        self._base_position = np.asarray(position, dtype=np.float64).copy()
        q = np.asarray(quaternion_wxyz, dtype=np.float64)
        self._base_quaternion = q / max(np.linalg.norm(q), 1e-9)

    def read(self) -> ViveSample:
        if self._start is None:
            raise RuntimeError("SineViveTracker.connect() must be called first.")
        now = time.monotonic()
        phase = 2 * np.pi * self.frequency_hz * (now - self._start)
        waves = np.sin(phase + np.asarray([0, 2 * np.pi / 3, 4 * np.pi / 3]))
        position = self._base_position + self.translation_amplitude * waves
        angles = self.rotation_amplitude * waves
        cx, cy, cz = np.cos(angles / 2)
        sx, sy, sz = np.sin(angles / 2)
        delta = np.asarray(
            [
                cx * cy * cz + sx * sy * sz,
                sx * cy * cz - cx * sy * sz,
                cx * sy * cz + sx * cy * sz,
                cx * cy * sz - sx * sy * cz,
            ]
        )
        quaternion = normalize_quat(quat_multiply(delta, self._base_quaternion))
        return ViveSample(position.astype(np.float32), quaternion.astype(np.float32), now)

    def close(self) -> None:
        self._start = None


class StretchGloveApiDevice(MockStretchGlove):
    """Placeholder for the real six-channel stretch-glove API."""

    def connect(self) -> None:
        raise NotImplementedError("Implement the stretch-glove transport API here.")


class ViveApiTracker(MockViveTracker):
    """Placeholder for OpenVR/SteamVR pose access."""

    def connect(self) -> None:
        raise NotImplementedError("Implement the Vive/OpenVR transport API here.")
