# -*- coding: utf-8 -*-
"""Reusable processing for array-based tactile signals."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np


def _validate_nonnegative(name: str, value: float) -> float:
    value = float(value)
    if value < 0.0:
        raise ValueError(f"{name} must be non-negative.")
    return value


def _gaussian_kernel1d(sigma: float) -> np.ndarray:
    if sigma <= 0.0:
        return np.asarray([1.0], dtype=np.float64)
    radius = max(1, int(np.ceil(3.0 * sigma)))
    offsets = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (offsets / sigma) ** 2)
    return kernel / kernel.sum()


def _convolve_axis_reflect(values: np.ndarray, kernel: np.ndarray, axis: int) -> np.ndarray:
    if len(kernel) == 1:
        return values.copy()
    radius = len(kernel) // 2
    pad_width = [(0, 0)] * values.ndim
    pad_width[axis] = (radius, radius)
    padded = np.pad(values, pad_width, mode="edge")
    result = np.zeros_like(values, dtype=np.float64)
    for index, weight in enumerate(kernel):
        start = index
        stop = start + values.shape[axis]
        slices = [slice(None)] * values.ndim
        slices[axis] = slice(start, stop)
        result += weight * padded[tuple(slices)]
    return result


def _gaussian_blur(values: np.ndarray, sigma: float) -> np.ndarray:
    kernel = _gaussian_kernel1d(sigma)
    blurred = _convolve_axis_reflect(values, kernel, axis=0)
    return _convolve_axis_reflect(blurred, kernel, axis=1)


def _neighbor_crosstalk(values: np.ndarray, amount: float) -> np.ndarray:
    """Diffuse ``amount`` equally across the available 8-neighborhood."""
    if amount <= 0.0:
        return values.copy()
    amount = float(np.clip(amount, 0.0, 1.0))
    rows, cols = values.shape
    result = (1.0 - amount) * values
    received = np.zeros_like(values, dtype=np.float64)

    # Normalize at the source, not the destination.  A taxel therefore sends
    # an equal share to each of its available neighbors; in the interior this
    # is exactly amount / 8 for all eight surrounding cells.
    source_neighbor_count = np.zeros((rows, cols), dtype=np.float64)
    offsets = tuple(
        (row_offset, col_offset)
        for row_offset in (-1, 0, 1)
        for col_offset in (-1, 0, 1)
        if (row_offset, col_offset) != (0, 0)
    )
    for row_offset, col_offset in offsets:
        source_rows = slice(max(0, -row_offset), rows - max(0, row_offset))
        source_cols = slice(max(0, -col_offset), cols - max(0, col_offset))
        source_neighbor_count[source_rows, source_cols] += 1.0

    for row_offset, col_offset in offsets:
        target_rows = slice(max(0, row_offset), rows + min(0, row_offset))
        target_cols = slice(max(0, col_offset), cols + min(0, col_offset))
        source_rows = slice(max(0, -row_offset), rows - max(0, row_offset))
        source_cols = slice(max(0, -col_offset), cols - max(0, col_offset))
        source_values = values[source_rows, source_cols]
        counts = source_neighbor_count[source_rows, source_cols]
        received[target_rows, target_cols] += np.divide(
            source_values,
            counts,
            out=np.zeros_like(source_values, dtype=np.float64),
            where=counts > 0.0,
        )

    result += amount * received
    result[source_neighbor_count == 0.0] = values[source_neighbor_count == 0.0]
    return result


@dataclass(frozen=True)
class TaxelPatch:
    name: str
    rows: int
    cols: int
    kind: str
    start: int
    stop: int

    @property
    def shape(self) -> tuple[int, int]:
        return (self.rows, self.cols)


@dataclass(frozen=True)
class TactileSignalProcessorConfig:
    deadzone: float = 0.0
    saturation: float = 1.0
    nonlinear_exponent: float = 1.0
    lowpass_alpha: float = 1.0
    crosstalk: float = 0.0
    gaussian_sigma: float = 0.0
    noise_std: float = 0.0
    normalize: bool = True
    seed: Optional[int] = None

    @classmethod
    def from_mapping(
        cls,
        values: Optional[Mapping[str, Any]],
    ) -> "TactileSignalProcessorConfig":
        if values is None:
            return cls()
        known = {field.name for field in cls.__dataclass_fields__.values()}
        unknown = sorted(set(values) - known)
        if unknown:
            raise ValueError(f"Unknown tactile signal processor option(s): {unknown}")
        return cls(**dict(values))

    def __post_init__(self) -> None:
        _validate_nonnegative("deadzone", self.deadzone)
        if self.saturation <= 0.0:
            raise ValueError("saturation must be positive.")
        if self.nonlinear_exponent <= 0.0:
            raise ValueError("nonlinear_exponent must be positive.")
        if not 0.0 <= self.lowpass_alpha <= 1.0:
            raise ValueError("lowpass_alpha must be in [0, 1].")
        if not 0.0 <= self.crosstalk <= 1.0:
            raise ValueError("crosstalk must be in [0, 1].")
        _validate_nonnegative("gaussian_sigma", self.gaussian_sigma)
        _validate_nonnegative("noise_std", self.noise_std)


class TactileSignalProcessor:
    """Post-process raw MuJoCo touch readings into tactile sensor signals."""

    def __init__(self, config: Optional[Mapping[str, Any]] = None) -> None:
        self.config = TactileSignalProcessorConfig.from_mapping(config)
        self._filtered: Optional[np.ndarray] = None
        self._rng = np.random.default_rng(self.config.seed)

    @property
    def normalized(self) -> bool:
        return self.config.normalize

    def reset(self, size: Optional[int] = None) -> None:
        self._filtered = None if size is None else np.zeros(size, dtype=np.float64)

    def process(self, raw: np.ndarray, patches: Sequence[TaxelPatch]) -> np.ndarray:
        cfg = self.config
        values = np.asarray(raw, dtype=np.float64).copy()
        values = np.maximum(values - cfg.deadzone, 0.0)
        values = np.clip(values, 0.0, cfg.saturation)
        if cfg.nonlinear_exponent != 1.0:
            values = cfg.saturation * (values / cfg.saturation) ** cfg.nonlinear_exponent

        if cfg.lowpass_alpha < 1.0:
            if self._filtered is None or self._filtered.shape != values.shape:
                self._filtered = values.copy()
            else:
                alpha = cfg.lowpass_alpha
                self._filtered = alpha * values + (1.0 - alpha) * self._filtered
            values = self._filtered.copy()

        if cfg.crosstalk > 0.0 or cfg.gaussian_sigma > 0.0:
            values = self._process_patch_images(values, patches)

        if cfg.noise_std > 0.0:
            values += self._rng.normal(0.0, cfg.noise_std, size=values.shape)
            values = np.clip(values, 0.0, cfg.saturation)

        if cfg.normalize:
            values = values / cfg.saturation
        return values.astype(np.float32)

    def _process_patch_images(
        self,
        values: np.ndarray,
        patches: Sequence[TaxelPatch],
    ) -> np.ndarray:
        cfg = self.config
        result = values.copy()
        for patch in patches:
            image = values[patch.start : patch.stop].reshape(patch.shape)
            if cfg.crosstalk > 0.0:
                image = _neighbor_crosstalk(image, cfg.crosstalk)
            if cfg.gaussian_sigma > 0.0:
                image = _gaussian_blur(image, cfg.gaussian_sigma)
            result[patch.start : patch.stop] = image.reshape(-1)
        return np.clip(result, 0.0, cfg.saturation)

    def metadata(self) -> Dict[str, Any]:
        cfg = self.config
        return {
            "deadzone": cfg.deadzone,
            "saturation": cfg.saturation,
            "nonlinear_exponent": cfg.nonlinear_exponent,
            "lowpass_alpha": cfg.lowpass_alpha,
            "crosstalk": cfg.crosstalk,
            "crosstalk_neighbors": 8,
            "gaussian_sigma": cfg.gaussian_sigma,
            "noise_std": cfg.noise_std,
            "normalize": cfg.normalize,
            "seed": cfg.seed,
        }
