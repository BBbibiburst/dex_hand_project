# -*- coding: utf-8 -*-
"""Tactile array processing and concrete sensor implementations."""

from source.sensors.tactile.signal_processing import (
    TactileSignalProcessor,
    TactileSignalProcessorConfig,
    TaxelPatch,
)

_DEX_HAND_EXPORTS = {
    "DEFAULT_TACTILE_BACKEND",
    "SUPPORTED_TACTILE_BACKENDS",
    "DEX_HAND_PATCH_LAYOUT",
    "DexHandTactileSensorBase",
    "DexHandTouchSensor",
    "SimpleBoxTactileSensor",
    "create_dex_hand_tactile_sensor",
    "sensor_name",
    "site_name",
}

__all__ = [
    *_DEX_HAND_EXPORTS,
    "TaxelPatch",
    "TactileSignalProcessor",
    "TactileSignalProcessorConfig",
]


def __getattr__(name: str):
    if name in _DEX_HAND_EXPORTS:
        from source.sensors.tactile import dex_hand

        return getattr(dex_hand, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
