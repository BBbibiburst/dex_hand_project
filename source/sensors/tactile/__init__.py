# -*- coding: utf-8 -*-
"""Tactile sensor implementations."""

__all__ = [
    "DEFAULT_TACTILE_BACKEND",
    "SUPPORTED_TACTILE_BACKENDS",
    "DEX_HAND_PATCH_LAYOUT",
    "DexHandTactileSensorBase",
    "DexHandTouchSensor",
    "SimpleBoxTactileSensor",
    "TactileSignalProcessor",
    "TactileSignalProcessorConfig",
    "create_dex_hand_tactile_sensor",
    "site_name",
]


def __getattr__(name):
    if name in __all__:
        from source.sensors.tactile import dex_hand
        return getattr(dex_hand, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
