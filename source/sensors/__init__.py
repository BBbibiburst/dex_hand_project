# -*- coding: utf-8 -*-
"""Sensor contracts and implementations.

The base Gym-facing sensor types are loaded lazily so utility subpackages such as ``source.sensors.tactile.surface_fitting`` can be imported without
importing Gymnasium first.
"""

_BASE_EXPORTS = {"NullTactileSensor", "TactileSensorBase"}

__all__ = sorted(_BASE_EXPORTS)


def __getattr__(name: str):
    if name in _BASE_EXPORTS:
        from source.sensors import base

        return getattr(base, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
