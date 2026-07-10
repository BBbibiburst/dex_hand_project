# -*- coding: utf-8 -*-
"""Tactile sensor implementations."""

__all__ = ["DEX_HAND_PATCH_LAYOUT", "DexHandTouchSensor", "site_name"]


def __getattr__(name):
    if name in __all__:
        from source.sensors.tactile import dex_hand

        return getattr(dex_hand, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
