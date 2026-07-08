# -*- coding: utf-8 -*-
"""Robot, end-effector, and base descriptors.

Defaults are loaded lazily so importing low-level modules such as
``source.robots.descriptors`` does not trigger built-in descriptor imports.
That keeps controller modules free from circular imports.
"""

__all__ = ["DEFAULT_ARM", "DEFAULT_BASE", "DEFAULT_HAND"]


def __getattr__(name: str):
    if name in __all__:
        from source.robots import defaults

        return getattr(defaults, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
