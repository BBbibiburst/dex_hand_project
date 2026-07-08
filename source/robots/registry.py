# -*- coding: utf-8 -*-
"""Registry for robot component descriptors.

Adding a new arm / hand / base only requires writing a new descriptor module
and calling the matching ``register_*`` function at import time (see
``source/robots/arms/rm75b.py`` for the pattern). No other module needs to
change.
"""

from __future__ import annotations

from source.robots.descriptors import ArmDescriptor, BaseDescriptor, EndEffectorDescriptor


_ARMS: dict[str, ArmDescriptor] = {}
_HANDS: dict[str, EndEffectorDescriptor] = {}
_BASES: dict[str, BaseDescriptor] = {}
_BUILTINS_LOADED = False


def load_builtin_descriptors() -> None:
    """Import built-in descriptor modules so their register_* calls run."""
    global _BUILTINS_LOADED
    if _BUILTINS_LOADED:
        return

    from source.robots.arms import rm75b as _rm75b
    from source.robots.bases import rethink_minimal_mount as _rethink_minimal_mount
    from source.robots.hands import dex_hand as _dex_hand

    _ = _rm75b, _rethink_minimal_mount, _dex_hand
    _BUILTINS_LOADED = True


def register_arm(descriptor: ArmDescriptor) -> ArmDescriptor:
    _ARMS[descriptor.name] = descriptor
    return descriptor


def register_hand(descriptor: EndEffectorDescriptor) -> EndEffectorDescriptor:
    _HANDS[descriptor.name] = descriptor
    return descriptor


def register_base(descriptor: BaseDescriptor) -> BaseDescriptor:
    _BASES[descriptor.name] = descriptor
    return descriptor


def get_arm(name: str) -> ArmDescriptor:
    load_builtin_descriptors()
    if name not in _ARMS:
        raise KeyError(f"Unknown arm {name!r}. Available arms: {sorted(_ARMS)}")
    return _ARMS[name]


def get_hand(name: str) -> EndEffectorDescriptor:
    load_builtin_descriptors()
    if name not in _HANDS:
        raise KeyError(f"Unknown hand {name!r}. Available hands: {sorted(_HANDS)}")
    return _HANDS[name]


def get_base(name: str) -> BaseDescriptor:
    load_builtin_descriptors()
    if name not in _BASES:
        raise KeyError(f"Unknown base {name!r}. Available bases: {sorted(_BASES)}")
    return _BASES[name]


def registered_arms() -> tuple[str, ...]:
    load_builtin_descriptors()
    return tuple(sorted(_ARMS))


def registered_hands() -> tuple[str, ...]:
    load_builtin_descriptors()
    return tuple(sorted(_HANDS))


def registered_bases() -> tuple[str, ...]:
    load_builtin_descriptors()
    return tuple(sorted(_BASES))
