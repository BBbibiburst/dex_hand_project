# -*- coding: utf-8 -*-
"""Registries for robot component descriptors."""

from __future__ import annotations

import importlib

from source.registry import Registry
from source.robots.descriptors import ArmDescriptor, BaseDescriptor, EndEffectorDescriptor

_ARMS = Registry[ArmDescriptor]("arm")
_HANDS = Registry[EndEffectorDescriptor]("hand")
_BASES = Registry[BaseDescriptor]("base")
_BUILTIN_DESCRIPTOR_MODULES = (
    "source.robots.arms.rm75b",
    "source.robots.bases.rethink_minimal_mount",
    "source.robots.hands.dex_hand",
    "source.robots.hands.pika_gripper",
)
_BUILTINS_LOADED = False


def load_builtin_descriptors() -> None:
    """Import the explicitly supported built-in descriptors once."""
    global _BUILTINS_LOADED
    if _BUILTINS_LOADED:
        return

    for module_name in _BUILTIN_DESCRIPTOR_MODULES:
        importlib.import_module(module_name)
    _BUILTINS_LOADED = True


def register_arm(descriptor: ArmDescriptor) -> ArmDescriptor:
    return _ARMS.register(descriptor.name, descriptor)


def register_hand(descriptor: EndEffectorDescriptor) -> EndEffectorDescriptor:
    return _HANDS.register(descriptor.name, descriptor)


def register_base(descriptor: BaseDescriptor) -> BaseDescriptor:
    return _BASES.register(descriptor.name, descriptor)


def get_arm(name: str) -> ArmDescriptor:
    load_builtin_descriptors()
    return _ARMS.get(name)


def get_hand(name: str) -> EndEffectorDescriptor:
    load_builtin_descriptors()
    return _HANDS.get(name)


def get_base(name: str) -> BaseDescriptor:
    load_builtin_descriptors()
    return _BASES.get(name)


def registered_arms() -> tuple[str, ...]:
    load_builtin_descriptors()
    return _ARMS.names()


def registered_hands() -> tuple[str, ...]:
    load_builtin_descriptors()
    return _HANDS.names()


def registered_bases() -> tuple[str, ...]:
    load_builtin_descriptors()
    return _BASES.names()
