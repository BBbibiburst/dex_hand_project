# -*- coding: utf-8 -*-
"""Registries and automatic discovery for robot descriptors."""

from __future__ import annotations

import importlib
import pkgutil

from source.registry import Registry
from source.robots.descriptors import ArmDescriptor, BaseDescriptor, EndEffectorDescriptor

_ARMS = Registry[ArmDescriptor]("arm")
_HANDS = Registry[EndEffectorDescriptor]("hand")
_BASES = Registry[BaseDescriptor]("base")
_BUILTINS_LOADED = False


def _import_package_modules(package_name: str) -> None:
    package = importlib.import_module(package_name)
    for module_info in pkgutil.iter_modules(package.__path__):
        if not module_info.name.startswith("_"):
            importlib.import_module(f"{package_name}.{module_info.name}")


def load_builtin_descriptors() -> None:
    global _BUILTINS_LOADED
    if _BUILTINS_LOADED:
        return
    _BUILTINS_LOADED = True
    for package_name in (
        "source.robots.arms",
        "source.robots.bases",
        "source.robots.hands",
    ):
        _import_package_modules(package_name)


def register_arm(descriptor: ArmDescriptor) -> ArmDescriptor:
    return _ARMS.register(descriptor.name, descriptor)

def register_hand(descriptor: EndEffectorDescriptor) -> EndEffectorDescriptor:
    return _HANDS.register(descriptor.name, descriptor)

def register_base(descriptor: BaseDescriptor) -> BaseDescriptor:
    return _BASES.register(descriptor.name, descriptor)

def get_arm(name: str) -> ArmDescriptor:
    load_builtin_descriptors(); return _ARMS.get(name)

def get_hand(name: str) -> EndEffectorDescriptor:
    load_builtin_descriptors(); return _HANDS.get(name)

def get_base(name: str) -> BaseDescriptor:
    load_builtin_descriptors(); return _BASES.get(name)

def registered_arms() -> tuple[str, ...]:
    load_builtin_descriptors(); return _ARMS.names()

def registered_hands() -> tuple[str, ...]:
    load_builtin_descriptors(); return _HANDS.names()

def registered_bases() -> tuple[str, ...]:
    load_builtin_descriptors(); return _BASES.names()
