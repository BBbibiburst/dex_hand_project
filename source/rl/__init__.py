"""Residual reinforcement learning for UltraDexGrasp trajectory refinement."""

from __future__ import annotations

_EXPORTS = {
    "ReferenceTrajectory": ("source.rl.reference", "ReferenceTrajectory"),
    "resolve_reference_manifest": ("source.rl.reference", "resolve_reference_manifest"),
    "ResidualTrajectory": ("source.rl.trajectory", "ResidualTrajectory"),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str):
    try:
        module_name, attribute = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(name) from exc
    from importlib import import_module

    return getattr(import_module(module_name), attribute)
