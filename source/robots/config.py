# -*- coding: utf-8 -*-
"""Global robot assembly configuration helpers.

The project-level default lives at ``configs/current_robot.json``. It may be
either a full robot config, or a small pointer to a reusable profile:

``{"profile": "robot_profiles/rm75b_pika_gripper.json", "overrides": {...}}``
"""

from __future__ import annotations

import json
from dataclasses import fields
from pathlib import Path
from typing import Any, Mapping, Optional, TypeVar

from source.assets import PROJECT_ROOT
from source.robots.registry import get_arm, get_base, get_hand

DEFAULT_ROBOT_CONFIG_PATH = PROJECT_ROOT / "configs" / "current_robot.json"

CONFIG_ONLY_KEYS = {
    "hand_attach_rot_xyz_deg",
    "attach_point_name",
    "base_mount_site_name",
    "add_preview_scene",
    "tactile_backend",
    "tactile_options",
}

T = TypeVar("T")


def robot_config_path(path: Optional[str | Path] = None) -> Path:
    """Resolve a robot config path, defaulting to ``configs/current_robot.json``."""
    if path is None:
        return DEFAULT_ROBOT_CONFIG_PATH
    candidate = Path(path)
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


def _resolve_profile_path(profile: str | Path, parent: Path) -> Path:
    candidate = Path(profile)
    if candidate.is_absolute():
        return candidate
    parent_relative = parent / candidate
    if parent_relative.exists():
        return parent_relative
    return PROJECT_ROOT / candidate


def _load_robot_config_file(path: Path, seen: set[Path]) -> dict[str, Any]:
    resolved = path.resolve()
    if resolved in seen:
        cycle = " -> ".join(str(item) for item in (*seen, resolved))
        raise ValueError(f"Robot config profile cycle detected: {cycle}")
    if not resolved.exists():
        raise FileNotFoundError(f"Robot config file not found: {resolved}")

    with resolved.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Robot config must be a JSON object: {resolved}")

    profile = data.get("profile")
    if profile is None:
        return data

    base = _load_robot_config_file(
        _resolve_profile_path(profile, resolved.parent),
        seen | {resolved},
    )
    overrides = data.get("overrides", {})
    if not isinstance(overrides, dict):
        raise ValueError(f"'overrides' must be a JSON object in {resolved}")

    inline_overrides = {
        key: value for key, value in data.items() if key not in {"profile", "overrides"}
    }
    merged = apply_config_overrides(base, inline_overrides)
    return apply_config_overrides(merged, overrides)


def load_robot_config(path: Optional[str | Path] = None) -> dict[str, Any]:
    """Load a robot assembly JSON config, resolving optional profiles."""
    resolved = robot_config_path(path)
    return _load_robot_config_file(resolved, set())


def apply_config_overrides(
    config: Mapping[str, Any],
    overrides: Mapping[str, Any],
) -> dict[str, Any]:
    """Return config with non-None override values applied."""
    merged = dict(config)
    for key, value in overrides.items():
        if value is not None:
            merged[key] = value
    return merged


def dataclass_from_robot_config(cls: type[T], config: Mapping[str, Any]) -> T:
    """Create a dataclass from matching keys in a robot config."""
    allowed = {field.name for field in fields(cls)}
    kwargs = {key: value for key, value in config.items() if key in allowed}
    unknown = set(config) - allowed - CONFIG_ONLY_KEYS
    if unknown:
        raise ValueError(f"Unknown robot config key(s) for {cls.__name__}: {sorted(unknown)}")
    return cls(**kwargs)


def descriptors_from_robot_config(config: Mapping[str, Any]):
    """Resolve registered arm, hand, and base descriptors from config names."""
    return (
        get_arm(str(config.get("arm_name", "rm75b"))),
        get_hand(str(config.get("hand_name", "dex_hand"))),
        get_base(str(config.get("base_name", "rethink_minimal_mount"))),
    )


def optional_tuple(config: Mapping[str, Any], key: str):
    """Return a tuple value from config, preserving ``None``."""
    value = config.get(key)
    if value is None:
        return None
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{key!r} must be a list/tuple or null, got {type(value).__name__}.")
    return tuple(value)
