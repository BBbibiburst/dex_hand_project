"""Configuration helpers for operator input devices."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from source.assets import PROJECT_ROOT


DEFAULT_TELEOP_CONFIG_PATH = PROJECT_ROOT / "configs" / "teleop.json"


def load_teleop_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load the shared glove/Vive configuration."""
    resolved = DEFAULT_TELEOP_CONFIG_PATH if path is None else Path(path)
    if not resolved.is_absolute():
        resolved = PROJECT_ROOT / resolved
    if not resolved.exists():
        return {}
    with resolved.open("r", encoding="utf-8") as stream:
        config = json.load(stream)
    if not isinstance(config, dict):
        raise ValueError(f"Teleop config must be a JSON object: {resolved}")
    return config


def save_glove_calibration(
    minimum: list[float],
    maximum: list[float],
    path: str | Path | None = None,
) -> Path:
    """Persist validated raw glove bounds for later teleoperation sessions."""
    resolved = DEFAULT_TELEOP_CONFIG_PATH if path is None else Path(path)
    if not resolved.is_absolute():
        resolved = PROJECT_ROOT / resolved
    config = load_teleop_config(resolved)
    config["glove_calibration"] = {
        "channel_order": ["thumb", "index", "middle", "ring", "pinky"],
        "open_minimum": [float(value) for value in minimum],
        "fist_maximum": [float(value) for value in maximum],
        "validated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    }
    temporary = resolved.with_suffix(resolved.suffix + ".tmp")
    temporary.write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(resolved)
    return resolved
