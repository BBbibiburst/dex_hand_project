"""Configuration helpers for operator input devices."""

from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

from source.assets import PROJECT_ROOT


DEFAULT_TELEOP_CONFIG_PATH = PROJECT_ROOT / "configs" / "teleop.json"
GLOVE_CHANNEL_COUNT = 5


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
    *,
    history_weight: float = 0.0,
) -> Path:
    """Persist raw glove bounds, optionally blending them with saved history."""
    resolved = DEFAULT_TELEOP_CONFIG_PATH if path is None else Path(path)
    if not resolved.is_absolute():
        resolved = PROJECT_ROOT / resolved
    config = load_teleop_config(resolved)
    current_minimum, current_maximum = _validate_glove_bounds(
        minimum, maximum, label="Current"
    )
    if not 0.0 <= history_weight <= 1.0:
        raise ValueError("history_weight must be between 0 and 1.")

    used_history = False
    calibration = config.get("glove_calibration")
    if history_weight > 0.0 and isinstance(calibration, dict):
        historical_minimum, historical_maximum = _validate_glove_bounds(
            calibration.get("open_minimum"),
            calibration.get("fist_maximum"),
            label="Historical",
        )
        current_weight = 1.0 - history_weight
        current_minimum = [
            history_weight * old + current_weight * new
            for old, new in zip(historical_minimum, current_minimum)
        ]
        current_maximum = [
            history_weight * old + current_weight * new
            for old, new in zip(historical_maximum, current_maximum)
        ]
        used_history = True

    config["glove_calibration"] = {
        "channel_order": ["thumb", "index", "middle", "ring", "pinky"],
        "open_minimum": current_minimum,
        "fist_maximum": current_maximum,
        "validated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "history_weight": history_weight if used_history else 0.0,
    }
    temporary = resolved.with_suffix(resolved.suffix + ".tmp")
    temporary.write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(resolved)
    return resolved


def _validate_glove_bounds(
    minimum: Sequence[float] | None,
    maximum: Sequence[float] | None,
    *,
    label: str,
) -> tuple[list[float], list[float]]:
    """Validate and normalize a pair of five-channel glove bounds."""
    if minimum is None or maximum is None:
        raise ValueError(f"{label} glove calibration is missing bounds.")
    if len(minimum) != GLOVE_CHANNEL_COUNT or len(maximum) != GLOVE_CHANNEL_COUNT:
        raise ValueError(
            f"{label} glove calibration must contain {GLOVE_CHANNEL_COUNT} channels."
        )
    normalized_minimum = [float(value) for value in minimum]
    normalized_maximum = [float(value) for value in maximum]
    if not all(
        math.isfinite(value)
        for value in (*normalized_minimum, *normalized_maximum)
    ):
        raise ValueError(f"{label} glove calibration contains non-finite values.")
    if any(
        upper <= lower
        for lower, upper in zip(normalized_minimum, normalized_maximum)
    ):
        raise ValueError(
            f"{label} glove fist values must exceed open-hand values."
        )
    return normalized_minimum, normalized_maximum
