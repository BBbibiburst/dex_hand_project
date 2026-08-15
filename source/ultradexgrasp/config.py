"""Configuration loading for the independent UltraDexGrasp pipeline."""

from __future__ import annotations

import json
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

from source.ultradexgrasp.executor import ExecutionConfig
from source.ultradexgrasp.synthesizer import SynthesisConfig

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "ultradexgrasp" / "default.json"


@dataclass(frozen=True)
class PipelineConfig:
    target_size: float
    surface_points: int
    surrogate_cache: Path
    surrogate_options: dict[str, Any]
    synthesis: SynthesisConfig
    execution: ExecutionConfig


def _dataclass_options(cls, payload: dict[str, Any]):
    allowed = {item.name for item in fields(cls)}
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise ValueError(f"Unknown {cls.__name__} options: {unknown}.")
    return cls(**payload)


def load_pipeline_config(path: str | Path = DEFAULT_CONFIG_PATH) -> PipelineConfig:
    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError(f"Unsupported UltraDexGrasp config schema in {path}.")
    cache = Path(payload["surrogate_cache"])
    if not cache.is_absolute():
        cache = PROJECT_ROOT / cache
    target_size = float(payload["target_size"])
    surface_points = int(payload["surface_points"])
    if target_size <= 0.0 or surface_points < 128:
        raise ValueError("target_size must be positive and surface_points at least 128.")
    synthesis = _dataclass_options(SynthesisConfig, dict(payload.get("synthesis", {})))
    execution = _dataclass_options(ExecutionConfig, dict(payload.get("execution", {})))
    synthesis.validate()
    execution.validate()
    return PipelineConfig(
        target_size=target_size,
        surface_points=surface_points,
        surrogate_cache=cache,
        surrogate_options=dict(payload.get("surrogate", {})),
        synthesis=synthesis,
        execution=execution,
    )
