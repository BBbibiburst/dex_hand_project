"""Configuration loading for GraspQP + DexEvolve generation."""

from __future__ import annotations

import json
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

from source.grasping.executor import ExecutionConfig
from source.grasping.seeds import SeedConfig

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "grasping" / "default.json"


@dataclass(frozen=True)
class PipelineConfig:
    target_size: float | None
    maximum_horizontal_diameter: float | None
    surface_points: int
    surrogate_cache: Path
    surrogate_options: dict[str, Any]
    seeds: SeedConfig
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
        raise ValueError(f"Unsupported grasp-generation config schema in {path}.")
    cache = Path(payload["surrogate_cache"])
    if not cache.is_absolute():
        cache = PROJECT_ROOT / cache
    raw_target_size = payload.get("target_size")
    target_size = None if raw_target_size is None else float(raw_target_size)
    raw_diameter = payload.get("maximum_horizontal_diameter", 0.075)
    maximum_horizontal_diameter = (
        None if raw_diameter is None else float(raw_diameter)
    )
    surface_points = int(payload["surface_points"])
    if (
        (target_size is not None and target_size <= 0.0)
        or (maximum_horizontal_diameter is not None and maximum_horizontal_diameter <= 0.0)
        or surface_points < 128
    ):
        raise ValueError("Object size limits must be positive and surface_points at least 128.")
    seeds = _dataclass_options(SeedConfig, dict(payload.get("seeds", {})))
    execution = _dataclass_options(ExecutionConfig, dict(payload.get("execution", {})))
    seeds.validate()
    execution.validate()
    return PipelineConfig(
        target_size=target_size,
        maximum_horizontal_diameter=maximum_horizontal_diameter,
        surface_points=surface_points,
        surrogate_cache=cache,
        surrogate_options=dict(payload.get("surrogate", {})),
        seeds=seeds,
        execution=execution,
    )
