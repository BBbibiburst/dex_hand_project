"""Compatibility surface for the historical monolithic grasp-search module.

Keep compatibility mechanics here so ``source.grasping.grasp_config_search`` can
remain a tiny facade.  Wrappers intentionally read the facade module at call
time, preserving monkeypatch semantics relied on by older tests and scripts.
"""

from __future__ import annotations

from pathlib import Path
import sys

from source.grasping.search import api as _api
from source.grasping.search.catalog import (
    grasp_benchmark_report_path,
    grasp_config_directory,
    grasp_config_name,
    load_cloud,
    manifest_objects,
    object_mesh_path,
    resolve_object,
    safe_name,
)
from source.grasping.search.common import MANIFEST, PROJECT_ROOT, ROOT, progress
from source.grasping.search.devices import DEVICES
from source.grasping.search.engine import search
from source.grasping.search.planning import approach, approach_direction_metadata
from source.grasping.search.scoring import _robot_execution_penalty
from source.grasping.search.types import (
    ApproachPlan,
    Candidate,
    Cloud,
    Device,
    GraspConfigSearchResult,
    Surface,
    ValidatedGraspConfigResult,
)
from source.grasping.standalone_validator import validate_grasp_config

_FACADE_MODULE = "source.grasping.grasp_config_search"
_SEARCH_GRASP_CONFIG_IMPL = _api.search_grasp_config
_GENERATE_VALIDATED_IMPL = _api.generate_validated_grasp_config
_REPLAN_EVOLVED_IMPL = _api.replan_evolved_payload
_SELECT_EXECUTABLE_IMPL = _api.select_executable_config
select_executable_config = _SELECT_EXECUTABLE_IMPL


def _facade():
    return sys.modules.get(_FACADE_MODULE, sys.modules[__name__])


def _hook(name: str, default):
    return getattr(_facade(), name, default)


def _sync_api_dependencies() -> None:
    _api.ROOT = _hook("ROOT", ROOT)
    _api.grasp_config_directory = _hook("grasp_config_directory", grasp_config_directory)
    _api.grasp_config_name = _hook("grasp_config_name", grasp_config_name)
    _api.resolve_object = _hook("resolve_object", resolve_object)
    _api.load_cloud = _hook("load_cloud", load_cloud)
    _api.DEVICES = _hook("DEVICES", DEVICES)
    _api.search = _hook("search", search)
    _api.select_executable_config = _hook(
        "select_executable_config", select_executable_config
    )
    _api.validate_grasp_config = _hook("validate_grasp_config", validate_grasp_config)


def search_grasp_config(*args, **kwargs) -> GraspConfigSearchResult:
    _sync_api_dependencies()
    _api.search_grasp_config = _SEARCH_GRASP_CONFIG_IMPL
    return _SEARCH_GRASP_CONFIG_IMPL(*args, **kwargs)


def generate_grasp_config(
    object_id: str,
    *,
    output: str | Path | None = None,
    **search_kwargs,
) -> Path:
    search_fn = _hook("search_grasp_config", search_grasp_config)
    return search_fn(object_id=object_id, output=output, **search_kwargs).output_path


def generate_validated_grasp_config(*args, **kwargs) -> ValidatedGraspConfigResult:
    _sync_api_dependencies()
    previous_search = _api.search_grasp_config
    try:
        _api.search_grasp_config = _hook("search_grasp_config", search_grasp_config)
        return _GENERATE_VALIDATED_IMPL(*args, **kwargs)
    finally:
        _api.search_grasp_config = previous_search


def replan_evolved_payload(*args, **kwargs) -> dict:
    _sync_api_dependencies()
    return _REPLAN_EVOLVED_IMPL(*args, **kwargs)


__all__ = [
    "ApproachPlan",
    "Candidate",
    "Cloud",
    "DEVICES",
    "Device",
    "GraspConfigSearchResult",
    "MANIFEST",
    "PROJECT_ROOT",
    "ROOT",
    "Surface",
    "ValidatedGraspConfigResult",
    "_robot_execution_penalty",
    "approach",
    "approach_direction_metadata",
    "generate_grasp_config",
    "generate_validated_grasp_config",
    "grasp_benchmark_report_path",
    "grasp_config_directory",
    "grasp_config_name",
    "load_cloud",
    "manifest_objects",
    "object_mesh_path",
    "progress",
    "replan_evolved_payload",
    "resolve_object",
    "safe_name",
    "search",
    "search_grasp_config",
    "select_executable_config",
    "validate_grasp_config",
]
