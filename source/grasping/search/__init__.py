"""Modular production grasp search.

Catalog I/O, data contracts, and pure planning metadata remain lightweight;
simulation-backed search/publication code is imported only when requested.
"""

from importlib import import_module

_EXPORTS = {
    "ApproachPlan": ("source.grasping.search.types", "ApproachPlan"),
    "Candidate": ("source.grasping.search.types", "Candidate"),
    "Cloud": ("source.grasping.search.types", "Cloud"),
    "Device": ("source.grasping.search.types", "Device"),
    "GraspConfigSearchResult": ("source.grasping.search.types", "GraspConfigSearchResult"),
    "Surface": ("source.grasping.search.types", "Surface"),
    "ValidatedGraspConfigResult": (
        "source.grasping.search.types",
        "ValidatedGraspConfigResult",
    ),
    "approach_direction_metadata": (
        "source.grasping.search.planning",
        "approach_direction_metadata",
    ),
    "grasp_benchmark_report_path": (
        "source.grasping.search.catalog",
        "grasp_benchmark_report_path",
    ),
    "grasp_config_directory": ("source.grasping.search.catalog", "grasp_config_directory"),
    "grasp_config_name": ("source.grasping.search.catalog", "grasp_config_name"),
    "resolve_object": ("source.grasping.search.catalog", "resolve_object"),
    "generate_grasp_config": ("source.grasping.search.api", "generate_grasp_config"),
    "generate_validated_grasp_config": (
        "source.grasping.search.api",
        "generate_validated_grasp_config",
    ),
    "replan_evolved_payload": ("source.grasping.search.api", "replan_evolved_payload"),
    "search_grasp_config": ("source.grasping.search.api", "search_grasp_config"),
    "select_executable_config": ("source.grasping.search.api", "select_executable_config"),
}
__all__ = list(_EXPORTS)


def __getattr__(name: str):
    try:
        module_name, attribute = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(name) from exc
    value = getattr(import_module(module_name), attribute)
    globals()[name] = value
    return value
