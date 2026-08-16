"""Grasp generation and standalone physics-validation APIs.

Imports are lazy so geometry metadata and configuration helpers remain usable
without importing MuJoCo until simulation-backed functionality is requested.
"""

from importlib import import_module

_EXPORTS = {
    "PosedDexHandSurface": ("source.grasping.dex_hand_surface", "PosedDexHandSurface"),
    "load_posed_dex_hand_surface": (
        "source.grasping.dex_hand_surface",
        "load_posed_dex_hand_surface",
    ),
    "GraspConfigSearchResult": ("source.grasping.search", "GraspConfigSearchResult"),
    "ValidatedGraspConfigResult": ("source.grasping.search", "ValidatedGraspConfigResult"),
    "generate_grasp_config": ("source.grasping.search", "generate_grasp_config"),
    "generate_validated_grasp_config": (
        "source.grasping.search",
        "generate_validated_grasp_config",
    ),
    "search_grasp_config": ("source.grasping.search", "search_grasp_config"),
    "DirectHoldValidationResult": (
        "source.grasping.standalone_validator",
        "DirectHoldValidationResult",
    ),
    "TrajectoryValidationResult": (
        "source.grasping.standalone_validator",
        "TrajectoryValidationResult",
    ),
    "validate_grasp_config": ("source.grasping.standalone_validator", "validate_grasp_config"),
    "validate_grasp_payload_direct": (
        "source.grasping.standalone_validator",
        "validate_grasp_payload_direct",
    ),
    "validate_grasp_payload_trajectory": (
        "source.grasping.standalone_validator",
        "validate_grasp_payload_trajectory",
    ),
    "validate_standalone": ("source.grasping.standalone_validator", "validate_standalone"),
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
