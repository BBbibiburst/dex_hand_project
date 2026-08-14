"""Grasp search, trajectory generation, and standalone validation APIs."""

from importlib import import_module

from source.grasping.dex_hand_surface import (
    PosedDexHandSurface,
    load_posed_dex_hand_surface,
)
from source.grasping.standalone_validator import (
    DirectHoldValidationResult,
    TrajectoryValidationResult,
    validate_grasp_config,
    validate_grasp_payload_direct,
    validate_grasp_payload_trajectory,
    validate_standalone,
)

_SEARCH_EXPORTS = {
    "GraspConfigSearchResult",
    "ValidatedGraspConfigResult",
    "generate_grasp_config",
    "generate_validated_grasp_config",
    "search_grasp_config",
}


def __getattr__(name: str):
    """Load the production search lazily so its ``python -m`` entry stays clean."""
    if name in _SEARCH_EXPORTS:
        module = import_module("source.grasping.grasp_config_search")
        return getattr(module, name)
    raise AttributeError(name)


__all__ = [
    "PosedDexHandSurface",
    "load_posed_dex_hand_surface",
    "GraspConfigSearchResult",
    "ValidatedGraspConfigResult",
    "generate_grasp_config",
    "generate_validated_grasp_config",
    "search_grasp_config",
    "DirectHoldValidationResult",
    "TrajectoryValidationResult",
    "validate_grasp_config",
    "validate_grasp_payload_direct",
    "validate_grasp_payload_trajectory",
    "validate_standalone",
]
