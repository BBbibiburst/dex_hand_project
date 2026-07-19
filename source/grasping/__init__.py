"""Grasp search, trajectory generation, and standalone validation APIs."""

from importlib import import_module

from source.grasping.approach_path_search import plan_approach_path
from source.grasping.dex_hand_surface import (
    PosedDexHandSurface,
    load_posed_dex_hand_surface,
)
from source.grasping.hand_closure_search import (
    HandClosureResult,
    search_hand_grasp,
)
from source.grasping.pika_gripper_search import (
    PikaGraspResult,
    search_pika_grasp,
)
from source.grasping.pika_gripper_surface import (
    PosedPikaGripperSurface,
    load_posed_pika_gripper_surface,
)
from source.grasping.standalone_validator import (
    StandaloneValidationResult,
    validate_grasp_config,
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
    "plan_approach_path",
    "GraspConfigSearchResult",
    "ValidatedGraspConfigResult",
    "generate_grasp_config",
    "generate_validated_grasp_config",
    "search_grasp_config",
    "HandClosureResult",
    "search_hand_grasp",
    "PikaGraspResult",
    "search_pika_grasp",
    "PosedPikaGripperSurface",
    "load_posed_pika_gripper_surface",
    "StandaloneValidationResult",
    "validate_grasp_config",
    "validate_standalone",
]
