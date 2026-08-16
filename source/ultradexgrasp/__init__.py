"""UltraDexGrasp-style demonstration generation for RM75B + Dex Hand.

The package is deliberately independent from ``source.grasping``.  It uses a
MuJoCo-calibrated differentiable model of the underactuated tendon hand, gradient-based
grasp synthesis, and a planning/execution stage that emits complete episodes.
"""

from source.ultradexgrasp.contracts import (
    EPISODE_SCHEMA_VERSION,
    GRASP_SCHEMA_VERSION,
    DemonstrationEpisode,
    GraspCandidate,
)
from source.ultradexgrasp.executor import ExecutionConfig, execute_grasp
from source.ultradexgrasp.synthesizer import SynthesisConfig, synthesize_grasps

__all__ = [
    "EPISODE_SCHEMA_VERSION",
    "GRASP_SCHEMA_VERSION",
    "DemonstrationEpisode",
    "ExecutionConfig",
    "GraspCandidate",
    "SynthesisConfig",
    "execute_grasp",
    "synthesize_grasps",
]
