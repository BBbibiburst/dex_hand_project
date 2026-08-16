"""Reinforcement-learning methods used to refine generated grasps.

The package has two explicit algorithm families:

``source.rl.residual``
    Residual actions along a complete UltraDexGrasp reference trajectory.
``source.rl.grasp_edit``
    Categorical wrist-template selection plus continuous six-actuator editing.
"""

from source.rl.residual import ReferenceTrajectory, ResidualTrajectory, resolve_reference_manifest

__all__ = ["ReferenceTrajectory", "ResidualTrajectory", "resolve_reference_manifest"]
