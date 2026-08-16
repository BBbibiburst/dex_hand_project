"""Raw teleoperation trajectory collection, optimization, replay, and validation."""

from __future__ import annotations

from importlib import import_module

_EXPORTS = {
    "TRAJECTORY_SCHEMA_VERSION": ("source.teleop.trajectory.contracts", "TRAJECTORY_SCHEMA_VERSION"),
    "TeleopTrajectory": ("source.teleop.trajectory.contracts", "TeleopTrajectory"),
    "TeleopTrajectoryBuffer": ("source.teleop.trajectory.contracts", "TeleopTrajectoryBuffer"),
    "TrajectoryOptimizationConfig": (
        "source.teleop.trajectory.optimize",
        "TrajectoryOptimizationConfig",
    ),
    "optimize_teleop_trajectory": (
        "source.teleop.trajectory.optimize",
        "optimize_teleop_trajectory",
    ),
    "TrajectoryReplayResult": ("source.teleop.trajectory.runtime", "TrajectoryReplayResult"),
    "make_trajectory_env": ("source.teleop.trajectory.runtime", "make_trajectory_env"),
    "replay_teleop_trajectory": ("source.teleop.trajectory.runtime", "replay_teleop_trajectory"),
    "restore_trajectory_initial_state": (
        "source.teleop.trajectory.runtime",
        "restore_trajectory_initial_state",
    ),
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
