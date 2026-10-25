"""Diffusion-policy observation and action contracts.

The default manipulation observation contains the complete MuJoCo state,
including free task-object joints.  Those values are simulator privileged
state and are deliberately excluded here.  The DP contract is expressed in
actuator coordinates, in the controller's arm-then-hand order: measured
actuator position, measured actuator velocity, and the last position command.
These fields have direct real-robot counterparts, subject to confirmation with
the external RM75B/Dex Hand control program.
"""

from __future__ import annotations

import numpy as np

DP_STATE_SCHEMA = "actuator_position_velocity_command_v1"
DP_ACTION_SCHEMA = "arm7_hand6_position_command_v1"


def actuator_ids(env) -> np.ndarray:
    """Return controlled actuator IDs in policy action order."""

    return np.concatenate(
        (
            env.controller.arm_controller.actuator_ids,
            env.controller.hand_controller.actuator_ids,
        )
    ).astype(np.int32)


def real_observable_state(env) -> np.ndarray:
    """Return position, velocity, and command for controlled actuators only."""

    ids = actuator_ids(env)
    return np.concatenate(
        (
            env.data.actuator_length[ids],
            env.data.actuator_velocity[ids],
            env.data.ctrl[ids],
        )
    ).astype(np.float32)


def state_layout(env) -> tuple[str, ...]:
    names = tuple(env.controller.actuator_names)
    return tuple(
        [f"position:{name}" for name in names]
        + [f"velocity:{name}" for name in names]
        + [f"command:{name}" for name in names]
    )


def action_layout(env) -> tuple[str, ...]:
    """Return the exact low-level position-action order used by DP."""

    if env.controller.control_mode != "position":
        raise ValueError("DP action layout requires position control mode.")
    return tuple(env.controller.actuator_names)


def validate_position_action_contract(env) -> None:
    layout = action_layout(env)
    if env.action_space.shape != (len(layout),):
        raise RuntimeError(
            f"Action space {env.action_space.shape} does not match actuator layout {len(layout)}."
        )
    if len(layout) != 13:
        raise RuntimeError(
            f"RM75B + six-drive Dex Hand DP action must contain 13 commands, got {len(layout)}."
        )
