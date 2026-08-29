"""Transfer an object-relative verified grasp into downstream manipulation tasks."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any

import numpy as np

from source.envs.manipulation import make_pick_place_env
from source.geometry import mat_to_quat
from source.grasping.contracts import DemonstrationEpisode, GraspCandidate
from source.grasping.executor import (
    STAGE_CODES,
    ExecutionConfig,
    _Recorder,
    _run_pose_segment,
    actuator_targets_from_fractions,
    execute_grasp,
    grasp_hand_targets,
)


@dataclass(frozen=True)
class PickPlaceTransferConfig:
    """Motion budget appended after the reusable grasp-and-lift prefix."""

    clearance_height: float = 0.18
    release_clearance: float = 0.025
    transport_steps: int = 90
    descend_steps: int = 55
    release_steps: int = 35
    retreat_steps: int = 45
    verify_steps: int = 30

    def validate(self) -> None:
        if self.clearance_height <= 0.12:
            raise ValueError("clearance_height must clear the 120 mm bin walls.")
        if self.release_clearance <= 0.0:
            raise ValueError("release_clearance must be positive.")
        if min(
            self.transport_steps,
            self.descend_steps,
            self.release_steps,
            self.retreat_steps,
            self.verify_steps,
        ) <= 0:
            raise ValueError("Every PickPlace transfer stage must contain at least one step.")

    @property
    def appended_steps(self) -> int:
        return (
            self.transport_steps
            + self.descend_steps
            + self.release_steps
            + self.retreat_steps
            + self.verify_steps
        )


def _combine_arrays(
    first: dict[str, np.ndarray], second: dict[str, np.ndarray]
) -> dict[str, np.ndarray]:
    if first.keys() != second.keys():
        raise ValueError("Transferred trajectory recorders have incompatible arrays.")
    return {name: np.concatenate((first[name], second[name]), axis=0) for name in first}


def execute_pick_place_transfer(
    candidate: GraspCandidate,
    *,
    seed: int = 0,
    grasp_config: ExecutionConfig | None = None,
    transfer_config: PickPlaceTransferConfig | None = None,
    render_mode: str | None = None,
) -> DemonstrationEpisode:
    """Retarget a Lift grasp and append transport, release, and verification."""

    transfer = transfer_config or PickPlaceTransferConfig()
    transfer.validate()
    base_grasp = grasp_config or ExecutionConfig()
    # Lift the object above the bin wall before translating laterally.
    grasp = replace(base_grasp, lift_height=max(base_grasp.lift_height, transfer.clearance_height))
    grasp.validate()
    episode_length = grasp.maximum_steps + transfer.appended_steps + 20
    env = make_pick_place_env(
        task_config={
            "object_id": candidate.object_id,
            "reward_shaping": False,
            "terminate_on_success": False,
        },
        control_mode="ik",
        enable_tactile_sensors=grasp.enable_tactile_sensors,
        episode_length=episode_length,
        render_mode=render_mode,
    )
    try:
        prefix = execute_grasp(candidate, seed=seed, config=grasp, environment=env)
        initial_object = np.asarray(prefix.metadata["object_initial_position"], dtype=np.float64)
        current_object = env.task._body_pos(env.model, env.data, "object").copy()
        lifted = current_object[2] - initial_object[2] >= 0.04
        if not lifted:
            prefix.success = False
            prefix.terminal_stage = "lift"
            prefix.failure_reason = "reused grasp did not retain the object to transport clearance"
            prefix.metadata.update(
                {
                    "task": "pick_place",
                    "source_skill": "lift",
                    "transfer_config": asdict(transfer),
                }
            )
            return prefix

        recorder = _Recorder()
        arm = env.controller.arm_controller
        ee_position = env.data.site_xpos[arm.site_id].astype(np.float64).copy()
        ee_quaternion = mat_to_quat(env.data.site_xmat[arm.site_id])
        ee_object_offset = ee_position - current_object
        approach_fractions, grip_fractions = grasp_hand_targets(
            candidate.actuator_fractions, grasp
        )
        open_hand = actuator_targets_from_fractions(env, approach_fractions)
        grip_hand = actuator_targets_from_fractions(env, grip_fractions)

        target_center = env.task._target_center()
        release_object_position = np.asarray(
            [
                target_center[0],
                target_center[1],
                env.task.table_top_z
                + env.task.objects[0].bottom_offset
                + transfer.release_clearance,
            ],
            dtype=np.float64,
        )
        transport_object_position = release_object_position.copy()
        transport_object_position[2] = max(
            current_object[2],
            env.task.table_top_z
            + env.task.objects[0].bottom_offset
            + transfer.clearance_height,
        )
        transport_ee = transport_object_position + ee_object_offset
        release_ee = release_object_position + ee_object_offset

        ended = False
        for stage, position, start_hand, target_hand, steps in (
            ("transport", transport_ee, grip_hand, grip_hand, transfer.transport_steps),
            ("descend", release_ee, grip_hand, grip_hand, transfer.descend_steps),
            ("release", release_ee, grip_hand, open_hand, transfer.release_steps),
            (
                "retreat",
                release_ee + np.asarray([0.0, 0.0, 0.12]),
                open_hand,
                open_hand,
                transfer.retreat_steps,
            ),
            (
                "task_verify",
                release_ee + np.asarray([0.0, 0.0, 0.12]),
                open_hand,
                open_hand,
                transfer.verify_steps,
            ),
        ):
            _, _, ended = _run_pose_segment(
                env,
                recorder,
                stage=stage,
                target_position=position,
                target_quaternion=ee_quaternion,
                start_hand=start_hand,
                target_hand=target_hand,
                steps=steps,
            )
            if ended:
                break

        suffix_arrays = recorder.arrays()
        arrays = _combine_arrays(prefix.arrays, suffix_arrays)
        selected = arrays["stage"] == STAGE_CODES["task_verify"]
        verify = arrays["task_success"][selected]
        success_fraction = float(verify.mean()) if len(verify) else 0.0
        success = bool(len(verify) and verify[-1] and success_fraction >= 0.8)
        final_object = env.task._body_pos(env.model, env.data, "object").copy()
        metadata: dict[str, Any] = dict(prefix.metadata)
        metadata.update(
            {
                "task": "pick_place",
                "source_skill": "lift",
                "source_relative_hand_translation": candidate.hand_translation.tolist(),
                "source_relative_hand_rotation_matrix": candidate.hand_rotation_matrix.tolist(),
                "transfer_config": asdict(transfer),
                "target_object_position": release_object_position.tolist(),
                "object_final_position": final_object.tolist(),
                "task_verify_success_fraction": success_fraction,
                "action_layout": list(env.controller.ik_action_layout()),
            }
        )
        return DemonstrationEpisode(
            object_id=candidate.object_id,
            seed=seed,
            candidate=candidate,
            arrays=arrays,
            success=success,
            terminal_stage="task_verify" if not ended else "retreat",
            failure_reason=None if success else "object was not released stably inside target bin",
            metadata=metadata,
        )
    finally:
        env.close()
