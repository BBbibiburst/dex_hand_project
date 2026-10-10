"""Transfer a verified Lift terminal state into a physical Stack task."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import mujoco
import numpy as np

from source.envs.manipulation import make_stack_env
from source.geometry import mat_to_quat
from source.grasping.contracts import DemonstrationEpisode
from source.grasping.executor import (
    STAGE_CODES,
    ExecutionConfig,
    _Recorder,
    _run_pose_segment,
    actuator_targets_from_fractions,
    grasp_hand_targets,
)
from source.grasping.task_transfer import _run_adaptive_release, _select_transport_hand
from source.grasping.push_transfer import _palm_geometry_in_ee_frame

STACK_TRANSFER_VERSION = "stack-transfer-v24"


class _StackTransferPlacement:
    """Deterministic separated reset poses; Lift state later replaces cubeA."""

    def sample(self, objects, *, rng, reference_pos, z_offset: float = 0.002):
        _ = rng
        reference = np.asarray(reference_pos, dtype=np.float64)
        # Keep a visibly meaningful source-to-stack transfer. Restored Lift
        # objects are typically near (0.54, -0.02), so this support centre is
        # about 22 cm away and its near edge remains physically separated.
        centers = ((0.54, -0.10), (0.46, 0.18))
        result: dict[str, Any] = {}
        for obj, center in zip(objects, centers, strict=True):
            result[obj.name] = (
                np.asarray(
                    [center[0], center[1], reference[2] + obj.bottom_offset + z_offset],
                    dtype=np.float64,
                ),
                np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
            )
        return result


@dataclass(frozen=True)
class StackTransferConfig:
    """Motion and release settings for placing one catalogue object on a support."""

    transport_steps: int = 90
    descend_steps: int = 55
    release_steps: int = 80
    retreat_steps: int = 100
    verify_steps: int = 150
    grip_settle_steps: int = 20
    drop_clearance: float = 0.001
    transport_clearance: float = 0.012
    platform_crossing_clearance: float = 0.040
    platform_edge_clearance: float = 0.010
    crossing_steps: int = 45
    servo_corrections: int = 30
    servo_position_tolerance: float = 0.008
    servo_maximum_correction: float = 0.012
    adaptive_transport_grip: bool = True
    transport_finger_boost: float = 0.0
    transport_thumb_boost: float = 0.0
    maximum_release_open_fraction: float = 0.85
    source_frame_rank: int = 0
    cradle_transport: bool = False
    cradle_steps: int = 80
    grasp_aligned_support: bool = True
    support_distance: float = 0.12
    support_half_size: float = 0.05
    support_height: float = 0.055
    support_direction_offset_degrees: float = 0.0

    def validate(self) -> None:
        if (
            min(
                self.transport_steps,
                self.descend_steps,
                self.release_steps,
                self.retreat_steps,
                self.verify_steps,
                self.grip_settle_steps,
                self.cradle_steps,
                self.crossing_steps,
            )
            <= 0
        ):
            raise ValueError("Every Stack transfer stage must contain at least one step.")
        if self.drop_clearance <= 0.0:
            raise ValueError("drop_clearance must be positive.")
        if self.transport_clearance <= 0.0:
            raise ValueError("transport_clearance must be positive.")
        if self.platform_crossing_clearance <= 0.0:
            raise ValueError("platform_crossing_clearance must be positive.")
        if self.platform_edge_clearance <= 0.0:
            raise ValueError("platform_edge_clearance must be positive.")
        if self.servo_corrections < 1:
            raise ValueError("servo_corrections must be at least one.")
        if self.servo_position_tolerance <= 0.0:
            raise ValueError("servo_position_tolerance must be positive.")
        if self.servo_maximum_correction <= 0.0:
            raise ValueError("servo_maximum_correction must be positive.")
        if not 0.0 < self.maximum_release_open_fraction <= 1.0:
            raise ValueError("maximum_release_open_fraction must lie in (0, 1].")
        if not 0.0 <= self.transport_finger_boost <= 0.40:
            raise ValueError("transport_finger_boost must lie in [0, 0.40].")
        if not 0.0 <= self.transport_thumb_boost <= 0.40:
            raise ValueError("transport_thumb_boost must lie in [0, 0.40].")
        if self.source_frame_rank < 0:
            raise ValueError("source_frame_rank must be non-negative.")
        if self.support_distance <= 0.0:
            raise ValueError("support_distance must be positive.")
        if not 0.045 <= self.support_half_size <= 0.09:
            raise ValueError("support_half_size must lie in [0.045, 0.09].")
        if not 0.055 <= self.support_height <= 0.085:
            raise ValueError("support_height must lie in [0.055, 0.085].")

    @property
    def maximum_steps(self) -> int:
        return (
            2 * self.transport_steps
            + self.descend_steps
            + self.release_steps
            + self.retreat_steps
            + self.verify_steps
            + self.grip_settle_steps
            + 2 * self.crossing_steps
            + (self.cradle_steps if self.cradle_transport else 0)
            + 30
        )


def _servo_grasped_object(
    env,
    recorder: _Recorder,
    *,
    stage: str,
    desired_object_position: np.ndarray,
    quaternion: np.ndarray,
    hand: np.ndarray,
    total_steps: int,
    corrections: int,
    tolerance: float,
    maximum_correction: float = 0.045,
    horizontal_tolerance: float | None = None,
    object_name: str = "cubeA",
) -> tuple[bool, float]:
    """Move a held object with measured Cartesian error correction.

    A fixed hand-to-object offset is not accurate enough for an underactuated
    grasp: passive joints settle during transport and arm IK may retain a
    residual.  Each short segment therefore measures the actual object error
    and updates the next end-effector target without teleporting either body.
    """

    arm = env.controller.arm_controller
    remaining_steps = int(total_steps)
    ended = False
    final_error = float("inf")
    for correction in range(corrections):
        current_object = env.task._body_pos(env.model, env.data, object_name).copy()
        error = np.asarray(desired_object_position, dtype=np.float64) - current_object
        final_error = float(np.linalg.norm(error))
        if horizontal_tolerance is not None and np.linalg.norm(error[:2]) <= horizontal_tolerance:
            break
        if final_error <= tolerance:
            break
        # Limit one correction to prevent a large IK residual from producing
        # a sudden Cartesian command that strips the grasp.
        error_norm = float(np.linalg.norm(error))
        if error_norm > maximum_correction:
            error *= maximum_correction / error_norm
        target_ee = env.data.site_xpos[arm.site_id].astype(np.float64).copy() + error
        passes_left = corrections - correction
        steps = max(1, remaining_steps // passes_left)
        remaining_steps -= steps
        _, _, ended = _run_pose_segment(
            env,
            recorder,
            stage=stage,
            target_position=target_ee,
            target_quaternion=quaternion,
            start_hand=hand,
            target_hand=hand,
            steps=steps,
            smooth=True,
        )
        if ended:
            break
    final_object = env.task._body_pos(env.model, env.data, object_name)
    final_error = float(np.linalg.norm(desired_object_position - final_object))
    return ended, final_error


def _best_lift_transfer_frame(source: DemonstrationEpisode, rank: int = 0) -> int:
    """Choose a late lifted frame that still has the strongest hand contact."""

    stage = np.asarray(source.arrays["stage"])
    eligible = np.flatnonzero((stage == STAGE_CODES["lift"]) | (stage == STAGE_CODES["verify"]))
    if not len(eligible):
        return -1
    digit_contacts = np.asarray(
        source.arrays.get("robot_object_digit_contact_count", np.zeros(len(stage)))
    )
    digits = (
        np.count_nonzero(digit_contacts > 0, axis=1) if digit_contacts.ndim > 1 else digit_contacts
    )
    contacts = np.asarray(source.arrays.get("robot_object_contact_count", np.zeros(len(stage))))
    force = np.asarray(source.arrays.get("robot_object_normal_force", np.zeros(len(stage))))
    object_velocity = np.linalg.norm(np.asarray(source.arrays["qvel"])[eligible, -6:], axis=1)
    # Highest pose is not necessarily the best transfer state: some Lift
    # episodes reach their peak ballistically after contact has weakened.
    # Prefer opposed contact with low object velocity, then continue the
    # vertical lift under closed-loop control in the Stack environment.
    score = (
        8.0 * digits[eligible]
        + 2.0 * contacts[eligible]
        + np.minimum(force[eligible], 20.0)
        - 100.0 * object_velocity
    )
    order = np.argsort(-(score + 1e-6 * eligible))
    return int(eligible[int(order[min(rank, len(order) - 1)])])


def _object_downward_extent(env, object_name: str) -> float:
    """Current distance from object origin to the exact lowest geom support."""

    binding = env.task._require_bindings().objects[object_name]
    origin_z = float(env.data.xpos[binding.body_id, 2])
    lows: list[float] = []
    for geom_id in binding.geom_ids:
        geom_type = int(env.model.geom_type[geom_id])
        rotation = env.data.geom_xmat[geom_id].reshape(3, 3)
        center_z = float(env.data.geom_xpos[geom_id, 2])
        size = env.model.geom_size[geom_id]
        if geom_type == int(mujoco.mjtGeom.mjGEOM_MESH):
            mesh_id = int(env.model.geom_dataid[geom_id])
            first = int(env.model.mesh_vertadr[mesh_id])
            count = int(env.model.mesh_vertnum[mesh_id])
            vertices = env.model.mesh_vert[first : first + count]
            lows.append(float(np.min(vertices @ rotation[2] + center_z)))
            continue
        if geom_type == int(mujoco.mjtGeom.mjGEOM_BOX):
            vertical_radius = float(np.dot(np.abs(rotation[2]), size))
        elif geom_type in {
            int(mujoco.mjtGeom.mjGEOM_CYLINDER),
            int(mujoco.mjtGeom.mjGEOM_CAPSULE),
        }:
            radial = float(size[0] * np.linalg.norm(rotation[2, :2]))
            axial = float(
                (size[1] + (size[0] if geom_type == int(mujoco.mjtGeom.mjGEOM_CAPSULE) else 0.0))
                * abs(rotation[2, 2])
            )
            vertical_radius = radial + axial
        elif geom_type == int(mujoco.mjtGeom.mjGEOM_ELLIPSOID):
            vertical_radius = float(np.linalg.norm(rotation[2] * size))
        elif geom_type == int(mujoco.mjtGeom.mjGEOM_SPHERE):
            vertical_radius = float(size[0])
        else:
            vertical_radius = float(env.model.geom_rbound[geom_id])
        lows.append(center_z - vertical_radius)
    lowest = min(lows)
    return max(0.0, origin_z - lowest)


def _restore_lift_terminal_into_stack(
    env, source: DemonstrationEpisode, *, frame_rank: int = 0
) -> int:
    """Restore a stable lifted frame while preserving the Stack support state."""

    bindings = env.task._require_bindings()
    cube_a = bindings.objects["cubeA"]
    frame = _best_lift_transfer_frame(source, frame_rank)
    source_qpos = np.asarray(source.arrays["qpos"][frame], dtype=np.float64)
    source_qvel = np.asarray(source.arrays["qvel"][frame], dtype=np.float64)
    source_ctrl = np.asarray(source.arrays["ctrl"][frame], dtype=np.float64)
    robot_nq = int(cube_a.qpos_adr)
    robot_nv = int(cube_a.qvel_adr)
    if source_qpos.shape != (robot_nq + 7,) or source_qvel.shape != (robot_nv + 6,):
        raise ValueError("Lift source is incompatible with the Stack robot layout.")
    if source_ctrl.shape != env.data.ctrl.shape:
        raise ValueError("Lift source has an incompatible actuator layout.")
    env.data.qpos[:robot_nq] = source_qpos[:robot_nq]
    env.data.qvel[:robot_nv] = source_qvel[:robot_nv]
    env.data.qpos[cube_a.qpos_adr : cube_a.qpos_adr + 7] = source_qpos[-7:]
    env.data.qvel[cube_a.qvel_adr : cube_a.qvel_adr + 6] = source_qvel[-6:]
    env.data.ctrl[:] = source_ctrl
    mujoco.mj_forward(env.model, env.data)
    return frame


def execute_stack_transfer(
    source: DemonstrationEpisode,
    *,
    seed: int = 0,
    grasp_config: ExecutionConfig | None = None,
    transfer_config: StackTransferConfig | None = None,
    render_mode: str | None = None,
) -> DemonstrationEpisode:
    """Continue one verified Lift episode by stacking its object on a solid support."""

    if not source.success:
        raise ValueError("Stack transfer requires a successful Lift episode.")
    transfer = transfer_config or StackTransferConfig()
    transfer.validate()
    grasp = grasp_config or ExecutionConfig()
    support_height = transfer.support_height
    source_frame = _best_lift_transfer_frame(source, transfer.source_frame_rank)
    source_object = np.asarray(source.arrays["object_position"][source_frame], dtype=np.float64)
    support_center = np.asarray([0.46, 0.18], dtype=np.float64)
    if transfer.grasp_aligned_support:
        carry_direction = np.asarray(source.candidate.hand_rotation_matrix[:, 0], dtype=np.float64)[
            :2
        ]
        carry_direction /= np.linalg.norm(carry_direction)
        angle = np.deg2rad(transfer.support_direction_offset_degrees)
        carry_direction = np.asarray(
            [
                np.cos(angle) * carry_direction[0] - np.sin(angle) * carry_direction[1],
                np.sin(angle) * carry_direction[0] + np.cos(angle) * carry_direction[1],
            ]
        )
        support_center = source_object[:2] + transfer.support_distance * carry_direction
        support_center = np.clip(support_center, (0.26, -0.31), (0.84, 0.31))
    stored_task_config = {
        "stacked_object_id": source.object_id,
        "support_height": support_height,
        "support_center": tuple(float(x) for x in support_center),
        "support_half_size": transfer.support_half_size,
        "terminate_on_success": False,
    }
    env = make_stack_env(
        task_config={
            **stored_task_config,
            "placement_sampler": _StackTransferPlacement(),
        },
        control_mode="ik",
        enable_tactile_sensors=grasp.enable_tactile_sensors,
        episode_length=transfer.maximum_steps,
        render_mode=render_mode,
    )
    try:
        env.reset(seed=seed)
        restored_frame = _restore_lift_terminal_into_stack(
            env, source, frame_rank=transfer.source_frame_rank
        )
        if restored_frame != source_frame:
            raise RuntimeError("Stack source-frame selection changed during environment setup.")
        recorder = _Recorder("cubeA")
        arm = env.controller.arm_controller
        ee_position = env.data.site_xpos[arm.site_id].astype(np.float64).copy()
        ee_quaternion = mat_to_quat(env.data.site_xmat[arm.site_id])
        arm._prev_target_q = env.data.qpos[arm.qpos_addrs].copy()
        arm._filtered_velocity = np.zeros_like(arm._prev_target_q)
        arm._prev_ee_target = np.concatenate((ee_position, ee_quaternion))
        cube_a = env.task._body_pos(env.model, env.data, "cubeA").copy()
        cube_b = env.task._body_pos(env.model, env.data, "cubeB").copy()
        approach_fractions, _ = grasp_hand_targets(source.candidate.actuator_fractions, grasp)
        open_hand = actuator_targets_from_fractions(env, approach_fractions)
        grip_hand = env.controller.hand_controller.current_action(env.model, env.data).astype(
            np.float32
        )
        if transfer.adaptive_transport_grip:
            transport_hand = _select_transport_hand(env, grip_hand, "cubeA")
        else:
            hand = env.controller.hand_controller
            grip_fraction = np.divide(
                grip_hand - hand.ctrl_low,
                hand.ctrl_high - hand.ctrl_low,
                out=np.zeros(6, dtype=np.float32),
                where=(hand.ctrl_high - hand.ctrl_low) != 0.0,
            )
            grip_fraction[:4] += transfer.transport_finger_boost
            grip_fraction[5] += transfer.transport_thumb_boost
            transport_hand = actuator_targets_from_fractions(env, np.clip(grip_fraction, 0.0, 1.0))
        # Lift success may be recorded while the object is still moving
        # upward.  Always dwell at the restored wrist pose before beginning
        # suspended transport, even when adaptive preload chooses the same
        # actuator target.  This dissipates inherited object velocity and
        # turns the imported Lift state into a stable grasp state.
        ended = False
        _, _, ended = _run_pose_segment(
            env,
            recorder,
            stage="clear",
            target_position=ee_position,
            target_quaternion=ee_quaternion,
            start_hand=grip_hand,
            target_hand=transport_hand,
            steps=transfer.grip_settle_steps,
            smooth=True,
        )
        if transfer.cradle_transport and not ended:
            _palm_center_local, palm_normal_local, palm_long_local = _palm_geometry_in_ee_frame(env)
            current_rotation = env.data.site_xmat[arm.site_id].reshape(3, 3).copy()
            object_world = env.task._body_pos(env.model, env.data, "cubeA").copy()
            object_local = current_rotation.T @ (object_world - ee_position)
            # The grasped object lies on the palm geom's negative-normal side;
            # align that contact face upward so gravity seats the object in
            # the hand during suspended lateral transport.
            world_normal = np.asarray([0.0, 0.0, -1.0], dtype=np.float64)
            world_long = np.asarray([0.0, 1.0, 0.0], dtype=np.float64)
            local_side = np.cross(palm_long_local, palm_normal_local)
            world_side = np.cross(world_long, world_normal)
            cradle_rotation = (
                np.column_stack((world_normal, world_long, world_side))
                @ np.column_stack((palm_normal_local, palm_long_local, local_side)).T
            )
            ee_quaternion = mat_to_quat(cradle_rotation)
            ee_position = object_world - cradle_rotation @ object_local
            _, _, ended = _run_pose_segment(
                env,
                recorder,
                stage="clear",
                target_position=ee_position,
                target_quaternion=ee_quaternion,
                start_hand=transport_hand,
                target_hand=transport_hand,
                steps=transfer.cradle_steps,
                smooth=True,
            )
        cube_a = env.task._body_pos(env.model, env.data, "cubeA").copy()
        support_top = cube_b[2] + env.task.objects[1].bottom_offset
        desired_object = np.asarray(
            [
                cube_b[0],
                cube_b[1],
                support_top + env.task.objects[0].bottom_offset + transfer.drop_clearance,
            ],
            dtype=np.float64,
        )
        direction = cube_b[:2] - cube_a[:2]
        direction /= np.linalg.norm(direction)
        edge_distance = (
            transfer.support_half_size
            + env.task.objects[0].horizontal_radius
            + transfer.platform_edge_clearance
        )
        edge_object = desired_object.copy()
        edge_object[:2] = cube_b[:2] - edge_distance * direction
        # Stay at the lowest height that clears the fixed platform while
        # travelling through free space. A short local lift is performed only
        # at the platform edge, which preserves marginal underactuated grasps.
        edge_object[2] = float(cube_a[2])
        # Translate through unobstructed free space at the inherited Lift
        # height. Marginal underactuated grasps are often stripped by an
        # unnecessary vertical move immediately after state restoration. Only
        # raise locally after reaching the outside edge of the platform.
        raise_error = 0.0
        ended, transport_error = _servo_grasped_object(
            env,
            recorder,
            stage="transport",
            desired_object_position=edge_object,
            quaternion=ee_quaternion,
            hand=transport_hand,
            total_steps=transfer.transport_steps,
            corrections=transfer.servo_corrections,
            tolerance=transfer.servo_position_tolerance,
            maximum_correction=transfer.servo_maximum_correction,
            horizontal_tolerance=0.008,
        )
        downward_extent = _object_downward_extent(env, "cubeA")
        crossing_object = edge_object.copy()
        crossing_object[2] = max(
            float(edge_object[2]),
            float(support_top + downward_extent + transfer.platform_crossing_clearance),
        )
        if not ended:
            ended, crossing_lift_error = _servo_grasped_object(
                env,
                recorder,
                stage="transport",
                desired_object_position=crossing_object,
                quaternion=ee_quaternion,
                hand=transport_hand,
                total_steps=transfer.crossing_steps,
                corrections=transfer.servo_corrections,
                tolerance=transfer.servo_position_tolerance,
                maximum_correction=transfer.servo_maximum_correction,
            )
        else:
            crossing_lift_error = float("inf")
        crossing_object[:2] = cube_b[:2]
        if not ended:
            ended, crossing_error = _servo_grasped_object(
                env,
                recorder,
                stage="transport",
                desired_object_position=crossing_object,
                quaternion=ee_quaternion,
                hand=transport_hand,
                total_steps=transfer.crossing_steps,
                corrections=transfer.servo_corrections,
                tolerance=transfer.servo_position_tolerance,
                maximum_correction=transfer.servo_maximum_correction,
                horizontal_tolerance=0.008,
            )
        else:
            crossing_error = float("inf")
        if not ended:
            # Once the suspended object is safely over the platform interior,
            # freeze that horizontal landing coordinate. The remaining motion
            # is strictly vertical; chasing the exact platform centre can
            # otherwise turn grasp sag into premature support contact.
            current_object = env.task._body_pos(env.model, env.data, "cubeA").copy()
            desired_object[:2] = current_object[:2]
            ended, descend_error = _servo_grasped_object(
                env,
                recorder,
                stage="descend",
                desired_object_position=desired_object,
                quaternion=ee_quaternion,
                hand=transport_hand,
                total_steps=transfer.descend_steps,
                corrections=transfer.servo_corrections,
                tolerance=transfer.servo_position_tolerance,
                maximum_correction=transfer.servo_maximum_correction,
            )
        else:
            descend_error = float("inf")
        current_ee = env.data.site_xpos[arm.site_id].astype(np.float64).copy()
        current_object = env.task._body_pos(env.model, env.data, "cubeA").copy()
        release_ee = current_ee + (desired_object - current_object)
        release_hand = open_hand
        if not ended:
            release_hand, ended, _ = _run_adaptive_release(
                env,
                recorder,
                position=release_ee,
                quaternion=ee_quaternion,
                grip_hand=transport_hand,
                open_hand=open_hand,
                steps=transfer.release_steps,
                maximum_open_fraction=transfer.maximum_release_open_fraction,
                object_name="cubeA",
                settled=lambda: env.task._stable_platform_contact(env.model, env.data),
            )
        retreat = release_ee + np.asarray([0.0, 0.0, 0.14])
        for stage, steps in (
            ("retreat", transfer.retreat_steps),
            ("task_verify", transfer.verify_steps),
        ):
            if ended:
                break
            _, _, ended = _run_pose_segment(
                env,
                recorder,
                stage=stage,
                target_position=retreat,
                target_quaternion=ee_quaternion,
                start_hand=release_hand,
                target_hand=release_hand,
                steps=steps,
                smooth=True,
            )
        arrays = recorder.arrays()
        verify = arrays["task_success"][arrays["stage"] == STAGE_CODES["task_verify"]]
        transport_support_contact = arrays["object_support_contact"][
            arrays["stage"] == STAGE_CODES["transport"]
        ]
        suspended_transport = not bool(np.any(transport_support_contact))
        success = bool(
            suspended_transport and len(verify) and verify[-1] and float(verify.mean()) >= 0.8
        )
        return DemonstrationEpisode(
            object_id=source.object_id,
            seed=seed,
            candidate=source.candidate,
            arrays=arrays,
            success=success,
            terminal_stage="task_verify",
            failure_reason=None if success else "object was not stacked stably on the support",
            metadata={
                "task": "stack",
                "source_skill": "lift",
                "source_frame": source_frame,
                "pipeline_version": STACK_TRANSFER_VERSION,
                "task_config": stored_task_config,
                "transfer_config": asdict(transfer),
                "support_position": cube_b.tolist(),
                "support_height": support_height,
                "target_object_position": desired_object.tolist(),
                "transport_downward_extent": downward_extent,
                "raise_position_error": raise_error,
                "transport_position_error": transport_error,
                "crossing_lift_position_error": crossing_lift_error,
                "crossing_position_error": crossing_error,
                "descend_position_error": descend_error,
                "object_final_position": env.task._body_pos(env.model, env.data, "cubeA").tolist(),
                "task_verify_success_fraction": float(verify.mean()) if len(verify) else 0.0,
                "suspended_transport": suspended_transport,
                "action_layout": list(env.controller.ik_action_layout()),
            },
        )
    finally:
        env.close()
