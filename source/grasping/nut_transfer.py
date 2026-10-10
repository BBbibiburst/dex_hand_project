"""Insert a grasped procedural bolt into a fixed physical nut."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace

import numpy as np
from scipy.spatial.transform import Rotation

from source.envs.manipulation import make_nut_assembly_env
from source.envs.manipulation.nut_assembly import (
    NUT_ASSEMBLY_DOMAIN_VERSION,
    procedural_nut_variant,
)
from source.envs.manipulation.objects import FreeBoltSpec
from source.geometry import mat_to_quat
from source.grasping.contracts import DemonstrationEpisode
from source.grasping.executor import (
    ExecutionConfig,
    STAGE_CODES,
    _Recorder,
    _run_pose_segment,
    actuator_targets_from_fractions,
    candidate_world_pose,
    grasp_hand_targets,
    hand_attach_rotation,
    quaternion_wxyz_to_matrix,
)
from source.grasping.stack_transfer import _servo_grasped_object
from source.grasping.task_transfer import _select_transport_hand

NUT_TRANSFER_VERSION = "guided-mating-shaft-transfer-v42-human-scale"


@dataclass(frozen=True)
class NutTransferConfig:
    settle_steps: int = 25
    transit_steps: int = 80
    pregrasp_steps: int = 65
    approach_steps: int = 50
    close_steps: int = 55
    hold_steps: int = 30
    lift_steps: int = 90
    transport_steps: int = 130
    descend_steps: int = 100
    release_steps: int = 55
    release_hold_steps: int = 30
    retreat_steps: int = 70
    verify_steps: int = 100
    pregrasp_distance: float = 0.09
    transport_clearance: float = 0.050
    servo_corrections: int = 5
    servo_tolerance: float = 0.005
    orientation_corrections: int = 6
    yaw_corrections: int = 8
    orientation_steps: int = 20
    insertion_levels: int = 6
    maximum_release_open_fraction: float = 1.0

    def validate(self) -> None:
        steps = (
            self.settle_steps,
            self.transit_steps,
            self.pregrasp_steps,
            self.approach_steps,
            self.close_steps,
            self.hold_steps,
            self.lift_steps,
            self.transport_steps,
            self.descend_steps,
            self.release_steps,
            self.release_hold_steps,
            self.retreat_steps,
            self.verify_steps,
        )
        if min(steps) <= 0:
            raise ValueError("Every NutAssembly stage must contain at least one step.")
        if (
            min(
                self.pregrasp_distance,
                self.transport_clearance,
                self.servo_tolerance,
            )
            <= 0.0
        ):
            raise ValueError("NutAssembly distances must be positive.")
        if self.servo_corrections < 1:
            raise ValueError("servo_corrections must be positive.")
        if (
            min(self.orientation_corrections, self.yaw_corrections) < 0
            or min(self.orientation_steps, self.insertion_levels) < 1
        ):
            raise ValueError("NutAssembly correction counts are invalid.")


def _servo_bolt_orientation(
    env,
    recorder: _Recorder,
    *,
    hand: np.ndarray,
    quaternion: np.ndarray,
    corrections: int,
    steps: int,
) -> tuple[np.ndarray, bool, float]:
    """Counter-rotate the wrist from measured bolt orientation error."""

    arm = env.controller.arm_controller
    target_quaternion = np.asarray(quaternion, dtype=np.float64)
    ended = False
    error_angle = float("inf")
    for _ in range(corrections):
        binding = env.task._require_bindings().objects["assembly_bolt"]
        object_rotation = env.data.xmat[binding.body_id].reshape(3, 3).copy()
        bolt_axis = object_rotation[:, 2]
        vertical = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
        cosine = float(np.clip(np.dot(bolt_axis, vertical), -1.0, 1.0))
        error_angle = float(np.arccos(cosine))
        if error_angle <= np.deg2rad(2.0):
            break
        axis = np.cross(bolt_axis, vertical)
        axis_norm = float(np.linalg.norm(axis))
        if axis_norm < 1e-9:
            break
        # The bolt is held by compliant tendon-driven fingers. Large wrist
        # rotations strip or flip that grasp, so correct tilt in 2-degree
        # increments and let the contacts settle between increments.
        rotation_vector = axis * (min(error_angle, np.deg2rad(2.0)) / axis_norm)
        increment = Rotation.from_rotvec(rotation_vector).as_matrix()
        ee_rotation = env.data.site_xmat[arm.site_id].reshape(3, 3).copy()
        target_quaternion = mat_to_quat(increment @ ee_rotation)
        position = env.data.site_xpos[arm.site_id].astype(np.float64).copy()
        _, _, ended = _run_pose_segment(
            env,
            recorder,
            stage="transport",
            target_position=position,
            target_quaternion=target_quaternion,
            start_hand=hand,
            target_hand=hand,
            steps=steps,
            smooth=True,
        )
        if ended:
            break
    binding = env.task._require_bindings().objects["assembly_bolt"]
    bolt_axis = env.data.xmat[binding.body_id].reshape(3, 3)[:, 2]
    error_angle = float(np.arccos(np.clip(bolt_axis[2], -1.0, 1.0)))
    return target_quaternion, ended, error_angle


def _servo_bolt_yaw(
    env,
    recorder: _Recorder,
    *,
    hand: np.ndarray,
    quaternion: np.ndarray,
    polygon_sides: int,
    corrections: int,
    steps: int,
) -> tuple[np.ndarray, bool, float]:
    """Align yaw to the nearest symmetry-equivalent polygon orientation."""

    if polygon_sides == 0 or corrections == 0:
        return np.asarray(quaternion, dtype=np.float64), False, 0.0
    arm = env.controller.arm_controller
    period = 2.0 * np.pi / polygon_sides
    target_quaternion = np.asarray(quaternion, dtype=np.float64)
    ended = False
    residual = float("inf")
    for _ in range(corrections):
        binding = env.task._require_bindings().objects["assembly_bolt"]
        rotation = env.data.xmat[binding.body_id].reshape(3, 3)
        yaw = float(np.arctan2(rotation[1, 0], rotation[0, 0]))
        residual = float((yaw + 0.5 * period) % period - 0.5 * period)
        if abs(residual) <= np.deg2rad(2.0):
            break
        correction = float(np.clip(-residual, -np.deg2rad(8.0), np.deg2rad(8.0)))
        increment = Rotation.from_rotvec([0.0, 0.0, correction]).as_matrix()
        ee_rotation = env.data.site_xmat[arm.site_id].reshape(3, 3).copy()
        target_quaternion = mat_to_quat(increment @ ee_rotation)
        position = env.data.site_xpos[arm.site_id].astype(np.float64).copy()
        _, _, ended = _run_pose_segment(
            env,
            recorder,
            stage="transport",
            target_position=position,
            target_quaternion=target_quaternion,
            start_hand=hand,
            target_hand=hand,
            steps=steps,
            smooth=True,
        )
        if ended:
            break
    return target_quaternion, ended, abs(residual)


def execute_nut_transfer(
    handle_source: DemonstrationEpisode,
    *,
    variant_index: int,
    seed: int = 0,
    grasp_config: ExecutionConfig | None = None,
    transfer_config: NutTransferConfig | None = None,
) -> DemonstrationEpisode:
    """Grasp the common bolt handle, transport it, and insert its shaft."""

    variant = procedural_nut_variant(variant_index)
    bolt_spec = FreeBoltSpec(
        "assembly_bolt",
        polygon_sides=variant.polygon_sides,
        shaft_radius=variant.peg_radius,
    )
    transfer = transfer_config or NutTransferConfig()
    transfer.validate()
    grasp = grasp_config or ExecutionConfig()
    object_id = f"procedural_bolt:{variant_index:03d}"
    source_object_rotation = quaternion_wxyz_to_matrix(
        np.asarray(handle_source.metadata["object_initial_quaternion_wxyz"], dtype=np.float64)
    )
    candidate = replace(
        handle_source.candidate,
        object_id=object_id,
        hand_translation=(
            source_object_rotation @ handle_source.candidate.hand_translation
            + np.asarray([0.0, 0.0, bolt_spec.handle_center_z], dtype=np.float64)
        ),
        hand_rotation_matrix=(
            source_object_rotation @ handle_source.candidate.hand_rotation_matrix
        ),
        contact_points=(handle_source.candidate.contact_points @ source_object_rotation.T),
        backend=f"{handle_source.candidate.backend}+fixed-nut-bolt-transfer",
    )
    # Preserve the proven Lift grasp's horizontal RM75B workspace.  Moving
    # the bolt source to y=-0.10 would shift this wrist target outside the
    # arm's low-table IK envelope before contact is ever established.
    # Rest the mating shaft on the table while keeping the separate
    # handle and shoulder entirely above it.
    initial_position = np.asarray(
        [0.54, -0.040, 0.5 + bolt_spec.bottom_offset + 0.002], dtype=np.float64
    )
    initial_quaternion = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    task_config = {
        "variant_index": variant_index,
        "fixed_bolt_position": tuple(float(value) for value in initial_position),
        "fixed_bolt_quaternion_wxyz": tuple(float(value) for value in initial_quaternion),
        "terminate_on_success": False,
    }
    maximum_steps = 1600
    env = make_nut_assembly_env(
        task_config=task_config,
        control_mode="ik",
        enable_tactile_sensors=True,
        episode_length=maximum_steps,
    )
    try:
        observation, _ = env.reset(seed=seed)
        recorder = _Recorder("assembly_bolt")
        arm = env.controller.arm_controller
        current_position = env.data.site_xpos[arm.site_id].astype(np.float64).copy()
        current_quaternion = mat_to_quat(env.data.site_xmat[arm.site_id])
        approach_fractions, grip_fractions = grasp_hand_targets(candidate.actuator_fractions, grasp)
        open_hand = actuator_targets_from_fractions(env, approach_fractions)
        grip_hand = actuator_targets_from_fractions(env, grip_fractions)
        _, _, ended = _run_pose_segment(
            env,
            recorder,
            stage="settle",
            target_position=current_position,
            target_quaternion=current_quaternion,
            start_hand=env.controller.hand_controller.current_action(env.model, env.data),
            target_hand=open_hand,
            steps=transfer.settle_steps,
            smooth=True,
        )
        bolt_position = env.task._body_pos(env.model, env.data, "assembly_bolt").copy()
        bolt_quaternion = observation["assembly_bolt_quat"].astype(np.float64)
        grasp_position, hand_rotation, grasp_quaternion = candidate_world_pose(
            candidate,
            bolt_position,
            bolt_quaternion,
            hand_attach_rotation(env),
        )
        pregrasp = grasp_position - hand_rotation @ np.asarray(
            [0.0, transfer.pregrasp_distance, 0.0]
        )
        transit = pregrasp.copy()
        transit[2] = max(float(pregrasp[2] + 0.16), 0.78)
        stages = (
            ("transit", transit, open_hand, open_hand, transfer.transit_steps),
            ("pregrasp", pregrasp, open_hand, open_hand, transfer.pregrasp_steps),
            ("approach", grasp_position, open_hand, open_hand, transfer.approach_steps),
            ("close", grasp_position, open_hand, grip_hand, transfer.close_steps),
            ("hold", grasp_position, grip_hand, grip_hand, transfer.hold_steps),
        )
        for stage, position, start_hand, target_hand, steps in stages:
            if ended:
                break
            _, _, ended = _run_pose_segment(
                env,
                recorder,
                stage=stage,
                target_position=position,
                target_quaternion=grasp_quaternion,
                start_hand=start_hand,
                target_hand=target_hand,
                steps=steps,
                smooth=True,
            )
        if not ended:
            grip_hand = _select_transport_hand(env, grip_hand, "assembly_bolt")

        nut_center = env.task._peg_center("assembly_bolt")
        socket_top = env.task.table_top_z + 2.0 * env.task.arena.wall_half_height
        desired_insert = np.asarray(
            [
                nut_center[0],
                nut_center[1],
                socket_top
                - env.task.objects[0].shoulder_center_z
                + env.task.objects[0].shoulder_half_height,
            ],
            dtype=np.float64,
        )
        # Seat the lower face of the physical shoulder just above the socket
        # rim. The shaft, rather than the handle, now fills the opening.
        desired_insert[2] -= 0.001
        desired_transport = desired_insert.copy()
        desired_transport[2] = (
            socket_top + env.task.objects[0].bottom_offset + transfer.transport_clearance
        )
        current_bolt = env.task._body_pos(env.model, env.data, "assembly_bolt").copy()
        desired_lift = current_bolt.copy()
        desired_lift[2] = desired_transport[2]
        errors: dict[str, float] = {}
        if not ended:
            ended, errors["lift"] = _servo_grasped_object(
                env,
                recorder,
                stage="lift",
                desired_object_position=desired_lift,
                quaternion=grasp_quaternion,
                hand=grip_hand,
                total_steps=transfer.lift_steps,
                corrections=transfer.servo_corrections,
                tolerance=transfer.servo_tolerance,
                object_name="assembly_bolt",
            )
        if not ended:
            ended, errors["transport"] = _servo_grasped_object(
                env,
                recorder,
                stage="transport",
                desired_object_position=desired_transport,
                quaternion=grasp_quaternion,
                hand=grip_hand,
                total_steps=transfer.transport_steps,
                corrections=transfer.servo_corrections,
                tolerance=transfer.servo_tolerance,
                horizontal_tolerance=0.003,
                object_name="assembly_bolt",
            )
        if not ended and transfer.yaw_corrections:
            grasp_quaternion, ended, errors["yaw"] = _servo_bolt_yaw(
                env,
                recorder,
                hand=grip_hand,
                quaternion=grasp_quaternion,
                polygon_sides=variant.polygon_sides,
                corrections=transfer.yaw_corrections,
                steps=transfer.orientation_steps,
            )
        if not ended and transfer.orientation_corrections:
            grasp_quaternion, ended, errors["orientation"] = _servo_bolt_orientation(
                env,
                recorder,
                hand=grip_hand,
                quaternion=grasp_quaternion,
                corrections=transfer.orientation_corrections,
                steps=transfer.orientation_steps,
            )
        if not ended:
            # Orientation/yaw adjustment and passive settling can move the
            # held shaft sideways. Re-centre immediately above the socket
            # before any downward command so the rim guides rather than
            # ejects the bolt.
            ended, errors["alignment"] = _servo_grasped_object(
                env,
                recorder,
                stage="transport",
                desired_object_position=desired_transport,
                quaternion=grasp_quaternion,
                hand=grip_hand,
                total_steps=120,
                corrections=20,
                tolerance=0.0015,
                maximum_correction=0.005,
                horizontal_tolerance=0.0025,
                object_name="assembly_bolt",
            )
        if not ended:
            insert_errors: list[float] = []
            start_height = float(env.task._body_pos(env.model, env.data, "assembly_bolt")[2])
            for level in range(1, transfer.insertion_levels + 1):
                fraction = level / transfer.insertion_levels
                level_target = desired_insert.copy()
                level_target[2] = (1.0 - fraction) * start_height + fraction * desired_insert[2]
                ended, level_error = _servo_grasped_object(
                    env,
                    recorder,
                    stage="descend",
                    desired_object_position=level_target,
                    quaternion=grasp_quaternion,
                    hand=grip_hand,
                    total_steps=max(1, transfer.descend_steps // transfer.insertion_levels),
                    corrections=transfer.servo_corrections,
                    tolerance=transfer.servo_tolerance,
                    object_name="assembly_bolt",
                )
                insert_errors.append(level_error)
                if ended:
                    break
                if transfer.orientation_corrections:
                    grasp_quaternion, ended, _ = _servo_bolt_orientation(
                        env,
                        recorder,
                        hand=grip_hand,
                        quaternion=grasp_quaternion,
                        corrections=2,
                        steps=transfer.orientation_steps,
                    )
                    if ended:
                        break
            errors["insert"] = insert_errors[-1] if insert_errors else float("inf")
        # NutAssembly is a held-tool task: the underactuated hand must keep
        # grasping the bolt while it is seated.  Releasing and withdrawing is
        # a different task and can pull a correctly inserted bolt back out.
        current_ee = env.data.site_xpos[arm.site_id].astype(np.float64).copy()
        if not ended:
            ended, errors["seat_verify"] = _servo_grasped_object(
                env,
                recorder,
                stage="task_verify",
                desired_object_position=desired_insert,
                quaternion=grasp_quaternion,
                hand=grip_hand,
                total_steps=transfer.verify_steps,
                corrections=max(transfer.servo_corrections, 6),
                tolerance=0.003,
                object_name="assembly_bolt",
            )
            current_ee = env.data.site_xpos[arm.site_id].astype(np.float64).copy()
        if not ended:
            _, _, ended = _run_pose_segment(
                env,
                recorder,
                stage="task_verify",
                target_position=current_ee,
                target_quaternion=grasp_quaternion,
                start_hand=grip_hand,
                target_hand=grip_hand,
                steps=transfer.verify_steps,
                smooth=True,
            )
        arrays = recorder.arrays()
        verify = arrays["task_success"][arrays["stage"] == STAGE_CODES["task_verify"]]
        # ``NutAssemblyTask`` only becomes true after 20 consecutive frames
        # satisfy insertion, alignment and low-velocity constraints.  The
        # final true sample therefore already proves the required dwell; an
        # additional 80%-of-window threshold would count the task's own
        # warm-up frames twice and reject valid stable insertions.
        success = bool(len(verify) and verify[-1])
        final = env.task._body_pos(env.model, env.data, "assembly_bolt").copy()
        return DemonstrationEpisode(
            object_id=object_id,
            seed=seed,
            candidate=candidate,
            arrays=arrays,
            success=success,
            terminal_stage="task_verify",
            failure_reason=None
            if success
            else "grasped bolt was not stably inserted in the fixed nut",
            metadata={
                "task": "nut_assembly",
                "source_skill": "lift_handle_grasp",
                "pipeline_version": NUT_TRANSFER_VERSION,
                "domain_version": NUT_ASSEMBLY_DOMAIN_VERSION,
                "task_config": task_config,
                "transfer_config": asdict(transfer),
                "variant": asdict(variant),
                "servo_errors": errors,
                "object_final_position": final.tolist(),
                "nut_position": nut_center.tolist(),
                "task_verify_success_fraction": float(verify.mean()) if len(verify) else 0.0,
                "tactile_metadata": env.tactile_sensor.metadata(),
            },
        )
    finally:
        env.close()
