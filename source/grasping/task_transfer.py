"""Transfer an object-relative verified grasp into downstream manipulation tasks."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from source.envs.manipulation import make_pick_place_env
from source.envs.manipulation.arenas import BinsArena
from source.envs.manipulation.placement import FixedTablePlacementSampler
from source.geometry import mat_to_quat
from source.grasping.contracts import DemonstrationEpisode, GraspCandidate
from source.grasping.executor import (
    STAGE_CODES,
    ExecutionConfig,
    _Recorder,
    _run_pose_segment,
    _step,
    actuator_targets_from_fractions,
    execute_grasp,
    grasp_hand_targets,
)

PICK_PLACE_PIPELINE_VERSION = "pick-place-transfer-v3"


@dataclass(frozen=True)
class PickPlaceTransferConfig:
    """Motion budget appended after the reusable grasp-and-lift prefix."""

    clearance_height: float = 0.065
    release_clearance: float = 0.002
    hand_table_clearance: float = 0.006
    release_in_air: bool = False
    table_supported_transport: bool = False
    reuse_verified_lift_terminal_state: bool = True
    replay_lift_from_reset: bool = False
    air_release_extra_height: float = 0.02
    target_near_edge_fraction: float = 0.0
    target_y_bias: float = 0.0
    air_release_target_y_bias: float = 0.02
    transport_grip_boost: float = 0.0
    transport_finger_boost: float = 0.0
    clear_steps: int = 0
    adaptive_transport_grip: bool = True
    adaptive_grip_steps: int = 8
    adaptive_release: bool = True
    maximum_release_open_fraction: float = 0.85
    transport_steps: int = 90
    descend_steps: int = 55
    release_steps: int = 35
    retreat_steps: int = 45
    verify_steps: int = 30

    def validate(self) -> None:
        if self.clearance_height <= 0.0:
            raise ValueError("clearance_height must be positive.")
        if self.release_clearance <= 0.0:
            raise ValueError("release_clearance must be positive.")
        if self.hand_table_clearance < 0.0:
            raise ValueError("hand_table_clearance must be non-negative.")
        if not 0.0 <= self.target_near_edge_fraction <= 0.75:
            raise ValueError("target_near_edge_fraction must lie in [0, 0.75].")
        if not 0.0 <= self.transport_grip_boost <= 0.20:
            raise ValueError("transport_grip_boost must lie in [0, 0.20].")
        if not 0.0 <= self.transport_finger_boost <= 0.40:
            raise ValueError("transport_finger_boost must lie in [0, 0.40].")
        if not 0.0 < self.maximum_release_open_fraction <= 1.0:
            raise ValueError("maximum_release_open_fraction must lie in (0, 1].")
        if (
            min(
                self.transport_steps,
                self.descend_steps,
                self.release_steps,
                self.retreat_steps,
                self.verify_steps,
            )
            <= 0
        ):
            raise ValueError("Every PickPlace transfer stage must contain at least one step.")
        if self.clear_steps < 0 or self.adaptive_grip_steps < 0:
            raise ValueError("Optional PickPlace stage lengths must be non-negative.")

    @property
    def appended_steps(self) -> int:
        return (
            self.clear_steps
            + (self.adaptive_grip_steps if self.adaptive_transport_grip else 0)
            + (self.descend_steps if self.table_supported_transport else 0)
            + self.transport_steps
            + self.descend_steps
            + self.release_steps
            + self.retreat_steps
            + self.verify_steps
        )


def _select_transport_hand(env, grip_hand: np.ndarray) -> np.ndarray:
    """Select a preload that survives a small lateral transport perturbation."""

    arm = env.controller.arm_controller
    hand = env.controller.hand_controller
    qpos = env.data.qpos.copy()
    qvel = env.data.qvel.copy()
    ctrl = env.data.ctrl.copy()
    time = float(env.data.time)
    elapsed = int(env.elapsed_steps)
    previous_q = arm._prev_target_q.copy()
    previous_velocity = arm._filtered_velocity.copy()
    previous_ee = arm._prev_ee_target.copy()
    ee_position = env.data.site_xpos[arm.site_id].copy()
    ee_quaternion = mat_to_quat(env.data.site_xmat[arm.site_id])
    object_position = env.task._body_pos(env.model, env.data, "object").copy()
    base_fraction = np.divide(
        grip_hand - hand.ctrl_low,
        hand.ctrl_high - hand.ctrl_low,
        out=np.zeros(6, dtype=np.float32),
        where=(hand.ctrl_high - hand.ctrl_low) != 0.0,
    )
    # Four underactuated fingers share the same mechanical role while the
    # thumb opposes them. Searching these two preload groups is cheap and
    # avoids assuming that "more closure" always produces a better grasp.
    deltas = (-0.20, 0.0, 0.20, 0.40)
    best_score = float("inf")
    best_hand = grip_hand.copy()
    for finger_delta in deltas:
        for thumb_delta in deltas:
            fraction = base_fraction.copy()
            fraction[:4] += finger_delta
            fraction[5] += thumb_delta
            trial_hand = actuator_targets_from_fractions(env, np.clip(fraction, 0.0, 1.0))
            env.data.qpos[:] = qpos
            env.data.qvel[:] = qvel
            env.data.ctrl[:] = ctrl
            env.data.time = time
            env.elapsed_steps = elapsed
            arm._prev_target_q = previous_q.copy()
            arm._filtered_velocity = previous_velocity.copy()
            arm._prev_ee_target = previous_ee.copy()
            mujoco.mj_forward(env.model, env.data)
            probe = _Recorder()
            _run_pose_segment(
                env,
                probe,
                stage="clear",
                target_position=ee_position,
                target_quaternion=ee_quaternion,
                start_hand=grip_hand,
                target_hand=trial_hand,
                steps=8,
                smooth=True,
            )
            _run_pose_segment(
                env,
                probe,
                stage="clear",
                target_position=ee_position + np.asarray([0.0, 0.004, 0.0]),
                target_quaternion=ee_quaternion,
                start_hand=trial_hand,
                target_hand=trial_hand,
                steps=12,
                smooth=True,
            )
            final_object = env.task._body_pos(env.model, env.data, "object").copy()
            final_ee = env.data.site_xpos[arm.site_id].copy()
            relative_drift = np.linalg.norm(
                (final_object - final_ee) - (object_position - ee_position)
            )
            height_loss = max(0.0, object_position[2] - final_object[2])
            arrays = probe.arrays()
            final_force = float(arrays["robot_object_normal_force"][-1])
            score = 20.0 * relative_drift + 30.0 * height_loss - 0.01 * min(final_force, 10.0)
            if score < best_score:
                best_score = score
                best_hand = trial_hand
    env.data.qpos[:] = qpos
    env.data.qvel[:] = qvel
    env.data.ctrl[:] = ctrl
    env.data.time = time
    env.elapsed_steps = elapsed
    arm._prev_target_q = previous_q
    arm._filtered_velocity = previous_velocity
    arm._prev_ee_target = previous_ee
    mujoco.mj_forward(env.model, env.data)
    return best_hand


def _combine_arrays(
    first: dict[str, np.ndarray], second: dict[str, np.ndarray]
) -> dict[str, np.ndarray]:
    if first.keys() != second.keys():
        raise ValueError("Transferred trajectory recorders have incompatible arrays.")
    return {name: np.concatenate((first[name], second[name]), axis=0) for name in first}


def _hand_collision_min_z(env) -> float:
    """Conservative lower bound of the hand's active collision geometry."""

    prefix = str(getattr(env.controller.hand_controller, "hand_prefix", "") or "")
    minimum = float("inf")
    for geom_id in env.task._require_bindings().robot_geom_ids:
        name = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_GEOM, int(geom_id)) or ""
        if prefix and not name.startswith(prefix):
            continue
        if env.model.geom_contype[geom_id] == 0 and env.model.geom_conaffinity[geom_id] == 0:
            continue
        # MuJoCo's compiled bounding radius contains the complete geom. It is
        # conservative for elongated links, which is preferable to allowing a
        # finger or palm to intersect the table during placement.
        lower = float(env.data.geom_xpos[geom_id, 2] - env.model.geom_rbound[geom_id])
        minimum = min(minimum, lower)
    if not np.isfinite(minimum):
        raise RuntimeError("Could not identify active hand collision geometry.")
    return minimum


def _safe_descent_distance(
    *,
    object_position_z: float,
    object_bottom_offset: float,
    table_top_z: float,
    desired_object_clearance: float,
    hand_min_z: float,
    hand_table_clearance: float,
) -> tuple[float, bool]:
    """Return a downward travel that places the object without striking the table."""

    object_travel = max(
        0.0,
        object_position_z
        - object_bottom_offset
        - table_top_z
        - desired_object_clearance,
    )
    hand_travel = max(0.0, hand_min_z - table_top_z - hand_table_clearance)
    travel = min(object_travel, hand_travel)
    return travel, hand_travel + 1e-9 < object_travel


def _object_touches_table(env) -> bool:
    bindings = env.task._require_bindings()
    object_geoms = bindings.objects["object"].geom_ids
    table_geom = mujoco.mj_name2id(
        env.model, mujoco.mjtObj.mjOBJ_GEOM, env.task.arena.table_geom_name
    )
    if table_geom < 0:
        return False
    for index in range(env.data.ncon):
        pair = {int(env.data.contact[index].geom1), int(env.data.contact[index].geom2)}
        if table_geom in pair and pair.intersection(object_geoms):
            return True
    return False


def _run_adaptive_release(
    env,
    recorder: _Recorder,
    *,
    position: np.ndarray,
    quaternion: np.ndarray,
    grip_hand: np.ndarray,
    open_hand: np.ndarray,
    steps: int,
    maximum_open_fraction: float,
) -> tuple[np.ndarray, bool, float]:
    """Loosen only until the object has settled and separated from the hand."""

    final_hand = grip_hand.copy()
    final_fraction = 0.0
    ended = False
    for index in range(steps):
        phase = (index + 1) / steps
        fraction = maximum_open_fraction * phase * phase * (3.0 - 2.0 * phase)
        final_fraction = fraction
        final_hand = (1.0 - fraction) * grip_hand + fraction * open_hand
        action = np.concatenate((position, quaternion, final_hand)).astype(np.float32)
        _, _, ended = _step(env, recorder, action, "release")
        touching_hand = env.task._is_robot_touching_object(env.model, env.data, "object")
        if ended or (_object_touches_table(env) and not touching_hand):
            break
    return final_hand.astype(np.float32), ended, final_fraction


def execute_pick_place_transfer(
    candidate: GraspCandidate,
    *,
    seed: int = 0,
    grasp_config: ExecutionConfig | None = None,
    transfer_config: PickPlaceTransferConfig | None = None,
    source_object_position: np.ndarray | None = None,
    source_object_quaternion_wxyz: np.ndarray | None = None,
    source_episode: DemonstrationEpisode | None = None,
    source_trajectory: Any | None = None,
    render_mode: str | None = None,
) -> DemonstrationEpisode:
    """Retarget a Lift grasp and append transport, release, and verification."""

    transfer = transfer_config or PickPlaceTransferConfig()
    transfer.validate()
    base_grasp = grasp_config or ExecutionConfig()
    # Preserve the exact verified Lift height. A separate suffix segment raises
    # the already-held object above the target-bin wall before translation.
    grasp = replace(base_grasp)
    grasp.validate()
    episode_length = grasp.maximum_steps + transfer.appended_steps + 20
    task_config: dict[str, Any] = {
        "object_id": candidate.object_id,
        "reward_shaping": False,
        "terminate_on_success": False,
    }
    if source_object_position is not None or source_object_quaternion_wxyz is not None:
        if source_object_position is None or source_object_quaternion_wxyz is None:
            raise ValueError("Both source object position and quaternion are required.")
        position = np.asarray(source_object_position, dtype=np.float64)
        quaternion = np.asarray(source_object_quaternion_wxyz, dtype=np.float64)
        if position.shape != (3,) or quaternion.shape != (4,):
            raise ValueError("Source object pose must have shapes (3,) and (4,).")
        arena = BinsArena()
        task_config["arena"] = arena
        placement_xy = (
            float(position[0] - arena.source_center[0]),
            float(position[1] - arena.source_center[1]),
        )
        if transfer.replay_lift_from_reset:
            yaw = Rotation.from_quat(quaternion[[1, 2, 3, 0]]).as_euler("xyz")[2]
            task_config["placement_sampler"] = FixedTablePlacementSampler(
                xy=placement_xy,
                yaw=float(yaw),
            )
        else:
            task_config["placement_sampler"] = FixedTablePlacementSampler(
                xy=placement_xy,
                quaternion_wxyz=quaternion,
                world_z=float(position[2]),
            )
    env = make_pick_place_env(
        task_config=task_config,
        control_mode="ik",
        enable_tactile_sensors=grasp.enable_tactile_sensors,
        episode_length=episode_length,
        render_mode=render_mode,
    )
    try:
        if source_episode is None:
            prefix = execute_grasp(candidate, seed=seed, config=grasp, environment=env)
        else:
            env.reset(seed=seed)
            prefix_recorder = _Recorder()
            stage_names = {value: name for name, value in STAGE_CODES.items()}
            source_actions = source_episode.arrays["action"]
            source_controls = source_episode.arrays["ctrl"]
            source_stages = source_episode.arrays["stage"]
            settle_indices = np.flatnonzero(source_stages == STAGE_CODES["settle"])
            if len(settle_indices) and not transfer.replay_lift_from_reset:
                settled_index = int(settle_indices[-1])
                source_qpos = source_episode.arrays["qpos"][settled_index]
                source_qvel = source_episode.arrays["qvel"][settled_index]
                source_ctrl = source_controls[settled_index]
                if source_qpos.shape != env.data.qpos.shape:
                    raise ValueError("Lift and PickPlace qpos layouts are incompatible.")
                if source_qvel.shape != env.data.qvel.shape:
                    raise ValueError("Lift and PickPlace qvel layouts are incompatible.")
                if source_ctrl.shape != env.data.ctrl.shape:
                    raise ValueError("Lift and PickPlace ctrl layouts are incompatible.")
                env.data.qpos[:] = source_qpos
                env.data.qvel[:] = source_qvel
                env.data.ctrl[:] = source_ctrl
                mujoco.mj_forward(env.model, env.data)
            # ``object_initial_*`` is captured after the source Lift settle
            # stage. Restore that complete physical state (including object
            # velocity and passive hand state), then continue at transit.
            if not transfer.replay_lift_from_reset:
                after_settle = source_stages != STAGE_CODES["settle"]
                source_actions = source_actions[after_settle]
                source_controls = source_controls[after_settle]
                source_stages = source_stages[after_settle]
            if source_trajectory is not None:
                approach_code = STAGE_CODES[str(source_trajectory.start_stage)]
                keep = source_stages < approach_code
                source_actions = source_actions[keep]
                source_controls = source_controls[keep]
                source_stages = source_stages[keep]
            for action, control, stage_code in zip(
                source_actions, source_controls, source_stages, strict=True
            ):
                stage = stage_names[int(stage_code)]
                # Reusing the verified low-level controls preserves the exact
                # arm IK branch. Re-solving identical Cartesian actions in a
                # scene with extra target-bin bodies can select a different
                # branch and miss the grasp despite an identical object pose.
                env.data.ctrl[:] = control
                env.step_physics(env.physics_steps_per_control, control_updates=1)
                env.elapsed_steps += 1
                observation = env._get_observation()
                task_result = env.task.evaluate(
                    observation, np.asarray(action), env.model, env.data
                )
                prefix_recorder.append(
                    env,
                    observation,
                    action,
                    stage,
                    task_result.reward,
                    task_result.success,
                )
                ended = bool(
                    task_result.terminated or env.elapsed_steps >= env.config.episode_length
                )
                if ended:
                    break
            if source_trajectory is not None and not ended:
                env.data.qpos[:] = source_trajectory.initial_qpos
                env.data.qvel[:] = source_trajectory.initial_qvel
                env.data.ctrl[:] = source_trajectory.controls[0]
                mujoco.mj_forward(env.model, env.data)
                stage_lengths = (
                    ("approach", grasp.approach_steps),
                    ("close", grasp.close_steps),
                    ("hold", grasp.hold_steps),
                    ("lift", grasp.lift_steps),
                    ("verify", grasp.verify_steps),
                )
                trajectory_index = 0
                for stage_name, stage_length in stage_lengths:
                    for _ in range(stage_length):
                        if trajectory_index >= len(source_trajectory.controls):
                            break
                        control = source_trajectory.controls[trajectory_index]
                        env.data.ctrl[:] = control
                        env.step_physics(env.physics_steps_per_control, control_updates=1)
                        env.elapsed_steps += 1
                        observation = env._get_observation()
                        arm_action = env.controller.arm_controller.current_action(
                            env.model, env.data
                        )
                        action = np.concatenate((arm_action, control[-6:])).astype(np.float32)
                        task_result = env.task.evaluate(observation, action, env.model, env.data)
                        prefix_recorder.append(
                            env,
                            observation,
                            action,
                            stage_name,
                            task_result.reward,
                            task_result.success,
                        )
                        trajectory_index += 1
                    if trajectory_index >= len(source_trajectory.controls):
                        break
            prefix = DemonstrationEpisode(
                object_id=candidate.object_id,
                seed=seed,
                candidate=candidate,
                arrays=prefix_recorder.arrays(),
                success=False,
                terminal_stage=stage,
                metadata={
                    **source_episode.metadata,
                    "replayed_lift_prefix": True,
                    "replayed_ppo_controls": source_trajectory is not None,
                },
            )
            if (
                source_trajectory is None
                and transfer.reuse_verified_lift_terminal_state
                and not transfer.replay_lift_from_reset
            ):
                # Direct/Lattice Lift episodes already contain a complete C
                # MuJoCo-verified physical trajectory. Preserve that prefix
                # verbatim and continue from its exact terminal state instead
                # of accepting numerical divergence from a second replay.
                final_qpos = source_episode.arrays["qpos"][-1]
                final_qvel = source_episode.arrays["qvel"][-1]
                final_ctrl = source_episode.arrays["ctrl"][-1]
                if final_qpos.shape != env.data.qpos.shape:
                    raise ValueError("Lift and PickPlace qpos layouts are incompatible.")
                if final_qvel.shape != env.data.qvel.shape:
                    raise ValueError("Lift and PickPlace qvel layouts are incompatible.")
                if final_ctrl.shape != env.data.ctrl.shape:
                    raise ValueError("Lift and PickPlace ctrl layouts are incompatible.")
                env.data.qpos[:] = final_qpos
                env.data.qvel[:] = final_qvel
                env.data.ctrl[:] = final_ctrl
                mujoco.mj_forward(env.model, env.data)
                prefix = DemonstrationEpisode(
                    object_id=source_episode.object_id,
                    seed=source_episode.seed,
                    candidate=source_episode.candidate,
                    arrays={
                        name: np.asarray(values).copy()
                        for name, values in source_episode.arrays.items()
                    },
                    success=False,
                    terminal_stage=source_episode.terminal_stage,
                    metadata={
                        **source_episode.metadata,
                        "reused_verified_lift_state": True,
                    },
                )
        initial_object = np.asarray(prefix.arrays["object_position"][0], dtype=np.float64)
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
        arm._prev_target_q = env.data.qpos[arm.qpos_addrs].copy()
        arm._filtered_velocity = np.zeros_like(arm._prev_target_q)
        arm._prev_ee_target = np.concatenate((ee_position, ee_quaternion))
        approach_fractions, _ = grasp_hand_targets(candidate.actuator_fractions, grasp)
        open_hand = actuator_targets_from_fractions(env, approach_fractions)
        grip_hand = env.controller.hand_controller.current_action(env.model, env.data).astype(
            np.float32
        )
        hand_controller = env.controller.hand_controller
        grip_fraction = np.divide(
            grip_hand - hand_controller.ctrl_low,
            hand_controller.ctrl_high - hand_controller.ctrl_low,
            out=np.zeros(6, dtype=np.float32),
            where=(hand_controller.ctrl_high - hand_controller.ctrl_low) != 0.0,
        )
        # Keep opposing-side boosts independently tunable: symmetric closure
        # can eject round objects, while thin objects may need extra finger
        # preload to survive lateral acceleration.
        boost = np.asarray(
            [
                transfer.transport_finger_boost,
                transfer.transport_finger_boost,
                transfer.transport_finger_boost,
                transfer.transport_finger_boost,
                0.0,
                transfer.transport_grip_boost,
            ],
            dtype=np.float32,
        )
        transport_hand = actuator_targets_from_fractions(
            env,
            np.clip(grip_fraction + boost, 0.0, 1.0),
        )
        if transfer.adaptive_transport_grip:
            transport_hand = _select_transport_hand(env, grip_hand)

        target_center = env.task._target_center()
        target_xy = target_center[:2].copy()
        target_xy[1] -= transfer.target_near_edge_fraction * env.task.arena.bin_half_size[1]
        target_xy[1] += (
            transfer.air_release_target_y_bias
            if transfer.release_in_air
            else transfer.target_y_bias
        )
        desired_release_object_position = np.asarray(
            [
                target_xy[0],
                target_xy[1],
                env.task.table_top_z
                + env.task.objects[0].bottom_offset
                + transfer.release_clearance,
            ],
            dtype=np.float64,
        )
        if transfer.release_in_air:
            desired_release_object_position[2] = (
                current_object[2] + transfer.air_release_extra_height
            )
        ee_object_offset = ee_position - current_object
        transport_object_position = desired_release_object_position.copy()
        transport_object_position[2] = current_object[2]
        transport_ee = transport_object_position + ee_object_offset
        clear_ee = ee_position.copy()

        ended = False
        if transfer.adaptive_transport_grip and transfer.adaptive_grip_steps:
            _, _, ended = _run_pose_segment(
                env,
                recorder,
                stage="clear",
                target_position=ee_position,
                target_quaternion=ee_quaternion,
                start_hand=grip_hand,
                target_hand=transport_hand,
                steps=transfer.adaptive_grip_steps,
                smooth=True,
            )
        if transfer.clear_steps:
            _, _, ended = _run_pose_segment(
                env,
                recorder,
                stage="clear",
                target_position=clear_ee,
                target_quaternion=ee_quaternion,
                start_hand=grip_hand,
                target_hand=transport_hand,
                steps=transfer.clear_steps,
            )
        if not ended and transfer.table_supported_transport:
            supported_descent, _ = _safe_descent_distance(
                object_position_z=float(current_object[2]),
                object_bottom_offset=float(env.task.objects[0].bottom_offset),
                table_top_z=float(env.task.table_top_z),
                desired_object_clearance=transfer.release_clearance,
                hand_min_z=_hand_collision_min_z(env),
                hand_table_clearance=transfer.hand_table_clearance,
            )
            supported_source_ee = ee_position.copy()
            supported_source_ee[2] -= supported_descent
            _, _, ended = _run_pose_segment(
                env,
                recorder,
                stage="descend",
                target_position=supported_source_ee,
                target_quaternion=ee_quaternion,
                start_hand=transport_hand,
                target_hand=transport_hand,
                steps=transfer.descend_steps,
                smooth=True,
            )
            supported_ee = env.data.site_xpos[arm.site_id].astype(np.float64).copy()
            supported_object = env.task._body_pos(env.model, env.data, "object").copy()
            supported_offset = supported_ee - supported_object
            transport_object_position[2] = supported_object[2]
            transport_ee = transport_object_position + supported_offset
        if not ended:
            _, _, ended = _run_pose_segment(
                env,
                recorder,
                stage="transport",
                target_position=transport_ee,
                target_quaternion=ee_quaternion,
                start_hand=(
                    transport_hand if transfer.adaptive_transport_grip else grip_hand
                ),
                target_hand=transport_hand,
                steps=transfer.transport_steps,
                smooth=True,
            )

        # The object can settle or slide inside an underactuated grasp during
        # transport. Re-measure the relative pose here instead of reusing the
        # offset captured immediately after Lift. Then cap the downward motion
        # at the first of object placement or hand/table clearance.
        transported_ee = env.data.site_xpos[arm.site_id].astype(np.float64).copy()
        transported_object = env.task._body_pos(env.model, env.data, "object").copy()
        transported_offset = transported_ee - transported_object
        release_ee = desired_release_object_position + transported_offset
        table_limited_descent = False
        if not transfer.release_in_air:
            descent_distance, table_limited_descent = _safe_descent_distance(
                object_position_z=float(transported_object[2]),
                object_bottom_offset=float(env.task.objects[0].bottom_offset),
                table_top_z=float(env.task.table_top_z),
                desired_object_clearance=transfer.release_clearance,
                hand_min_z=_hand_collision_min_z(env),
                hand_table_clearance=transfer.hand_table_clearance,
            )
            release_ee[2] = transported_ee[2] - descent_distance

        released_hand = open_hand
        release_open_fraction = 1.0
        if not ended:
            _, _, ended = _run_pose_segment(
                env,
                recorder,
                stage="descend",
                target_position=release_ee,
                target_quaternion=ee_quaternion,
                start_hand=transport_hand,
                target_hand=transport_hand,
                steps=transfer.descend_steps,
                smooth=True,
            )
        if not ended and transfer.adaptive_release:
            released_hand, ended, release_open_fraction = _run_adaptive_release(
                env,
                recorder,
                position=release_ee,
                quaternion=ee_quaternion,
                grip_hand=transport_hand,
                open_hand=open_hand,
                steps=transfer.release_steps,
                maximum_open_fraction=transfer.maximum_release_open_fraction,
            )
        elif not ended:
            _, _, ended = _run_pose_segment(
                env,
                recorder,
                stage="release",
                target_position=release_ee,
                target_quaternion=ee_quaternion,
                start_hand=transport_hand,
                target_hand=open_hand,
                steps=transfer.release_steps,
                smooth=True,
            )

        for stage, position, start_hand, target_hand, steps in (
            (
                "retreat",
                release_ee + np.asarray([0.0, 0.0, 0.12]),
                released_hand,
                released_hand,
                transfer.retreat_steps,
            ),
            (
                "task_verify",
                release_ee + np.asarray([0.0, 0.0, 0.12]),
                released_hand,
                released_hand,
                transfer.verify_steps,
            ),
        ):
            if ended:
                break
            _, _, ended = _run_pose_segment(
                env,
                recorder,
                stage=stage,
                target_position=position,
                target_quaternion=ee_quaternion,
                start_hand=start_hand,
                target_hand=target_hand,
                steps=steps,
                smooth=True,
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
                "source_object_position": initial_object.tolist(),
                "transfer_config": asdict(transfer),
                "target_object_position": desired_release_object_position.tolist(),
                "transported_object_position": transported_object.tolist(),
                "transported_ee_object_offset": transported_offset.tolist(),
                "release_ee_position": release_ee.tolist(),
                "released_hand_target": released_hand.tolist(),
                "release_open_fraction": release_open_fraction,
                "table_limited_descent": table_limited_descent,
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
            failure_reason=None if success else "object was not released stably inside target region",
            metadata=metadata,
        )
    finally:
        env.close()
