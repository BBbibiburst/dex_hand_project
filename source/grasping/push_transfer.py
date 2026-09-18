"""Generate a tactile planar Push with the four straight fingers as one plane."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import mujoco
import numpy as np

from source.envs.manipulation import make_push_env
from source.geometry import mat_to_quat
from source.grasping.contracts import DemonstrationEpisode
from source.grasping.executor import (
    ExecutionConfig,
    STAGE_CODES,
    _Recorder,
    _run_pose_segment,
)

PUSH_TRANSFER_VERSION = "four-finger-tactile-push-v30"


@dataclass(frozen=True)
class PushTransferConfig:
    """Geometry, tactile threshold, and motion budget for four-finger pushing."""

    distance: float = 0.12
    direction_degrees: float = 270.0
    finger_plane_roll_degrees: float = 225.0
    precontact_distance: float = 0.10
    predicted_contact_gap: float = 0.006
    contact_probe_increment: float = 0.001
    initial_maximum_contact_probe: float = 0.15
    maximum_contact_probe: float = 0.040
    push_increment: float = 0.004
    maximum_push_distance: float = 0.28
    target_stop_radius: float = 0.035
    post_success_push_distance: float = 0.0
    contact_height_fraction: float = 0.50
    minimum_contact_height: float = 0.055
    tactile_threshold: float = 1.0e-6
    required_finger_contact_frames: int = 3
    settle_steps: int = 25
    transit_steps: int = 80
    precontact_steps: int = 70
    increment_steps: int = 8
    retreat_steps: int = 70
    verify_steps: int = 80

    def validate(self) -> None:
        distances = (
            self.distance,
            self.precontact_distance,
            self.contact_probe_increment,
            self.initial_maximum_contact_probe,
            self.maximum_contact_probe,
            self.push_increment,
            self.maximum_push_distance,
            self.target_stop_radius,
            self.post_success_push_distance,
        )
        if min(distances[:-1]) <= 0.0 or self.post_success_push_distance < 0.0:
            raise ValueError("Push distances must be positive; post-success distance may be zero.")
        if self.maximum_push_distance <= self.distance:
            raise ValueError("maximum_push_distance must exceed distance.")
        if not 0.0 <= self.contact_height_fraction <= 1.0:
            raise ValueError("contact_height_fraction must lie in [0, 1].")
        if self.minimum_contact_height <= 0.0:
            raise ValueError("minimum_contact_height must be positive.")
        if self.required_finger_contact_frames < 1:
            raise ValueError("required_finger_contact_frames must be positive.")
        if (
            min(
                self.settle_steps,
                self.transit_steps,
                self.precontact_steps,
                self.increment_steps,
                self.retreat_steps,
                self.verify_steps,
            )
            <= 0
        ):
            raise ValueError("Every Push stage must contain at least one step.")


def _palm_geometry_in_ee_frame(env) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return palm centre, outward normal, and long axis in the EE frame."""

    prefix = str(getattr(env, "hand_prefix", "") or "")
    names = tuple(dict.fromkeys((f"{prefix}skin_palm_p", "skin_palm_p")))
    palm_id = next(
        (
            geom_id
            for name in names
            if (geom_id := mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, name)) >= 0
        ),
        -1,
    )
    if palm_id < 0:
        raise ValueError("Dex Hand palm collision geometry was not found.")
    arm = env.controller.arm_controller
    mujoco.mj_forward(env.model, env.data)
    ee_rotation = env.data.site_xmat[arm.site_id].reshape(3, 3).copy()
    palm_rotation = env.data.geom_xmat[palm_id].reshape(3, 3).copy()
    centre = ee_rotation.T @ (env.data.geom_xpos[palm_id] - env.data.site_xpos[arm.site_id])
    local_rotation = ee_rotation.T @ palm_rotation
    # The STL's local +X axis points through the palm towards the hand back,
    # not out of the tactile face.  Using +X therefore presented the dorsum
    # to the object and allowed the object to reach the palm skin from behind.
    # Likewise, local +Z puts the thumb below the palm in the vertical pushing
    # pose.  Flip both axes: the real palm skin faces the object and the thumb
    # stays above the table.
    normal = -local_rotation[:, 0]
    long_axis = -local_rotation[:, 2]
    normal /= np.linalg.norm(normal)
    long_axis -= normal * float(np.dot(normal, long_axis))
    long_axis /= np.linalg.norm(long_axis)
    return centre, normal, long_axis


def _palm_facing_rotation(
    direction: np.ndarray,
    normal_local: np.ndarray,
    long_local: np.ndarray,
    *,
    roll_degrees: float = 225.0,
) -> np.ndarray:
    """Orient a tactile face along ``direction`` with its span vertical."""

    push = np.asarray(direction, dtype=np.float64)
    push /= np.linalg.norm(push)
    vertical = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
    side_world = np.cross(vertical, push)
    side_world /= np.linalg.norm(side_world)
    side_local = np.cross(long_local, normal_local)
    side_local /= np.linalg.norm(side_local)
    roll = np.deg2rad(roll_degrees)
    span_world = np.cos(roll) * vertical + np.sin(roll) * side_world
    third_world = np.cross(span_world, push)
    third_world /= np.linalg.norm(third_world)
    world_basis = np.column_stack((push, span_world, third_world))
    local_basis = np.column_stack((normal_local, long_local, side_local))
    return world_basis @ local_basis.T


def _four_finger_geometry_in_ee_frame(
    env,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit the centre, outward normal, and span of the four tactile surfaces.

    The four skin meshes share their owning link/body origins, so ``geom_xpos``
    is not the surface centre and is identical for several pads. The generated
    taxel sites lie on the fitted STL surfaces and carry outward-facing local
    Z axes; use those authoritative positions instead.
    """

    prefix = str(getattr(env, "hand_prefix", "") or "")
    arm = env.controller.arm_controller
    mujoco.mj_forward(env.model, env.data)
    ee_position = env.data.site_xpos[arm.site_id]
    ee_rotation = env.data.site_xmat[arm.site_id].reshape(3, 3)
    centres: list[np.ndarray] = []
    normals: list[np.ndarray] = []
    for digit in range(4):
        site_prefixes = tuple(
            dict.fromkeys(
                (
                    f"{prefix}taxel_skin_{digit}_1_p_",
                    f"taxel_skin_{digit}_1_p_",
                )
            )
        )
        site_ids = [
            site_id
            for site_id in range(env.model.nsite)
            if any(
                (mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_SITE, site_id) or "").startswith(
                    candidate
                )
                for candidate in site_prefixes
            )
        ]
        if not site_ids:
            raise ValueError(f"Dex Hand middle finger taxels for digit {digit} were not found.")
        world_centres = env.data.site_xpos[site_ids]
        world_normals = env.data.site_xmat[site_ids].reshape(-1, 3, 3)[:, :, 2]
        centres.append(ee_rotation.T @ (world_centres.mean(axis=0) - ee_position))
        normal = ee_rotation.T @ world_normals.mean(axis=0)
        normals.append(normal / np.linalg.norm(normal))
    centre = np.mean(centres, axis=0)
    normal = np.mean(normals, axis=0)
    normal /= np.linalg.norm(normal)
    span = centres[-1] - centres[0]
    span -= normal * float(np.dot(span, normal))
    span /= np.linalg.norm(span)
    return centre, normal, span


def _four_finger_tactile_indices(env) -> np.ndarray:
    patches = dict(env.tactile_sensor.metadata().get("patches", {}))
    indices: list[int] = []
    for digit in range(4):
        for segment in range(3):
            name = f"skin_{digit}_{segment}_p"
            patch = patches.get(name)
            if not isinstance(patch, dict):
                raise ValueError(f"Tactile patch {name} is unavailable.")
            start, stop = patch["flat_slice"]
            indices.extend(range(int(start), int(stop)))
    return np.asarray(indices, dtype=np.int32)


def _four_finger_object_contact(env, object_name: str = "object") -> bool:
    bindings = env.task._require_bindings()
    object_geoms = set(int(value) for value in bindings.objects[object_name].geom_ids)
    for index in range(env.data.ncon):
        contact = env.data.contact[index]
        pair = {int(contact.geom1), int(contact.geom2)}
        if not pair.intersection(object_geoms):
            continue
        for geom_id in pair - object_geoms:
            name = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
            if any(f"skin_{digit}_" in name for digit in range(4)):
                return True
    return False


def _hand_table_contact_frames(env, arrays: dict[str, np.ndarray], mask: np.ndarray) -> int:
    """Count recorded frames where any Dex Hand geom touches the table."""

    table_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, "table_collision")
    if table_id < 0:
        raise ValueError("Push scene has no table_collision geom.")
    count = 0
    for frame in np.flatnonzero(mask):
        env.data.qpos[:] = arrays["qpos"][frame]
        env.data.qvel[:] = arrays["qvel"][frame]
        env.data.ctrl[:] = arrays["ctrl"][frame]
        mujoco.mj_forward(env.model, env.data)
        touching = False
        for index in range(env.data.ncon):
            contact = env.data.contact[index]
            pair = {int(contact.geom1), int(contact.geom2)}
            if table_id not in pair:
                continue
            other = next(iter(pair - {table_id}), -1)
            name = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_GEOM, other) or ""
            if "dexhand_" in name or name.startswith("skin_"):
                touching = True
                break
        count += int(touching)
    return count


def execute_push_transfer(
    source: DemonstrationEpisode,
    *,
    seed: int = 0,
    grasp_config: ExecutionConfig | None = None,
    transfer_config: PushTransferConfig | None = None,
) -> DemonstrationEpisode:
    """Push one object with measured four-finger contact and tactile feedback."""

    transfer = transfer_config or PushTransferConfig()
    transfer.validate()
    _ = grasp_config
    object_position = np.asarray(source.metadata["object_initial_position"], dtype=np.float64)
    object_quaternion = np.asarray(
        source.metadata["object_initial_quaternion_wxyz"], dtype=np.float64
    )
    angle = np.deg2rad(transfer.direction_degrees)
    direction = np.asarray([np.cos(angle), np.sin(angle), 0.0], dtype=np.float64)
    target = object_position.copy()
    target[:2] += transfer.distance * direction[:2]
    target[2] = 0.501
    task_config = {
        "object_id": source.object_id,
        "fixed_object_position": tuple(float(x) for x in object_position),
        "fixed_object_quaternion_wxyz": tuple(float(x) for x in object_quaternion),
        "fixed_target_position": tuple(float(x) for x in target),
        "terminate_on_success": False,
    }
    increments = int(transfer.maximum_push_distance / transfer.push_increment)
    probes = int(transfer.initial_maximum_contact_probe / transfer.contact_probe_increment)
    maximum_steps = 400 + (increments + probes) * transfer.increment_steps
    env = make_push_env(
        task_config=task_config,
        control_mode="ik",
        enable_tactile_sensors=True,
        tactile_options={
            # Slight overlap prevents gaps between the discretised curved
            # finger skin boxes. It changes sensing coverage only, not the
            # collision geometry used by the physical-contact requirement.
            "taxel_overlap": 2.2,
        },
        episode_length=maximum_steps,
    )
    try:
        env.reset(seed=seed)
        recorder = _Recorder()
        arm = env.controller.arm_controller
        finger_tactile_indices = _four_finger_tactile_indices(env)
        finger_center_local, finger_normal_local, finger_span_local = (
            _four_finger_geometry_in_ee_frame(env)
        )
        finger_rotation = _palm_facing_rotation(
            direction,
            finger_normal_local,
            finger_span_local,
            roll_degrees=transfer.finger_plane_roll_degrees,
        )
        finger_quaternion = mat_to_quat(finger_rotation)
        hand = env.controller.hand_controller
        # Exact requested planar hand pose: four straight fingers and the
        # thumb rotated completely out of the pushing plane.
        planar_hand = np.asarray([0.0, 0.0, 0.0, 0.0, -0.0069, 0.0], dtype=np.float32)
        planar_hand = np.clip(planar_hand, hand.ctrl_low, hand.ctrl_high)
        current_ee = env.data.site_xpos[arm.site_id].astype(np.float64).copy()
        current_quaternion = mat_to_quat(env.data.site_xmat[arm.site_id])
        current_hand = env.controller.hand_controller.current_action(env.model, env.data)
        _, _, ended = _run_pose_segment(
            env,
            recorder,
            stage="settle",
            target_position=current_ee,
            target_quaternion=current_quaternion,
            start_hand=current_hand,
            target_hand=planar_hand,
            steps=transfer.settle_steps,
            smooth=True,
        )
        settled_object = env.task._body_pos(env.model, env.data, "object").copy()
        finger_contact = settled_object.copy()
        # Aim the centre of the four-finger surface at the object's body
        # centre. Using ``object_top - 10 mm`` targets bottle caps, ring holes,
        # and other narrow upper features instead of the load-bearing body.
        # The floor keeps the complete finger plane and rotated-away thumb
        # clear of the table for very low objects.
        half_height = float(env.task.objects[0].bottom_offset)
        object_bottom = float(settled_object[2] - half_height)
        contact_height = object_bottom + 2.0 * half_height * transfer.contact_height_fraction
        finger_contact[2] = max(
            contact_height, env.task.table_top_z + transfer.minimum_contact_height
        )
        finger_contact[:2] -= direction[:2] * (
            env.task.objects[0].horizontal_radius + transfer.predicted_contact_gap
        )
        contact_ee = finger_contact - finger_rotation @ finger_center_local
        precontact_ee = contact_ee - direction * transfer.precontact_distance
        transit_ee = precontact_ee.copy()
        transit_ee[2] += 0.14
        for stage, position, steps in (
            ("transit", transit_ee, transfer.transit_steps),
            ("pregrasp", precontact_ee, transfer.precontact_steps),
        ):
            if ended:
                break
            _, _, ended = _run_pose_segment(
                env,
                recorder,
                stage=stage,
                target_position=position,
                target_quaternion=finger_quaternion,
                start_hand=planar_hand,
                target_hand=planar_hand,
                steps=steps,
                smooth=True,
            )

        last_ee = precontact_ee.copy()
        if not ended:
            for index in range(probes):
                last_ee = precontact_ee + direction * transfer.contact_probe_increment * (index + 1)
                _, _, ended = _run_pose_segment(
                    env,
                    recorder,
                    stage="approach",
                    target_position=last_ee,
                    target_quaternion=finger_quaternion,
                    start_hand=planar_hand,
                    target_hand=planar_hand,
                    steps=transfer.increment_steps,
                    smooth=True,
                )
                physical_finger_contact = _four_finger_object_contact(env)
                # Taxel boxes extend a few millimetres beyond the collision
                # mesh. Do not stop the approach on proximity alone: require
                # contact with an actual non-thumb finger skin geom.
                if ended or physical_finger_contact:
                    break

        push_direction = direction.copy()
        feedback_probe = 0.0
        locked_success_direction: np.ndarray | None = None
        post_success_distance = 0.0
        if not ended:
            for _ in range(increments):
                current = env.task._body_pos(env.model, env.data, "object").copy()
                target_delta = target[:2] - current[:2]
                target_distance = float(np.linalg.norm(target_delta))
                if (
                    locked_success_direction is not None
                    and post_success_distance >= transfer.post_success_push_distance
                ):
                    break
                if locked_success_direction is not None:
                    push_direction = locked_success_direction
                elif target_distance <= 1.0e-9:
                    push_direction = direction.copy()
                else:
                    push_direction = np.asarray(
                        [
                            target_delta[0] / target_distance,
                            target_delta[1] / target_distance,
                            0.0,
                        ],
                        dtype=np.float64,
                    )
                if (
                    locked_success_direction is None
                    and env.task._stable_count >= env.task.stable_steps
                ):
                    if transfer.post_success_push_distance == 0.0:
                        break
                    # Lock the current approach direction before adding a
                    # small stability margin. Re-aiming at tiny residual XY
                    # errors here can rotate the hand sharply and put the
                    # thumb or finger edge on the table.
                    locked_success_direction = push_direction.copy()
                # Non-convex and round objects can redirect a fixed-axis
                # push. Steer the finger plane from measured object motion while
                # preserving the current four-finger plane, then advance it
                # toward the target by one bounded increment.
                touching = _four_finger_object_contact(env)
                if touching:
                    finger_world = last_ee + finger_rotation @ finger_center_local
                else:
                    finger_world = current.copy()
                    finger_world[2] = finger_contact[2]
                    finger_world[:2] -= push_direction[:2] * (
                        env.task.objects[0].horizontal_radius + transfer.predicted_contact_gap
                    )
                finger_rotation = _palm_facing_rotation(
                    push_direction,
                    finger_normal_local,
                    finger_span_local,
                    roll_degrees=transfer.finger_plane_roll_degrees,
                )
                finger_quaternion = mat_to_quat(finger_rotation)
                # Inside the target region, hold the finger plane stationary and let
                # the object dissipate momentum. If it drifts out, the next
                # iteration resumes feedback correction.
                advance = (
                    transfer.push_increment
                    if locked_success_direction is not None
                    else (
                        0.0
                        if target_distance <= transfer.target_stop_radius
                        else transfer.push_increment
                    )
                )
                finger_world += push_direction * (advance + feedback_probe)
                last_ee = finger_world - finger_rotation @ finger_center_local
                _, _, ended = _run_pose_segment(
                    env,
                    recorder,
                    stage="transport",
                    target_position=last_ee,
                    target_quaternion=finger_quaternion,
                    start_hand=planar_hand,
                    target_hand=planar_hand,
                    steps=transfer.increment_steps,
                    smooth=True,
                )
                if _four_finger_object_contact(env):
                    feedback_probe = 0.0
                else:
                    feedback_probe = min(
                        feedback_probe + transfer.contact_probe_increment,
                        transfer.maximum_contact_probe,
                    )
                if locked_success_direction is not None:
                    post_success_distance += advance
                if ended:
                    break

        # Once the object is inside the goal, withdraw the finger plane before
        # judging stability. Continuing to steer while calling the stage
        # "verification" can apply a new lateral impulse to irregular objects
        # and does not test whether the completed push remains successful.
        verify_ee = last_ee.copy()
        verify_ee[:2] -= push_direction[:2] * 0.025
        verify_ee[2] += 0.140
        if not ended:
            _, _, ended = _run_pose_segment(
                env,
                recorder,
                stage="retreat",
                target_position=verify_ee,
                target_quaternion=finger_quaternion,
                start_hand=planar_hand,
                target_hand=planar_hand,
                steps=transfer.retreat_steps,
                smooth=True,
            )
        if not ended:
            _, _, ended = _run_pose_segment(
                env,
                recorder,
                stage="task_verify",
                target_position=verify_ee,
                target_quaternion=finger_quaternion,
                start_hand=planar_hand,
                target_hand=planar_hand,
                steps=transfer.verify_steps,
                smooth=True,
            )

        arrays = recorder.arrays()
        verify_mask = arrays["stage"] == STAGE_CODES["task_verify"]
        verify = arrays["task_success"][verify_mask]
        active_mask = (
            (arrays["stage"] == STAGE_CODES["approach"])
            | (arrays["stage"] == STAGE_CODES["transport"])
            | verify_mask
        )
        digit_contacts = arrays["robot_object_digit_contact_count"]
        finger_contacts = int(np.count_nonzero(np.sum(digit_contacts[active_mask, :4], axis=1)))
        thumb_contacts = int(np.count_nonzero(digit_contacts[active_mask, 4]))
        tactile = arrays["tactile"]
        finger_tactile = (
            tactile[:, finger_tactile_indices] if tactile.ndim == 2 else np.zeros((len(tactile), 0))
        )
        finger_tactile_max = float(np.max(finger_tactile, initial=0.0))
        finger_contact_mask = np.sum(digit_contacts[:, :4], axis=1) > 0
        tactile_contact_mask = (
            np.max(finger_tactile, axis=1) >= transfer.tactile_threshold
            if finger_tactile.shape[1]
            else np.zeros(len(finger_contact_mask), dtype=bool)
        )
        tactile_contact_frames = int(np.count_nonzero(finger_contact_mask & tactile_contact_mask))
        tactile_valid = tactile_contact_frames >= transfer.required_finger_contact_frames
        hand_table_contacts = _hand_table_contact_frames(env, arrays, active_mask | verify_mask)
        stable_tail = verify[-min(len(verify), 20) :] if len(verify) else verify
        success = bool(
            len(verify)
            and verify[-1]
            and bool(np.all(stable_tail))
            and finger_contacts >= transfer.required_finger_contact_frames
            and thumb_contacts == 0
            and hand_table_contacts == 0
            and tactile_valid
        )
        final = env.task._body_pos(env.model, env.data, "object").copy()
        return DemonstrationEpisode(
            object_id=source.object_id,
            seed=seed,
            candidate=source.candidate,
            arrays=arrays,
            success=success,
            terminal_stage="task_verify",
            failure_reason=None
            if success
            else "object was not stably pushed by the four-finger tactile plane",
            metadata={
                "task": "push",
                "source_skill": "lift_object_pose",
                "pipeline_version": PUSH_TRANSFER_VERSION,
                "task_config": task_config,
                "transfer_config": asdict(transfer),
                "target_position": target.tolist(),
                "object_final_position": final.tolist(),
                "final_goal_distance": float(np.linalg.norm(final[:2] - target[:2])),
                "finger_object_contact_frames": finger_contacts,
                "thumb_object_contact_frames": thumb_contacts,
                "hand_table_contact_frames": hand_table_contacts,
                "finger_tactile_max": finger_tactile_max,
                "finger_tactile_contact_frames": tactile_contact_frames,
                "tactile_metadata": env.tactile_sensor.metadata(),
                "finger_plane_center_in_ee": finger_center_local.tolist(),
                "finger_plane_normal_in_ee": finger_normal_local.tolist(),
                "finger_plane_span_in_ee": finger_span_local.tolist(),
                "planar_hand_target": planar_hand.tolist(),
            },
        )
    finally:
        env.close()
