"""Batched MJWarp direct-hold evaluation for one object's evolution population."""

from __future__ import annotations

from pathlib import Path

import mujoco
import mujoco_warp as mjw
import numpy as np
import warp as wp

from source.grasping.constants import DEFAULT_GRIP_PRELOAD
from source.grasping.standalone_validator import (
    DirectHoldValidationResult,
    build_cached_direct_hold_model,
    resolve_payload_mesh_path,
    set_hand_targets,
    set_object_pose_for_hand_pose,
)
from source.robots.registry import get_hand


@wp.kernel
def _pin_free_object(
    qpos: wp.array2d(dtype=wp.float32),
    qvel: wp.array2d(dtype=wp.float32),
    fixed_pose: wp.array2d(dtype=wp.float32),
    qpos_address: int,
    dof_address: int,
):
    world = wp.tid()
    for index in range(7):
        qpos[world, qpos_address + index] = fixed_pose[world, index]
    for index in range(6):
        qvel[world, dof_address + index] = 0.0


def _contact_counts(data, object_geom: int) -> np.ndarray:
    """Copy only final contact metadata and count object contacts by world."""

    counts = np.zeros(data.nworld, dtype=np.int32)
    nacon = int(data.nacon.numpy()[0])
    if nacon <= 0:
        return counts
    dimensions = data.contact.dim.numpy()[:nacon]
    if not np.any(dimensions > 0):
        return counts
    geoms = data.contact.geom.numpy()[:nacon]
    worlds = data.contact.worldid.numpy()[:nacon]
    active = dimensions > 0
    touching = active & ((geoms[:, 0] == object_geom) | (geoms[:, 1] == object_geom))
    np.add.at(counts, worlds[touching], 1)
    return counts


class MjWarpPopulationEvaluator:
    """Keep one device model alive and evaluate homogeneous candidate batches."""

    def __init__(
        self,
        seed_payload: dict,
        *,
        device: str = "cuda:0",
        nconmax: int = 128,
        njmax: int = 512,
        cache_dir: str | Path = "/tmp/dexhand_mjwarp_cache",
    ) -> None:
        if nconmax <= 0 or njmax <= 0:
            raise ValueError("MJWarp contact and constraint capacities must be positive.")
        wp.config.kernel_cache_dir = str(Path(cache_dir).expanduser())
        wp.init()
        wp.set_device(device)
        if not wp.get_device().is_cuda:
            raise RuntimeError("Evolution acceleration requires a CUDA Warp device.")

        self.nconmax = int(nconmax)
        self.njmax = int(njmax)
        self.end_effector_name = seed_payload.get("end_effector_name", "dex_hand")
        self.actuator_names = tuple(get_hand(self.end_effector_name).position_actuator_names)
        self.model, _ = build_cached_direct_hold_model(
            object_mesh=resolve_payload_mesh_path(seed_payload["mesh"]),
            mesh_center=np.asarray(seed_payload["mesh_center"], dtype=np.float64),
            mesh_scale=float(seed_payload["mesh_scale"]),
            hand_translation=np.asarray(seed_payload["hand_translation"], dtype=np.float64),
            hand_rotation_matrix=np.asarray(seed_payload["hand_rotation_matrix"], dtype=np.float64),
            object_table_height=seed_payload.get("object_table_height"),
            end_effector_name=self.end_effector_name,
        )
        self.device_model = mjw.put_model(self.model)
        joint_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_JOINT,
            "validation_object_freejoint",
        )
        self.qpos_address = int(self.model.jnt_qposadr[joint_id])
        self.dof_address = int(self.model.jnt_dofadr[joint_id])
        self.body_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_BODY,
            "validation_object_body",
        )
        self.object_geom = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_GEOM,
            "validation_object_collision",
        )

    def _initial_states(self, payloads: list[dict]):
        qpos, qvel, ctrl, fixed_pose = [], [], [], []
        first_data = None
        for payload in payloads:
            data = mujoco.MjData(self.model)
            set_object_pose_for_hand_pose(
                self.model,
                data,
                np.asarray(payload["hand_translation"], dtype=np.float64),
                np.asarray(payload["hand_rotation_matrix"], dtype=np.float64),
            )
            set_hand_targets(
                self.model,
                data,
                np.asarray(payload["hand_actuator_values"], dtype=np.float64),
                grip_preload=DEFAULT_GRIP_PRELOAD,
                preload_weights=np.asarray(payload["hand_preload_weights"], dtype=np.float64),
                preload_directions=np.asarray(
                    payload.get(
                        "hand_preload_directions",
                        np.ones(len(self.actuator_names)),
                    ),
                    dtype=np.float64,
                ),
                actuator_names=self.actuator_names,
            )
            mujoco.mj_forward(self.model, data)
            first_data = data if first_data is None else first_data
            qpos.append(data.qpos.copy())
            qvel.append(data.qvel.copy())
            ctrl.append(data.ctrl.copy())
            fixed_pose.append(data.qpos[self.qpos_address : self.qpos_address + 7].copy())
        return (
            first_data,
            np.asarray(qpos, dtype=np.float32),
            np.asarray(qvel, dtype=np.float32),
            np.asarray(ctrl, dtype=np.float32),
            np.asarray(fixed_pose, dtype=np.float32),
        )

    def evaluate(
        self,
        payloads: list[dict],
        *,
        seconds: float,
        settle_seconds: float,
    ) -> list[DirectHoldValidationResult]:
        if not payloads:
            return []
        first_data, qpos, qvel, ctrl, fixed_pose = self._initial_states(payloads)
        data = mjw.put_data(
            self.model,
            first_data,
            nworld=len(payloads),
            nconmax=self.nconmax,
            njmax=self.njmax,
        )
        data.qpos.assign(qpos)
        data.qvel.assign(qvel)
        data.ctrl.assign(ctrl)
        fixed_pose_device = wp.from_numpy(
            fixed_pose,
            dtype=wp.float32,
            device=wp.get_device(),
        )

        settle_steps = int(np.ceil(settle_seconds / self.model.opt.timestep))
        settle_graph = None
        if settle_steps:
            with wp.ScopedCapture(device=wp.get_device()) as capture:
                mjw.step(self.device_model, data)
                wp.launch(
                    _pin_free_object,
                    dim=len(payloads),
                    inputs=[
                        data.qpos,
                        data.qvel,
                        fixed_pose_device,
                        self.qpos_address,
                        self.dof_address,
                    ],
                )
                mjw.forward(self.device_model, data)
            settle_graph = capture.graph
        for _ in range(settle_steps):
            wp.capture_launch(settle_graph)

        wp.synchronize()
        initial_position = data.xpos.numpy()[:, self.body_id].copy()
        initial_quaternion = data.xquat.numpy()[:, self.body_id].copy()
        initial_contacts = _contact_counts(data, self.object_geom)

        steps = int(np.ceil(seconds / self.model.opt.timestep))
        seating_step = min(steps - 1, int(np.ceil(1.0 / self.model.opt.timestep)))
        seated_position = initial_position.copy()
        with wp.ScopedCapture(device=wp.get_device()) as capture:
            mjw.step(self.device_model, data)
        hold_graph = capture.graph
        for step in range(steps):
            wp.capture_launch(hold_graph)
            if step == seating_step:
                wp.synchronize()
                seated_position = data.xpos.numpy()[:, self.body_id].copy()

        wp.synchronize()
        overflow = data.overflow.numpy()
        if np.any(overflow):
            worlds = np.flatnonzero(overflow)
            raise RuntimeError(
                "MJWarp capacity overflow in world(s) "
                f"{worlds[:8].tolist()}; increase mjwarp_nconmax/mjwarp_njmax."
            )
        final_position = data.xpos.numpy()[:, self.body_id].copy()
        final_quaternion = data.xquat.numpy()[:, self.body_id].copy()
        final_contacts = _contact_counts(data, self.object_geom)
        finite = np.all(np.isfinite(data.qpos.numpy()), axis=1) & np.all(
            np.isfinite(data.qvel.numpy()), axis=1
        )

        initial_displacement = np.linalg.norm(final_position - initial_position, axis=1)
        position_drift = np.linalg.norm(final_position - seated_position, axis=1)
        quaternion_dot = np.abs(np.sum(initial_quaternion * final_quaternion, axis=1))
        rotation_drift = 2.0 * np.arccos(np.clip(quaternion_dot, 0.0, 1.0))
        vertical_drop = initial_position[:, 2] - final_position[:, 2]
        stable = (
            finite
            & (position_drift <= 0.01)
            & (rotation_drift <= 0.35)
            & (vertical_drop <= 0.015)
            & (final_contacts >= 2)
        )
        return [
            DirectHoldValidationResult(
                direct_hold_stable=bool(stable[index]),
                initial_displacement=float(initial_displacement[index]),
                position_drift=float(position_drift[index]),
                rotation_drift=float(rotation_drift[index]),
                vertical_drop=float(vertical_drop[index]),
                initial_contacts=int(initial_contacts[index]),
                final_contacts=int(final_contacts[index]),
                simulated_seconds=float(steps * self.model.opt.timestep),
            )
            for index in range(len(payloads))
        ]
