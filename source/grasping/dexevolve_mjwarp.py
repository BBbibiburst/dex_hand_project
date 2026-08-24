"""Batched MJWarp disturbance evaluation for DexEvolve closed-state snapshots."""

from __future__ import annotations

from dataclasses import dataclass
import time

import mujoco_warp as mjw
import numpy as np
import torch
import warp as wp

from source.envs.manipulation import make_lift_env
from source.grasping.contracts import DemonstrationEpisode
from source.grasping.executor import STAGE_CODES, _contact_digit


@dataclass(frozen=True)
class MjWarpLifetimeConfig:
    device: str = "cuda:0"
    disturbance_steps: int = 80
    lateral_force: float = 3.0
    upward_force_ratio: float = 2.0
    maximum_drift: float = 0.035
    require_opposed_contact: bool = True
    nconmax: int = 192
    njmax: int = 768


class MjWarpLifetimeEvaluator:
    """Evaluate one closed-state snapshot per MJWarp world in parallel."""

    def __init__(self, object_id: str, maximum_worlds: int, config=None):
        self.config = config or MjWarpLifetimeConfig()
        self.maximum_worlds = int(maximum_worlds)
        if self.maximum_worlds <= 0:
            raise ValueError("maximum_worlds must be positive.")
        self.host_env = make_lift_env(
            task_config={"object_id": object_id, "terminate_on_success": False},
            control_mode="ik",
            enable_tactile_sensors=False,
            render_mode=None,
            episode_length=400,
        )
        self.host_env.reset(seed=0)
        if hasattr(wp, "config") and hasattr(wp.config, "quiet"):
            wp.config.quiet = True
        wp.init()
        wp.set_device(self.config.device)
        self.wp_device = wp.get_device()
        self.torch_device = torch.device(wp.device_to_torch(self.wp_device))
        self.device_model = mjw.put_model(self.host_env.model)
        self.data = mjw.put_data(
            self.host_env.model,
            self.host_env.data,
            nworld=self.maximum_worlds,
            nconmax=self.config.nconmax,
            njmax=self.config.njmax,
        )
        self.qpos = wp.to_torch(self.data.qpos, requires_grad=False)
        self.qvel = wp.to_torch(self.data.qvel, requires_grad=False)
        self.ctrl = wp.to_torch(self.data.ctrl, requires_grad=False)
        self.xpos = wp.to_torch(self.data.xpos, requires_grad=False)
        self.xfrc = wp.to_torch(self.data.xfrc_applied, requires_grad=False)
        self.contact_geom = wp.to_torch(self.data.contact.geom, requires_grad=False)
        self.contact_world = wp.to_torch(self.data.contact.worldid, requires_grad=False)
        self.active_contact_count = wp.to_torch(self.data.nacon, requires_grad=False)
        binding = self.host_env.task._require_bindings().objects["object"]
        self.object_body_id = int(binding.body_id)
        self.object_mass = float(self.host_env.model.body_mass[self.object_body_id])
        object_geoms = set(int(value) for value in binding.geom_ids)
        geom_roles = np.full(self.host_env.model.ngeom, -2, dtype=np.int64)
        for geom_id in object_geoms:
            geom_roles[geom_id] = -1
        for geom_id in self.host_env.task._require_bindings().robot_geom_ids:
            digit = _contact_digit(self.host_env, int(geom_id))
            if digit >= 0:
                geom_roles[int(geom_id)] = digit
        self.geom_roles = torch.as_tensor(
            geom_roles, device=self.torch_device, dtype=torch.int64
        )
        with wp.ScopedCapture(device=self.wp_device) as capture:
            mjw.step(self.device_model, self.data)
        self.step_graph = capture.graph
        self.completed_worlds = 0
        self.total_evaluation_seconds = 0.0

    def _opposed_contact(self, count: int) -> torch.Tensor:
        """Return worlds having thumb-object and opposing-finger-object contact."""
        active = int(self.active_contact_count.cpu().item())
        opposed = torch.zeros(count, device=self.torch_device, dtype=torch.bool)
        if active <= 0:
            return opposed
        pairs = self.contact_geom[:active].long()
        worlds = self.contact_world[:active].long()
        first = self.geom_roles[pairs[:, 0]]
        second = self.geom_roles[pairs[:, 1]]
        first_is_object = first == -1
        second_is_object = second == -1
        digits = torch.where(first_is_object, second, first)
        relevant = (
            (worlds >= 0)
            & (worlds < count)
            & ((first_is_object & (second >= 0)) | (second_is_object & (first >= 0)))
        )
        if not bool(relevant.any()):
            return opposed
        presence = torch.zeros((count, 5), device=self.torch_device, dtype=torch.bool)
        presence[worlds[relevant], digits[relevant]] = True
        return presence[:, 4] & presence[:, :4].any(dim=1)

    def _sync_to_warp(self):
        if self.torch_device.type == "cuda":
            torch.cuda.synchronize(self.torch_device)

    def _sync_to_torch(self):
        wp.synchronize_device(self.wp_device)

    @staticmethod
    def _closed_frame(episode: DemonstrationEpisode) -> int:
        stages = np.asarray(episode.arrays["stage"])
        hold = np.flatnonzero(stages == STAGE_CODES["hold"])
        if not len(hold):
            raise ValueError("Episode has no closed hold-state snapshot.")
        return int(hold[-1])

    def evaluate(self, episodes: list[DemonstrationEpisode]) -> np.ndarray:
        started = time.perf_counter()
        count = len(episodes)
        if not 0 < count <= self.maximum_worlds:
            raise ValueError(f"Expected 1..{self.maximum_worlds} episodes, got {count}.")
        frames = [self._closed_frame(item) for item in episodes]
        qpos = np.stack([item.arrays["qpos"][frame] for item, frame in zip(episodes, frames)])
        ctrl = np.stack([item.arrays["ctrl"][frame] for item, frame in zip(episodes, frames)])
        # Inactive padding worlds duplicate world zero and are ignored.
        qpos = np.concatenate([qpos, np.repeat(qpos[:1], self.maximum_worlds - count, axis=0)])
        ctrl = np.concatenate([ctrl, np.repeat(ctrl[:1], self.maximum_worlds - count, axis=0)])
        self.qpos.copy_(torch.as_tensor(qpos, device=self.torch_device, dtype=torch.float32))
        self.qvel.zero_()
        self.ctrl.copy_(torch.as_tensor(ctrl, device=self.torch_device, dtype=torch.float32))
        self.xfrc.zero_()
        self._sync_to_warp()
        mjw.forward(self.device_model, self.data)
        self._sync_to_torch()
        initial = self.xpos[:count, self.object_body_id].clone()
        alive = torch.ones(count, device=self.torch_device, dtype=torch.bool)
        lifetime = torch.zeros(count, device=self.torch_device)
        up = self.config.upward_force_ratio * self.object_mass * 9.81
        forces = (
            (0, 0, up),
            (self.config.lateral_force, 0, up),
            (-self.config.lateral_force, 0, up),
            (0, self.config.lateral_force, up),
            (0, -self.config.lateral_force, up),
        )
        for force in forces:
            self.xfrc.zero_()
            self.xfrc[:count, self.object_body_id, :3] = torch.tensor(
                force, device=self.torch_device, dtype=torch.float32
            )
            self._sync_to_warp()
            for _ in range(self.config.disturbance_steps):
                wp.capture_launch(self.step_graph)
            self._sync_to_torch()
            drift = torch.linalg.vector_norm(
                self.xpos[:count, self.object_body_id] - initial, dim=1
            )
            alive &= torch.isfinite(drift) & (drift <= self.config.maximum_drift)
            if self.config.require_opposed_contact:
                alive &= self._opposed_contact(count)
            lifetime += alive.float()
        self.xfrc.zero_()
        result = (lifetime / len(forces)).cpu().numpy()
        self.completed_worlds += count
        self.total_evaluation_seconds += time.perf_counter() - started
        return result

    def metrics(self) -> dict[str, float | str]:
        return {
            "backend": "mujoco-warp",
            "device": str(self.wp_device),
            "maximum_worlds": float(self.maximum_worlds),
            "completed_worlds": float(self.completed_worlds),
            "evaluation_seconds": self.total_evaluation_seconds,
            "worlds_per_second": self.completed_worlds / max(self.total_evaluation_seconds, 1e-9),
        }

    def close(self):
        self.host_env.close()
