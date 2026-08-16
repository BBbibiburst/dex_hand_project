"""MJWarp vector environment for residual refinement of UltraDexGrasp trajectories."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mujoco
import mujoco_warp as mjw
import numpy as np
import torch
import warp as wp

from source.envs.manipulation import make_lift_env
from source.rl.residual.reference import (
    STAGE_CODES,
    EpisodeRecord,
    ReferenceTrajectory,
    resolve_reference_manifest,
)
from source.rl.residual.trajectory import ResidualTrajectory


@wp.kernel
def _collect_object_contacts(
    dimensions: wp.array(dtype=wp.int32),
    geoms: wp.array(dtype=wp.vec2i),
    world_ids: wp.array(dtype=wp.int32),
    object_mask: wp.array(dtype=wp.int32),
    robot_mask: wp.array(dtype=wp.int32),
    digit_lookup: wp.array(dtype=wp.int32),
    contact_counts: wp.array(dtype=wp.int32),
    digit_flags: wp.array(dtype=wp.int32),
):
    contact_index = wp.tid()
    if dimensions[contact_index] <= 0:
        return
    geom_pair = geoms[contact_index]
    geom0 = geom_pair[0]
    geom1 = geom_pair[1]
    # MJWarp uses -1 for non-geom/flex contact endpoints.  This project
    # currently scores rigid robot/object geom contacts only.
    if geom0 < 0 or geom1 < 0:
        return
    robot_geom = -1
    if object_mask[geom0] != 0 and robot_mask[geom1] != 0:
        robot_geom = geom1
    elif object_mask[geom1] != 0 and robot_mask[geom0] != 0:
        robot_geom = geom0
    if robot_geom < 0:
        return
    world = world_ids[contact_index]
    if world < 0:
        return
    wp.atomic_add(contact_counts, world, 1)
    digit = digit_lookup[robot_geom]
    if digit >= 0 and digit < 5:
        wp.atomic_max(digit_flags, world * 5 + digit, 1)


@dataclass(frozen=True)
class ResidualLiftConfig:
    num_envs: int = 1024
    device: str = "cuda:0"
    action_mode: str = "hand"
    start_stage: str = "approach"
    hand_residual_fraction: float = 0.12
    arm_residual_radians: float = 0.04
    nconmax: int = 192
    njmax: int = 768
    success_lift_height: float = 0.055
    success_hold_steps: int = 8
    minimum_contact_digits: int = 2
    maximum_object_speed: float = 0.65
    drop_margin: float = 0.025
    # Reward v2: generic contact is only a small shaping term.  The important
    # contact event is thumb + at least one non-thumb digit on the object.
    reward_version: int = 2
    contact_reward: float = 0.10
    thumb_contact_reward: float = 0.20
    opposition_reward: float = 0.60
    lift_reward: float = 3.0
    success_reward: float = 10.0
    drop_penalty: float = 2.0
    action_penalty: float = 0.015
    action_delta_penalty: float = 0.01

    def validate(self) -> None:
        if self.num_envs <= 0 or self.nconmax <= 0 or self.njmax <= 0:
            raise ValueError("MJWarp environment sizes and capacities must be positive.")
        if self.action_mode not in {"hand", "arm_hand"}:
            raise ValueError("action_mode must be 'hand' or 'arm_hand'.")
        if not 0.0 < self.hand_residual_fraction <= 0.5:
            raise ValueError("hand_residual_fraction must lie in (0, 0.5].")
        if self.arm_residual_radians <= 0.0:
            raise ValueError("arm_residual_radians must be positive.")
        if self.success_lift_height <= 0.0 or self.success_hold_steps <= 0:
            raise ValueError("Success height and hold steps must be positive.")
        if not 1 <= self.minimum_contact_digits <= 5:
            raise ValueError("minimum_contact_digits must lie in [1, 5].")


class MjWarpResidualLiftEnv:
    """Synchronous GPU worlds that refine one grasp reference trajectory.

    Worlds intentionally share the same object, initial state, and reference
    time index.  Exploration noise creates different residual trajectories.  A
    fixed synchronized horizon avoids CPU-side per-world reset logic and keeps
    the first implementation focused on high-throughput trajectory search.
    """

    def __init__(
        self,
        reference_manifest: str | Path,
        config: ResidualLiftConfig | None = None,
    ) -> None:
        self.config = config or ResidualLiftConfig()
        self.config.validate()
        self.num_envs = self.config.num_envs

        reference_manifest = resolve_reference_manifest(reference_manifest)

        # Normal production path: an Ultra/lattice DemonstrationEpisode.
        #
        # Additive bridge path: a ResidualTrajectory produced by grasp_edit or
        # primitive_grasp_edit. In that case the edited trajectory already
        # contains the low-level arm+hand controls we want to refine. We reuse
        # the source template only for stage labels, actuator metadata, control
        # limits and the control value immediately before the approach stage.
        residual_reference: ResidualTrajectory | None = None
        base_manifest = reference_manifest
        try:
            self.episode = EpisodeRecord.load(reference_manifest)
        except (OSError, KeyError, TypeError, ValueError):
            residual_reference = ResidualTrajectory.load(reference_manifest)
            if self.config.start_stage != residual_reference.start_stage:
                raise ValueError(
                    "ResidualTrajectory references must currently be trained from "
                    f"their recorded start_stage={residual_reference.start_stage!r}; "
                    f"got --start-stage={self.config.start_stage!r}."
                )

            base_path = residual_reference.metadata.get("template_manifest")
            if not base_path:
                base_path = residual_reference.source_manifest
            base_manifest = resolve_reference_manifest(Path(str(base_path)))
            self.episode = EpisodeRecord.load(base_manifest)

            if self.episode.object_id != residual_reference.object_id:
                raise ValueError(
                    "ResidualTrajectory object does not match its source template: "
                    f"{residual_reference.object_id!r} != {self.episode.object_id!r}."
                )

        if residual_reference is None:
            source_seed = int(self.episode.seed)
            object_id = self.episode.object_id
            reference_control_dt = float(self.episode.metadata.get("control_dt", 0.05))
        else:
            source_seed = int(residual_reference.metadata.get("source_seed", self.episode.seed))
            object_id = residual_reference.object_id
            reference_control_dt = float(
                residual_reference.metadata.get(
                    "control_dt",
                    self.episode.metadata.get("control_dt", 0.05),
                )
            )

        self.host_env = make_lift_env(
            task_config={
                "object_id": object_id,
                "reward_shaping": False,
                "terminate_on_success": False,
            },
            control_mode="position",
            enable_tactile_sensors=False,
            render_mode=None,
            control_dt=reference_control_dt,
        )
        self.host_env.reset(seed=source_seed)

        base_reference = ReferenceTrajectory.from_episode(
            self.episode,
            self.host_env,
            source_manifest=base_manifest,
            start_stage=self.config.start_stage,
        )

        if residual_reference is None:
            self.reference = base_reference
        else:
            controls = np.asarray(residual_reference.controls, dtype=np.float32)
            initial_qpos = np.asarray(residual_reference.initial_qpos, dtype=np.float32)
            initial_qvel = np.asarray(residual_reference.initial_qvel, dtype=np.float32)

            if controls.shape != base_reference.controls.shape:
                raise ValueError(
                    "ResidualTrajectory controls do not match the source template "
                    f"horizon/action shape: residual={controls.shape}, "
                    f"source={base_reference.controls.shape}."
                )
            if initial_qpos.shape != base_reference.initial_qpos.shape:
                raise ValueError(
                    "ResidualTrajectory initial_qpos does not match the current robot model: "
                    f"residual={initial_qpos.shape}, source={base_reference.initial_qpos.shape}."
                )
            if initial_qvel.shape != base_reference.initial_qvel.shape:
                raise ValueError(
                    "ResidualTrajectory initial_qvel does not match the current robot model: "
                    f"residual={initial_qvel.shape}, source={base_reference.initial_qvel.shape}."
                )

            self.reference = ReferenceTrajectory(
                object_id=residual_reference.object_id,
                source_manifest=Path(reference_manifest),
                source_seed=source_seed,
                start_stage=residual_reference.start_stage,
                control_dt=reference_control_dt,
                initial_qpos=initial_qpos.copy(),
                initial_qvel=initial_qvel.copy(),
                initial_ctrl=base_reference.initial_ctrl.copy(),
                controls=controls.copy(),
                stages=base_reference.stages.copy(),
                actuator_ids=base_reference.actuator_ids.copy(),
                actuator_names=base_reference.actuator_names,
                ctrl_low=base_reference.ctrl_low.copy(),
                ctrl_high=base_reference.ctrl_high.copy(),
                arm_action_size=base_reference.arm_action_size,
                initial_object_position=base_reference.initial_object_position.copy(),
            )
        if self.reference.hand_action_size != 6:
            raise ValueError(
                "Residual grasp RL currently expects the six-drive Dex Hand, got "
                f"{self.reference.hand_action_size} hand actions."
            )
        required_stage_codes = {STAGE_CODES["lift"], STAGE_CODES["verify"]}
        present_stage_codes = {int(value) for value in self.reference.stages}
        if not required_stage_codes.issubset(present_stage_codes):
            raise ValueError(
                "RL reference must reach both lift and verify. Choose a full grasp "
                "trajectory rather than one that aborted during approach/closure."
            )
        if abs(self.host_env.config.control_dt - self.reference.control_dt) > 1e-9:
            raise RuntimeError("Reference and training control rates do not match.")

        self.model = self.host_env.model
        self.host_data = self.host_env.data
        self.host_data.qpos[:] = self.reference.initial_qpos
        self.host_data.qvel[:] = self.reference.initial_qvel
        self.host_data.ctrl[:] = self.reference.initial_ctrl
        mujoco.mj_forward(self.model, self.host_data)

        wp.init()
        wp.set_device(self.config.device)
        self.wp_device = wp.get_device()
        if not self.wp_device.is_cuda:
            raise RuntimeError("Residual grasp RL requires a CUDA Warp device.")
        self.torch_device = torch.device(wp.device_to_torch(self.wp_device))
        self.device_model = mjw.put_model(self.model)
        self.data = mjw.put_data(
            self.model,
            self.host_data,
            nworld=self.num_envs,
            nconmax=self.config.nconmax,
            njmax=self.config.njmax,
        )

        self.qpos = wp.to_torch(self.data.qpos, requires_grad=False)
        self.qvel = wp.to_torch(self.data.qvel, requires_grad=False)
        self.ctrl = wp.to_torch(self.data.ctrl, requires_grad=False)
        self.xpos = wp.to_torch(self.data.xpos, requires_grad=False)
        self.xquat = wp.to_torch(self.data.xquat, requires_grad=False)

        self.initial_qpos = torch.as_tensor(
            self.reference.initial_qpos, device=self.torch_device, dtype=torch.float32
        )
        self.initial_qvel = torch.as_tensor(
            self.reference.initial_qvel, device=self.torch_device, dtype=torch.float32
        )
        self.initial_ctrl = torch.as_tensor(
            self.reference.initial_ctrl, device=self.torch_device, dtype=torch.float32
        )
        self.reference_controls = torch.as_tensor(
            self.reference.controls, device=self.torch_device, dtype=torch.float32
        )
        self.reference_stages = torch.as_tensor(
            self.reference.stages, device=self.torch_device, dtype=torch.float32
        )
        self.actuator_ids = torch.as_tensor(
            self.reference.actuator_ids, device=self.torch_device, dtype=torch.long
        )
        self.ctrl_low = torch.as_tensor(
            self.reference.ctrl_low, device=self.torch_device, dtype=torch.float32
        )
        self.ctrl_high = torch.as_tensor(
            self.reference.ctrl_high, device=self.torch_device, dtype=torch.float32
        )

        if self.config.action_mode == "hand":
            controlled = np.arange(
                self.reference.arm_action_size,
                self.reference.action_dim,
                dtype=np.int64,
            )
        else:
            controlled = np.arange(self.reference.action_dim, dtype=np.int64)
        self.controlled_positions = torch.as_tensor(
            controlled, device=self.torch_device, dtype=torch.long
        )
        residual_scale = np.empty(len(controlled), dtype=np.float32)
        for output_index, control_index in enumerate(controlled):
            if control_index < self.reference.arm_action_size:
                residual_scale[output_index] = self.config.arm_residual_radians
            else:
                residual_scale[output_index] = self.config.hand_residual_fraction * (
                    self.reference.ctrl_high[control_index] - self.reference.ctrl_low[control_index]
                )
        self.residual_scale = torch.as_tensor(
            residual_scale, device=self.torch_device, dtype=torch.float32
        )
        self.action_dim = len(controlled)

        bindings = self.host_env.task._require_bindings()
        object_binding = bindings.objects["object"]
        self.object_body_id = int(object_binding.body_id)
        self.object_qvel_adr = int(object_binding.qvel_adr)
        self.initial_object_z = float(self.reference.initial_object_position[2])
        self._prepare_contact_lookup(bindings)

        self.physics_steps_per_control = max(
            1, round(self.reference.control_dt / self.model.opt.timestep)
        )
        with wp.ScopedCapture(device=self.wp_device) as capture:
            mjw.step(self.device_model, self.data)
        self.step_graph = capture.graph

        self.step_index = 0
        self.success_streak = torch.zeros(
            self.num_envs, device=self.torch_device, dtype=torch.int32
        )
        self.success_reached = torch.zeros(
            self.num_envs, device=self.torch_device, dtype=torch.bool
        )
        self.episode_return = torch.zeros(self.num_envs, device=self.torch_device)
        # Diagnostic state for choosing a rollout to visualize. This is
        # intentionally independent of the formal success criterion.
        self.episode_max_lift = torch.full(
            (self.num_envs,), -float("inf"), device=self.torch_device
        )
        self.episode_opposition_steps = torch.zeros(
            self.num_envs, device=self.torch_device, dtype=torch.int32
        )
        self.last_action = torch.zeros((self.num_envs, self.action_dim), device=self.torch_device)
        self.action_history = torch.zeros(
            (self.reference.horizon, self.num_envs, self.action_dim),
            device=self.torch_device,
            dtype=torch.float32,
        )
        self.completed_episodes = 0
        self.last_success_rate = 0.0
        self.last_mean_return = 0.0
        # Update-level diagnostics stay on the GPU during rollout; converting
        # to Python scalars only once per PPO update avoids per-step syncs.
        self._diagnostic_steps = 0
        self._diagnostic_lift_sum = torch.zeros((), device=self.torch_device)
        self._diagnostic_max_lift = torch.full((), -float("inf"), device=self.torch_device)
        self._diagnostic_contact_sum = torch.zeros((), device=self.torch_device)
        self._diagnostic_thumb_sum = torch.zeros((), device=self.torch_device)
        self._diagnostic_opposition_sum = torch.zeros((), device=self.torch_device)
        self._diagnostic_stable_sum = torch.zeros((), device=self.torch_device)
        self._diagnostic_hold_max = torch.zeros((), device=self.torch_device, dtype=torch.int32)
        self._last_diagnostics = {
            "mean_lift": 0.0,
            "max_lift": 0.0,
            "mean_contact_digits": 0.0,
            "thumb_contact_rate": 0.0,
            "opposition_rate": 0.0,
            "stable_rate": 0.0,
            "max_hold_steps": 0.0,
        }
        # Keep the most informative failed rollout as well as the formal
        # successful trajectory. Diagnostic ranking is lift-first, not
        # shaping-reward-first.
        self.best_attempt_lift = -np.inf
        self.best_attempt_return = -np.inf
        self.best_attempt_trajectory: ResidualTrajectory | None = None
        self.best_attempt_version = 0
        self.best_success_return = -np.inf
        self.best_trajectory: ResidualTrajectory | None = None
        self.best_version = 0
        self._reset_all()
        self.obs_dim = int(self._observation().shape[1])

    def _prepare_contact_lookup(self, bindings) -> None:
        """Build object/robot masks and map the full finger branches to 5 digits.

        The previous classifier only recognized ``skin_<digit>_*`` contact geoms. Real
        grasps can contact the object through a side-link collision geom, so
        those contacts disappeared from ``digit_flags`` even though MuJoCo was
        physically supporting the object.  The v4 mapping keeps direct skin-name
        matches but also assigns any robot geom on a digit's kinematic branch.
        """
        object_mask = np.zeros(self.model.ngeom, dtype=np.int32)
        robot_mask = np.zeros(self.model.ngeom, dtype=np.int32)
        digit_lookup = np.full(self.model.ngeom, -1, dtype=np.int32)

        for geom in bindings.objects["object"].geom_ids:
            object_mask[int(geom)] = 1

        def ancestors(body_id: int) -> tuple[int, ...]:
            values: list[int] = []
            current = int(body_id)
            seen: set[int] = set()
            while current >= 0 and current not in seen:
                values.append(current)
                seen.add(current)
                if current == 0:
                    break
                current = int(self.model.body_parentid[current])
            return tuple(values)

        def deepest_common_ancestor(body_ids: list[int]) -> int:
            valid = [int(value) for value in body_ids if int(value) >= 0]
            if not valid:
                return -1
            ancestor_lists = [ancestors(value) for value in valid]
            common = set(ancestor_lists[0])
            for chain in ancestor_lists[1:]:
                common.intersection_update(chain)
            if not common:
                return -1
            # The first common body encountered from the first anchor toward
            # the root is the deepest common ancestor.
            return next((body for body in ancestor_lists[0] if body in common), -1)

        hand_prefix = str(
            getattr(self.host_env.controller.hand_controller, "hand_prefix", "") or ""
        )

        def resolve_hand_geom(local_name: str) -> int:
            candidates = []
            if hand_prefix:
                candidates.append(f"{hand_prefix}{local_name}")
            candidates.append(local_name)
            for candidate in candidates:
                geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, candidate)
                if geom_id >= 0:
                    return int(geom_id)
            matches = [
                geom_id
                for geom_id in range(self.model.ngeom)
                if (
                    mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
                ).endswith(local_name)
            ]
            return int(matches[0]) if len(matches) == 1 else -1

        digit_roots: list[int] = []
        for digit in range(5):
            anchor_bodies: list[int] = []
            for part in range(3):
                geom_id = resolve_hand_geom(f"skin_{digit}_{part}_p")
                if geom_id >= 0:
                    anchor_bodies.append(int(self.model.geom_bodyid[geom_id]))
            digit_roots.append(deepest_common_ancestor(anchor_bodies))

        for geom in bindings.robot_geom_ids:
            geom = int(geom)
            robot_mask[geom] = 1
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom) or ""

            direct = -1
            for digit in range(5):
                if f"skin_{digit}_" in name:
                    direct = digit
                    break
            if direct >= 0:
                digit_lookup[geom] = direct
                continue

            # Palm/wrist/arm geoms deliberately remain unassigned.  For other
            # robot geoms, assign only when exactly one digit branch root is an
            # ancestor, which avoids labeling a shared palm ancestor as a finger.
            body_id = int(self.model.geom_bodyid[geom])
            chain = set(ancestors(body_id))
            candidates = [
                digit for digit, root in enumerate(digit_roots) if root > 0 and root in chain
            ]
            if len(candidates) == 1:
                digit_lookup[geom] = candidates[0]

        self.has_digit_contacts = bool(np.any(digit_lookup >= 0))
        self.object_mask = wp.from_numpy(object_mask, dtype=wp.int32, device=self.wp_device)
        self.robot_mask = wp.from_numpy(robot_mask, dtype=wp.int32, device=self.wp_device)
        self.digit_lookup = wp.from_numpy(digit_lookup, dtype=wp.int32, device=self.wp_device)
        self.contact_counts_wp = wp.zeros(self.num_envs, dtype=wp.int32, device=self.wp_device)
        self.digit_flags_wp = wp.zeros(self.num_envs * 5, dtype=wp.int32, device=self.wp_device)
        self.contact_counts = wp.to_torch(self.contact_counts_wp, requires_grad=False)
        self.digit_flags = wp.to_torch(self.digit_flags_wp, requires_grad=False).view(
            self.num_envs, 5
        )

    def _sync_torch_before_warp(self) -> None:
        torch.cuda.synchronize(self.torch_device)

    def _sync_warp_before_torch(self) -> None:
        wp.synchronize_device(self.wp_device)

    def _reset_all(self) -> None:
        self.qpos[:] = self.initial_qpos.unsqueeze(0)
        self.qvel[:] = self.initial_qvel.unsqueeze(0)
        self.ctrl[:] = self.initial_ctrl.unsqueeze(0)
        self.success_streak.zero_()
        self.success_reached.zero_()
        self.episode_return.zero_()
        self.episode_max_lift.fill_(-float("inf"))
        self.episode_opposition_steps.zero_()
        self.last_action.zero_()
        self.action_history.zero_()
        self.step_index = 0
        self._sync_torch_before_warp()
        mjw.forward(self.device_model, self.data)
        self._update_contacts()

    def _update_contacts(self) -> None:
        self.contact_counts_wp.zero_()
        self.digit_flags_wp.zero_()
        wp.launch(
            _collect_object_contacts,
            dim=int(self.data.contact.dim.shape[0]),
            inputs=[
                self.data.contact.dim,
                self.data.contact.geom,
                self.data.contact.worldid,
                self.object_mask,
                self.robot_mask,
                self.digit_lookup,
                self.contact_counts_wp,
                self.digit_flags_wp,
            ],
            device=self.wp_device,
        )
        self._sync_warp_before_torch()

    def _contact_signal(self) -> torch.Tensor:
        if self.has_digit_contacts:
            return self.digit_flags.sum(dim=1).float()
        return torch.clamp(self.contact_counts.float(), max=5.0)

    def _observation(self) -> torch.Tensor:
        object_position = self.xpos[:, self.object_body_id]
        object_delta = object_position - torch.as_tensor(
            self.reference.initial_object_position,
            device=self.torch_device,
            dtype=torch.float32,
        ).unsqueeze(0)
        object_quaternion = self.xquat[:, self.object_body_id]
        progress = torch.full(
            (self.num_envs, 1),
            float(self.step_index) / max(self.reference.horizon - 1, 1),
            device=self.torch_device,
        )
        stage_index = min(self.step_index, self.reference.horizon - 1)
        stage = torch.full(
            (self.num_envs, 1),
            float(self.reference_stages[stage_index].item()) / 7.0,
            device=self.torch_device,
        )
        contact_count = torch.clamp(self.contact_counts.float().unsqueeze(1), max=8.0) / 8.0
        return torch.cat(
            [
                self.qpos,
                self.qvel,
                object_delta,
                object_quaternion,
                self.digit_flags.float(),
                contact_count,
                progress,
                stage,
                self.last_action,
            ],
            dim=1,
        ).float()

    def reset(self) -> torch.Tensor:
        self._reset_all()
        return self._observation()

    def _apply_residual(self, actions: torch.Tensor) -> torch.Tensor:
        actions = torch.clamp(actions, -1.0, 1.0)
        reference = self.reference_controls[self.step_index].unsqueeze(0).expand(self.num_envs, -1)
        target = reference.clone()
        target[:, self.controlled_positions] += actions * self.residual_scale.unsqueeze(0)
        target = torch.maximum(torch.minimum(target, self.ctrl_high), self.ctrl_low)
        self.ctrl[:, self.actuator_ids] = target
        return target

    def _reward(self, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        object_position = self.xpos[:, self.object_body_id]
        lift = object_position[:, 2] - self.initial_object_z
        lift_progress = torch.clamp(lift / self.config.success_lift_height, 0.0, 1.0)
        contact_signal = self._contact_signal()
        contact_progress = torch.clamp(
            contact_signal / float(self.config.minimum_contact_digits), 0.0, 1.0
        )

        if self.has_digit_contacts:
            # Dex Hand contact groups 0..3 are fingers and group 4 is thumb.
            thumb_contact = self.digit_flags[:, 4] > 0
            non_thumb_contact = self.digit_flags[:, :4].sum(dim=1) > 0
        else:
            # Conservative fallback for models without named skin groups.
            thumb_contact = contact_signal >= 1.0
            non_thumb_contact = contact_signal >= 2.0
        opposition = thumb_contact & non_thumb_contact
        # Track physical lift over the entire horizon. This is later used to
        # choose a diagnostic rollout even if every policy sample formally fails.
        self.episode_max_lift = torch.maximum(self.episode_max_lift, lift)
        self.episode_opposition_steps += opposition.to(torch.int32)

        object_velocity = self.qvel[:, self.object_qvel_adr : self.object_qvel_adr + 6]
        object_speed = torch.linalg.vector_norm(object_velocity, dim=1)
        stable = (
            (lift >= self.config.success_lift_height)
            & (contact_signal >= self.config.minimum_contact_digits)
            & opposition
            & (object_speed <= self.config.maximum_object_speed)
        )
        self.success_streak = torch.where(
            stable,
            self.success_streak + 1,
            torch.zeros_like(self.success_streak),
        )
        success_now = self.success_streak >= self.config.success_hold_steps
        new_success = success_now & ~self.success_reached
        self.success_reached |= success_now
        dropped = lift < -self.config.drop_margin

        # Do not let one-sided contact dominate the objective.  Lift shaping is
        # strongest when a thumb/finger opposition pair is actually present,
        # while retaining a small gradient before opposition is discovered.
        lift_quality = 0.25 + 0.75 * opposition.float()
        action_cost = actions.square().mean(dim=1)
        delta_cost = (actions - self.last_action).square().mean(dim=1)
        reward = (
            self.config.contact_reward * contact_progress
            + self.config.thumb_contact_reward * thumb_contact.float()
            + self.config.opposition_reward * opposition.float()
            + self.config.lift_reward * lift_progress * lift_quality
            + self.config.success_reward * new_success.float()
            - self.config.drop_penalty * dropped.float()
            - self.config.action_penalty * action_cost
            - self.config.action_delta_penalty * delta_cost
        )

        self._diagnostic_steps += 1
        self._diagnostic_lift_sum += lift.mean()
        self._diagnostic_max_lift = torch.maximum(self._diagnostic_max_lift, lift.max())
        self._diagnostic_contact_sum += contact_signal.mean()
        self._diagnostic_thumb_sum += thumb_contact.float().mean()
        self._diagnostic_opposition_sum += opposition.float().mean()
        self._diagnostic_stable_sum += stable.float().mean()
        self._diagnostic_hold_max = torch.maximum(
            self._diagnostic_hold_max, self.success_streak.max()
        )
        return reward, new_success

    def _finalize_episode(self) -> None:
        # A demonstration is useful only if the object is still stably held at
        # the end of the reference horizon.  ``success_reached`` is deliberately
        # sticky for the one-time reward bonus, but must not qualify a trajectory
        # that subsequently drops the object.
        success = self.success_streak >= self.config.success_hold_steps
        returns = self.episode_return
        self.completed_episodes += self.num_envs
        self.last_success_rate = float(success.float().mean().item())
        self.last_mean_return = float(returns.mean().item())

        # Diagnostic rollout: choose the world with the largest physical object
        # lift. Episode return is only a tie-breaker because shaping reward can
        # otherwise dominate selection.
        max_lift_value = torch.max(self.episode_max_lift)
        lift_candidates = torch.nonzero(
            self.episode_max_lift >= max_lift_value - 1e-6,
            as_tuple=False,
        ).flatten()
        candidate_returns = returns[lift_candidates]
        attempt_local = int(torch.argmax(candidate_returns).item())
        attempt_world = int(lift_candidates[attempt_local].item())
        attempt_lift = float(self.episode_max_lift[attempt_world].item())
        attempt_return = float(returns[attempt_world].item())
        attempt_opp_steps = int(self.episode_opposition_steps[attempt_world].item())
        if attempt_lift > self.best_attempt_lift + 1e-6 or (
            abs(attempt_lift - self.best_attempt_lift) <= 1e-6
            and attempt_return > self.best_attempt_return
        ):
            residual = self.action_history[:, attempt_world].detach().cpu().numpy()
            controls = self.reference.controls.copy()
            positions = self.controlled_positions.detach().cpu().numpy()
            physical_delta = residual * self.residual_scale.detach().cpu().numpy()[None, :]
            controls[:, positions] += physical_delta
            controls = np.clip(
                controls,
                self.reference.ctrl_low[None, :],
                self.reference.ctrl_high[None, :],
            )
            self.best_attempt_lift = attempt_lift
            self.best_attempt_return = attempt_return
            self.best_attempt_trajectory = ResidualTrajectory(
                object_id=self.reference.object_id,
                source_manifest=str(self.reference.source_manifest),
                start_stage=self.reference.start_stage,
                action_mode=self.config.action_mode,
                residual_actions=residual,
                controls=controls,
                initial_qpos=self.reference.initial_qpos.copy(),
                initial_qvel=self.reference.initial_qvel.copy(),
                success=bool(success[attempt_world].item()),
                episode_return=attempt_return,
                metadata={
                    "actuator_names": list(self.reference.actuator_names),
                    "controlled_positions": positions.tolist(),
                    "residual_scale": self.residual_scale.detach().cpu().numpy().tolist(),
                    "success_lift_height": self.config.success_lift_height,
                    "success_hold_steps": self.config.success_hold_steps,
                    "minimum_contact_digits": self.config.minimum_contact_digits,
                    "require_thumb_opposition": True,
                    "reward_version": getattr(self.config, "reward_version", 1),
                    "control_dt": self.reference.control_dt,
                    "source_seed": self.reference.source_seed,
                    "mjwarp_max_lift": attempt_lift,
                    "mjwarp_opposition_steps": attempt_opp_steps,
                    "diagnostic_attempt": True,
                },
            )
            self.best_attempt_version += 1

        if success.any():
            successful_indices = torch.nonzero(success, as_tuple=False).flatten()
            successful_returns = returns[successful_indices]
            local = int(torch.argmax(successful_returns).item())
            world = int(successful_indices[local].item())
            score = float(returns[world].item())
            if score > self.best_success_return:
                residual = self.action_history[:, world].detach().cpu().numpy()
                controls = self.reference.controls.copy()
                positions = self.controlled_positions.detach().cpu().numpy()
                physical_delta = residual * self.residual_scale.detach().cpu().numpy()[None, :]
                controls[:, positions] += physical_delta
                controls = np.clip(
                    controls,
                    self.reference.ctrl_low[None, :],
                    self.reference.ctrl_high[None, :],
                )
                self.best_success_return = score
                self.best_trajectory = ResidualTrajectory(
                    object_id=self.reference.object_id,
                    source_manifest=str(self.reference.source_manifest),
                    start_stage=self.reference.start_stage,
                    action_mode=self.config.action_mode,
                    residual_actions=residual,
                    controls=controls,
                    initial_qpos=self.reference.initial_qpos.copy(),
                    initial_qvel=self.reference.initial_qvel.copy(),
                    success=True,
                    episode_return=score,
                    metadata={
                        "actuator_names": list(self.reference.actuator_names),
                        "controlled_positions": positions.tolist(),
                        "residual_scale": self.residual_scale.detach().cpu().numpy().tolist(),
                        "success_lift_height": self.config.success_lift_height,
                        "success_hold_steps": self.config.success_hold_steps,
                        "minimum_contact_digits": self.config.minimum_contact_digits,
                        "require_thumb_opposition": True,
                        "reward_version": self.config.reward_version,
                        "control_dt": self.reference.control_dt,
                        "source_seed": self.reference.source_seed,
                    },
                )
                self.best_version += 1

    def step(
        self,
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        if actions.shape != (self.num_envs, self.action_dim):
            raise ValueError(
                f"Residual actions must have shape {(self.num_envs, self.action_dim)}, "
                f"got {tuple(actions.shape)}."
            )
        if actions.device != self.torch_device:
            actions = actions.to(self.torch_device)
        actions = torch.clamp(actions.float(), -1.0, 1.0)
        self.action_history[self.step_index] = actions
        self._apply_residual(actions)
        self._sync_torch_before_warp()
        for _ in range(self.physics_steps_per_control):
            wp.capture_launch(self.step_graph)
        self._update_contacts()

        reward, new_success = self._reward(actions)
        self.episode_return += reward
        self.last_action.copy_(actions)
        self.step_index += 1
        episode_end = self.step_index >= self.reference.horizon
        if episode_end:
            done = torch.ones(self.num_envs, device=self.torch_device, dtype=torch.bool)
            success_rate = float(self.success_reached.float().mean().item())
            mean_return = float(self.episode_return.mean().item())
            self._finalize_episode()
            self._reset_all()
            info = {
                "episode_end": True,
                "success_rate": success_rate,
                "mean_return": mean_return,
                "new_successes": int(new_success.sum().item()),
                "best_version": self.best_version,
            }
        else:
            done = torch.zeros(self.num_envs, device=self.torch_device, dtype=torch.bool)
            info = {
                "episode_end": False,
                "new_successes": int(new_success.sum().item()),
                "best_version": self.best_version,
            }
        return self._observation(), reward, done, info

    def training_metrics(self) -> dict[str, float]:
        if self._diagnostic_steps > 0:
            scale = 1.0 / float(self._diagnostic_steps)
            self._last_diagnostics = {
                "mean_lift": float((self._diagnostic_lift_sum * scale).item()),
                "max_lift": float(self._diagnostic_max_lift.item()),
                "mean_contact_digits": float((self._diagnostic_contact_sum * scale).item()),
                "thumb_contact_rate": float((self._diagnostic_thumb_sum * scale).item()),
                "opposition_rate": float((self._diagnostic_opposition_sum * scale).item()),
                "stable_rate": float((self._diagnostic_stable_sum * scale).item()),
                "max_hold_steps": float(self._diagnostic_hold_max.item()),
            }
            self._diagnostic_steps = 0
            self._diagnostic_lift_sum.zero_()
            self._diagnostic_max_lift.fill_(-float("inf"))
            self._diagnostic_contact_sum.zero_()
            self._diagnostic_thumb_sum.zero_()
            self._diagnostic_opposition_sum.zero_()
            self._diagnostic_stable_sum.zero_()
            self._diagnostic_hold_max.zero_()
        return {
            "episode_success_rate": self.last_success_rate,
            "episode_mean_return": self.last_mean_return,
            "best_success_return": (
                float(self.best_success_return) if np.isfinite(self.best_success_return) else 0.0
            ),
            "best_attempt_lift": (
                float(self.best_attempt_lift) if np.isfinite(self.best_attempt_lift) else 0.0
            ),
            "best_attempt_return": (
                float(self.best_attempt_return) if np.isfinite(self.best_attempt_return) else 0.0
            ),
            "completed_episodes": float(self.completed_episodes),
            **self._last_diagnostics,
        }

    def close(self) -> None:
        self.host_env.close()
