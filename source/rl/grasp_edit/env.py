"""Single-step hybrid wrist-template + 6D hand grasp editor backed by MuJoCo Warp."""

from __future__ import annotations

from dataclasses import dataclass

import mujoco_warp as mjw
import numpy as np
import torch
import warp as wp

from source.envs.manipulation import make_lift_env
from source.rl.grasp_edit.templates import GraspEditTemplate
from source.grasp_pipeline.reference import STAGE_CODES, ReferenceTrajectory, load_reference
from source.grasp_pipeline.trajectory import GraspTrajectory
from source.ultradexgrasp.contracts import DemonstrationEpisode
from source.ultradexgrasp.hand_surrogate import OPEN_FRACTIONS


@dataclass(frozen=True)
class GraspEditConfig:
    num_envs: int = 256
    device: str = "cuda:0"
    wrist_translation_scale: float = 0.02
    wrist_rotation_scale_degrees: float = 45.0
    hand_edit_fraction: float = 0.35
    success_lift_height: float = 0.055
    success_tail_steps: int = 8
    maximum_object_speed: float = 0.10
    maximum_object_angular_speed: float = 0.10
    nconmax: int = 192
    njmax: int = 768

    def validate(self) -> None:
        if self.num_envs <= 0:
            raise ValueError("num_envs must be positive.")
        if self.wrist_translation_scale <= 0.0:
            raise ValueError("wrist_translation_scale must be positive.")
        if self.wrist_rotation_scale_degrees <= 0.0:
            raise ValueError("wrist_rotation_scale_degrees must be positive.")
        if not 0.0 < self.hand_edit_fraction <= 1.0:
            raise ValueError("hand_edit_fraction must lie in (0, 1].")
        if self.success_lift_height <= 0.0 or self.success_tail_steps <= 0:
            raise ValueError("success thresholds must be positive.")
        if self.maximum_object_speed <= 0.0:
            raise ValueError("maximum_object_speed must be positive.")
        if self.maximum_object_angular_speed <= 0.0:
            raise ValueError("maximum_object_angular_speed must be positive.")
        if self.nconmax <= 0 or self.njmax <= 0:
            raise ValueError("MJWarp capacities must be positive.")


class MjWarpGraspEditEnv:
    """One PPO step selects a reachable wrist template and edits the hand.

    The hybrid policy samples the wrist template from a categorical
    distribution and the six physical Dex Hand edits from a squashed Gaussian.
    The dense action transport has seven columns: template_id + hand_edit[6].
    There is no continuous-wrist nearest-neighbour projection.
    """

    def __init__(
        self,
        object_id: str,
        templates: tuple[GraspEditTemplate, ...],
        config: GraspEditConfig | None = None,
    ) -> None:
        if not templates:
            raise ValueError("Grasp editing requires at least one template.")
        self.object_id = object_id
        self.templates = templates
        self.config = config or GraspEditConfig()
        self.config.validate()
        self.num_envs = self.config.num_envs

        first_episode = DemonstrationEpisode.load(templates[0].manifest)
        control_dt = float(first_episode.metadata.get("control_dt", 0.05))
        self.host_env = make_lift_env(
            task_config={
                "object_id": object_id,
                "reward_shaping": False,
                "terminate_on_success": False,
            },
            control_mode="position",
            enable_tactile_sensors=False,
            render_mode=None,
            control_dt=control_dt,
            episode_length=400,
        )
        self.model = self.host_env.model
        self.host_data = self.host_env.data
        self.host_env.reset(seed=int(first_episode.seed))

        references = tuple(
            load_reference(item.manifest, self.host_env, start_stage="approach")
            for item in templates
        )
        self._validate_references(references)
        self.references = references
        reference = references[0]
        self.horizon = reference.horizon
        self.template_count = len(references)
        self.hand_action_dim = 6
        self.action_dim = 1 + self.hand_action_dim

        # Warp's per-kernel load messages dominate the terminal but are not
        # training diagnostics.  Silence them when the installed Warp version
        # exposes the standard quiet switch.
        if hasattr(wp, "config") and hasattr(wp.config, "quiet"):
            wp.config.quiet = True
        wp.init()
        wp.set_device(self.config.device)
        self.wp_device = wp.get_device()
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

        self.template_initial_qpos = torch.as_tensor(
            np.stack([item.initial_qpos for item in references]),
            device=self.torch_device,
            dtype=torch.float32,
        )
        self.template_initial_qvel = torch.as_tensor(
            np.stack([item.initial_qvel for item in references]),
            device=self.torch_device,
            dtype=torch.float32,
        )
        self.template_initial_ctrl = torch.as_tensor(
            np.stack([item.initial_ctrl for item in references]),
            device=self.torch_device,
            dtype=torch.float32,
        )
        self.template_controls = torch.as_tensor(
            np.stack([item.controls for item in references]),
            device=self.torch_device,
            dtype=torch.float32,
        )
        self.template_object_z = torch.as_tensor(
            [float(item.initial_object_position[2]) for item in references],
            device=self.torch_device,
            dtype=torch.float32,
        )
        self.actuator_ids = torch.as_tensor(
            reference.actuator_ids, device=self.torch_device, dtype=torch.long
        )
        self.ctrl_low = torch.as_tensor(
            reference.ctrl_low, device=self.torch_device, dtype=torch.float32
        )
        self.ctrl_high = torch.as_tensor(
            reference.ctrl_high, device=self.torch_device, dtype=torch.float32
        )
        self.arm_action_size = int(reference.arm_action_size)
        self.hand_slice = slice(self.arm_action_size, reference.action_dim)
        self.hand_low = self.ctrl_low[self.hand_slice]
        self.hand_high = self.ctrl_high[self.hand_slice]
        self.open_fractions = torch.as_tensor(
            OPEN_FRACTIONS, device=self.torch_device, dtype=torch.float32
        )

        candidate_fractions = []
        grip_fractions = []
        source_seeds = []
        hand_low_np = reference.ctrl_low[reference.hand_slice]
        hand_high_np = reference.ctrl_high[reference.hand_slice]
        for index, template in enumerate(templates):
            episode = DemonstrationEpisode.load(template.manifest)
            candidate_fractions.append(
                np.asarray(episode.candidate.actuator_fractions, dtype=np.float32)
            )
            # Reconstruct the actually executed Ultra grip/preload target from
            # the authoritative low-level reference.  A zero RL hand edit must
            # reproduce the demonstrated closure instead of dropping preload.
            final_hand_ctrl = references[index].controls[-1, references[index].hand_slice]
            grip = (final_hand_ctrl - hand_low_np) / np.maximum(hand_high_np - hand_low_np, 1e-8)
            grip_fractions.append(np.clip(grip, 0.0, 1.0).astype(np.float32))
            source_seeds.append(int(episode.seed))
        self.template_candidate_fractions = torch.as_tensor(
            np.stack(candidate_fractions), device=self.torch_device, dtype=torch.float32
        )
        self.template_grip_fractions = torch.as_tensor(
            np.stack(grip_fractions), device=self.torch_device, dtype=torch.float32
        )
        self.source_seeds = tuple(source_seeds)

        stages = np.asarray(reference.stages, dtype=np.int16)
        self.stages = torch.as_tensor(stages, device=self.torch_device, dtype=torch.int16)
        self.approach_alpha = self._stage_alpha(stages, STAGE_CODES["approach"])
        self.close_alpha = self._stage_alpha(stages, STAGE_CODES["close"])

        bindings = self.host_env.task._require_bindings()
        object_binding = bindings.objects["object"]
        self.object_body_id = int(object_binding.body_id)
        self.object_qvel_adr = int(object_binding.qvel_adr)

        self.physics_steps_per_control = max(
            1, round(reference.control_dt / self.model.opt.timestep)
        )
        with wp.ScopedCapture(device=self.wp_device) as capture:
            mjw.step(self.device_model, self.data)
        self.step_graph = capture.graph

        self._obs_vector = torch.as_tensor(
            [
                float(self.template_count) / 16.0,
                float(self.config.hand_edit_fraction),
                float(self.config.success_lift_height) / 0.1,
                1.0,
            ],
            device=self.torch_device,
            dtype=torch.float32,
        )
        self.obs_dim = len(self._obs_vector)

        self.completed_episodes = 0
        self.last_success_rate = 0.0
        self.last_mean_max_lift = 0.0
        self.last_mean_final_lift = 0.0
        self.last_template_histogram = np.zeros(self.template_count, dtype=np.float64)
        self.best_attempt_lift = -np.inf
        self.best_attempt_final_lift = -np.inf
        self.best_attempt_return = -np.inf
        self.best_attempt_trajectory: GraspTrajectory | None = None
        self.best_attempt_version = 0
        self.best_success_return = -np.inf
        self.best_trajectory: GraspTrajectory | None = None
        self.best_version = 0

    def _validate_references(self, references: tuple[ReferenceTrajectory, ...]) -> None:
        first = references[0]
        if first.hand_action_size != 6:
            raise ValueError(
                f"Grasp editor expects six Dex Hand actuators, got {first.hand_action_size}."
            )
        for index, item in enumerate(references[1:], start=1):
            problems = []
            if item.object_id != first.object_id:
                problems.append("object")
            if item.horizon != first.horizon:
                problems.append("horizon")
            if item.action_dim != first.action_dim:
                problems.append("action_dim")
            if item.arm_action_size != first.arm_action_size:
                problems.append("arm_action_size")
            if not np.array_equal(item.actuator_ids, first.actuator_ids):
                problems.append("actuator_ids")
            if not np.array_equal(item.stages, first.stages):
                problems.append("stages")
            if abs(item.control_dt - first.control_dt) > 1e-9:
                problems.append("control_dt")
            if problems:
                raise ValueError(
                    f"Template {index} is incompatible with template 0: {', '.join(problems)}."
                )

    def _stage_alpha(self, stages: np.ndarray, code: int) -> torch.Tensor:
        indices = np.flatnonzero(stages == code)
        result = np.zeros(len(stages), dtype=np.float32)
        if len(indices):
            result[indices] = np.linspace(0.0, 1.0, len(indices), dtype=np.float32)
        return torch.as_tensor(result, device=self.torch_device)

    def _observation(self) -> torch.Tensor:
        return self._obs_vector.unsqueeze(0).expand(self.num_envs, -1).clone()

    def reset(self) -> torch.Tensor:
        return self._observation()

    def _sync_torch_before_warp(self) -> None:
        if self.torch_device.type == "cuda":
            torch.cuda.synchronize(self.torch_device)

    def _sync_warp_before_torch(self) -> None:
        wp.synchronize_device(self.wp_device)

    def _initialise_selected_worlds(self, template_ids: torch.Tensor) -> None:
        self.qpos.copy_(self.template_initial_qpos[template_ids])
        self.qvel.copy_(self.template_initial_qvel[template_ids])
        self.ctrl.copy_(self.template_initial_ctrl[template_ids])
        self._sync_torch_before_warp()
        mjw.forward(self.device_model, self.data)
        self._sync_warp_before_torch()

    def _hand_controls(
        self,
        template_ids: torch.Tensor,
        final_fractions: torch.Tensor,
        step_index: int,
    ) -> torch.Tensor:
        candidate = self.template_candidate_fractions[template_ids]
        stage = int(self.stages[step_index].item())
        if stage == STAGE_CODES["approach"]:
            alpha = self.approach_alpha[step_index]
            fractions = self.open_fractions.unsqueeze(0) + alpha * (
                candidate - self.open_fractions.unsqueeze(0)
            )
        elif stage == STAGE_CODES["close"]:
            alpha = self.close_alpha[step_index]
            fractions = candidate + alpha * (final_fractions - candidate)
        else:
            fractions = final_fractions
        return self.hand_low.unsqueeze(0) + fractions * (self.hand_high - self.hand_low).unsqueeze(
            0
        )

    def _trajectory_from_world(
        self,
        world: int,
        actions: torch.Tensor,
        template_ids: torch.Tensor,
        controls_history: torch.Tensor,
        reward: torch.Tensor,
        max_lift: torch.Tensor,
        final_lift: torch.Tensor,
        tail_max_speed: torch.Tensor,
        tail_max_angular_speed: torch.Tensor,
        success: torch.Tensor,
    ) -> GraspTrajectory:
        template_id = int(template_ids[world].item())
        action = actions[world].detach().cpu().numpy().astype(np.float32)
        controls = controls_history[:, world].detach().cpu().numpy().astype(np.float32)
        repeated_action = np.repeat(action[None, :], self.horizon, axis=0)
        reference = self.references[template_id]
        template = self.templates[template_id]
        return GraspTrajectory(
            object_id=self.object_id,
            source_manifest=str(template.manifest),
            start_stage="approach",
            action_mode="grasp_edit_hybrid",
            residual_actions=repeated_action,
            controls=controls,
            initial_qpos=reference.initial_qpos.copy(),
            initial_qvel=reference.initial_qvel.copy(),
            success=bool(success[world].item()),
            episode_return=float(reward[world].item()),
            metadata={
                "control_dt": reference.control_dt,
                "source_seed": self.source_seeds[template_id],
                "template_id": template_id,
                "template_label": template.label,
                "template_manifest": str(template.manifest),
                "template_translation_offset": list(template.translation_offset),
                "template_rotation_offset_degrees": list(template.rotation_offset_degrees),
                "template_pre_rl_success": bool(template.success),
                "template_selection_mode": "categorical",
                "base_grip_fractions": self.template_grip_fractions[template_id]
                .detach()
                .cpu()
                .tolist(),
                "hand_edit_normalized": action[1:].tolist(),
                "grasp_edit_action": action.tolist(),
                "mjwarp_max_lift": float(max_lift[world].item()),
                "mjwarp_final_lift": float(final_lift[world].item()),
                "mjwarp_tail_max_speed": float(tail_max_speed[world].item()),
                "mjwarp_tail_max_angular_speed": float(
                    tail_max_angular_speed[world].item()
                ),
                "single_step_grasp_edit": True,
            },
        )

    def step(self, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        if actions.shape != (self.num_envs, self.action_dim):
            raise ValueError(
                f"Grasp-edit actions must have shape {(self.num_envs, self.action_dim)}, "
                f"got {tuple(actions.shape)}."
            )
        actions = actions.to(self.torch_device).float()
        template_ids = torch.round(actions[:, 0]).long().clamp(0, self.template_count - 1)
        hand_edit = torch.clamp(actions[:, 1:], -1.0, 1.0)
        grip_fractions = self.template_grip_fractions[template_ids]
        final_fractions = torch.clamp(
            grip_fractions + self.config.hand_edit_fraction * hand_edit,
            0.0,
            1.0,
        )

        self._initialise_selected_worlds(template_ids)
        initial_z = self.template_object_z[template_ids]
        max_lift = torch.full((self.num_envs,), -float("inf"), device=self.torch_device)
        tail_min_lift = torch.full((self.num_envs,), float("inf"), device=self.torch_device)
        tail_max_speed = torch.zeros((self.num_envs,), device=self.torch_device)
        tail_max_angular_speed = torch.zeros(
            (self.num_envs,), device=self.torch_device
        )
        controls_history = torch.empty(
            (self.horizon, self.num_envs, self.references[0].action_dim),
            device=self.torch_device,
            dtype=torch.float32,
        )

        for step_index in range(self.horizon):
            target = self.template_controls[template_ids, step_index].clone()
            target[:, self.hand_slice] = self._hand_controls(
                template_ids, final_fractions, step_index
            )
            target = torch.maximum(torch.minimum(target, self.ctrl_high), self.ctrl_low)
            controls_history[step_index] = target
            self.ctrl[:, self.actuator_ids] = target
            self._sync_torch_before_warp()
            for _ in range(self.physics_steps_per_control):
                wp.capture_launch(self.step_graph)
            self._sync_warp_before_torch()

            lift = self.xpos[:, self.object_body_id, 2] - initial_z
            max_lift = torch.maximum(max_lift, lift)
            if step_index >= self.horizon - self.config.success_tail_steps:
                tail_min_lift = torch.minimum(tail_min_lift, lift)
                object_velocity = self.qvel[
                    :, self.object_qvel_adr : self.object_qvel_adr + 6
                ]
                tail_max_speed = torch.maximum(
                    tail_max_speed,
                    torch.linalg.vector_norm(object_velocity[:, :3], dim=1),
                )
                tail_max_angular_speed = torch.maximum(
                    tail_max_angular_speed,
                    torch.linalg.vector_norm(object_velocity[:, 3:], dim=1),
                )

        final_lift = self.xpos[:, self.object_body_id, 2] - initial_z
        success = (
            (tail_min_lift >= self.config.success_lift_height)
            & (tail_max_speed <= self.config.maximum_object_speed)
            & (
                tail_max_angular_speed
                <= self.config.maximum_object_angular_speed
            )
        )

        max_progress = torch.clamp(max_lift / self.config.success_lift_height, 0.0, 1.0)
        final_progress = torch.clamp(final_lift / self.config.success_lift_height, 0.0, 1.0)
        speed_ratio = torch.clamp(
            tail_max_speed / self.config.maximum_object_speed, min=0.0, max=5.0
        )
        angular_speed_ratio = torch.clamp(
            tail_max_angular_speed / self.config.maximum_object_angular_speed,
            min=0.0,
            max=5.0,
        )
        hand_cost = hand_edit.square().mean(dim=1)
        reward = (
            5.0 * max_progress
            + 5.0 * final_progress
            + 12.0 * success.float()
            - 0.25 * speed_ratio
            - 0.50 * angular_speed_ratio
            - 0.02 * hand_cost
        )

        self.completed_episodes += self.num_envs
        self.last_success_rate = float(success.float().mean().item())
        self.last_mean_max_lift = float(max_lift.mean().item())
        self.last_mean_final_lift = float(final_lift.mean().item())
        counts = torch.bincount(template_ids, minlength=self.template_count).float()
        self.last_template_histogram = (counts / float(self.num_envs)).detach().cpu().numpy()

        attempt_world = int(torch.argmax(reward).item())
        attempt_lift = float(max_lift[attempt_world].item())
        attempt_final = float(final_lift[attempt_world].item())
        attempt_return = float(reward[attempt_world].item())
        if attempt_return > self.best_attempt_return:
            self.best_attempt_lift = attempt_lift
            self.best_attempt_final_lift = attempt_final
            self.best_attempt_return = attempt_return
            self.best_attempt_trajectory = self._trajectory_from_world(
                attempt_world,
                actions,
                template_ids,
                controls_history,
                reward,
                max_lift,
                final_lift,
                tail_max_speed,
                tail_max_angular_speed,
                success,
            )
            self.best_attempt_version += 1

        if success.any():
            successful = torch.nonzero(success, as_tuple=False).flatten()
            success_rewards = reward[successful]
            world = int(successful[int(torch.argmax(success_rewards).item())].item())
            score = float(reward[world].item())
            if score > self.best_success_return:
                self.best_success_return = score
                self.best_trajectory = self._trajectory_from_world(
                    world,
                    actions,
                    template_ids,
                    controls_history,
                    reward,
                    max_lift,
                    final_lift,
                    tail_max_speed,
                    tail_max_angular_speed,
                    success,
                )
                self.best_version += 1

        done = torch.ones(self.num_envs, device=self.torch_device, dtype=torch.bool)
        return (
            self._observation(),
            reward,
            done,
            {
                "success_rate": self.last_success_rate,
                "new_successes": int(success.sum().item()),
            },
        )

    def training_metrics(self) -> dict[str, float]:
        result: dict[str, float] = {
            "episode_success_rate": self.last_success_rate,
            "mean_max_lift": self.last_mean_max_lift,
            "mean_final_lift": self.last_mean_final_lift,
            "best_attempt_lift": (
                float(self.best_attempt_lift) if np.isfinite(self.best_attempt_lift) else 0.0
            ),
            "best_attempt_final_lift": (
                float(self.best_attempt_final_lift)
                if np.isfinite(self.best_attempt_final_lift)
                else 0.0
            ),
            "best_attempt_return": (
                float(self.best_attempt_return)
                if np.isfinite(self.best_attempt_return)
                else 0.0
            ),
            "best_success_return": (
                float(self.best_success_return) if np.isfinite(self.best_success_return) else 0.0
            ),
            "completed_episodes": float(self.completed_episodes),
        }
        for index, value in enumerate(self.last_template_histogram):
            result[f"template_{index}_rate"] = float(value)
        return result

    def template_summary(self) -> list[dict]:
        return [
            {
                "id": index,
                "label": template.label,
                "translation_offset": list(template.translation_offset),
                "rotation_offset_degrees": list(template.rotation_offset_degrees),
                "success_before_edit": bool(template.success),
                "precheck_score": template.precheck_score,
                "precheck_position_error": template.precheck_position_error,
                "precheck_orientation_error": template.precheck_orientation_error,
                "source_manifest": str(template.source_manifest),
                "manifest": str(template.manifest),
            }
            for index, template in enumerate(self.templates)
        ]

    def close(self) -> None:
        self.host_env.close()
