"""Primitive-conditioned extension of the single-step MJWarp grasp editor.

This module deliberately leaves :mod:`source.rl.grasp_edit.env` untouched so
``wrap`` remains a clean baseline.  The categorical branch selects
``wrist-template x grasp-primitive`` while the continuous branch still edits
six physical Dex Hand actuators.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import warp as wp

from source.rl.grasp_edit.env import GraspEditConfig, MjWarpGraspEditEnv
from source.rl.grasp_edit.primitives import resolve_grasp_primitives
from source.rl.residual.reference import STAGE_CODES
from source.rl.residual.trajectory import ResidualTrajectory


@dataclass(frozen=True)
class PrimitiveGraspEditConfig(GraspEditConfig):
    grasp_primitives: tuple[str, ...] = ("wrap",)
    primitive_bias_scale: float = 1.0

    def validate(self) -> None:
        super().validate()
        resolve_grasp_primitives(self.grasp_primitives)
        if not 0.0 <= self.primitive_bias_scale <= 2.0:
            raise ValueError("primitive_bias_scale must lie in [0, 2].")


class PrimitiveMjWarpGraspEditEnv(MjWarpGraspEditEnv):
    """Choose a wrist template and grasp style, then apply a continuous hand edit."""

    def __init__(self, object_id, templates, config: PrimitiveGraspEditConfig | None = None):
        config = config or PrimitiveGraspEditConfig()
        super().__init__(object_id, templates, config)

        self.base_template_count = len(self.templates)
        self.primitives = resolve_grasp_primitives(config.grasp_primitives)
        self.primitive_count = len(self.primitives)
        self.primitive_names = tuple(item.name for item in self.primitives)

        # HybridPPOTrainer uses ``template_count`` as its categorical cardinality.
        # Here one categorical choice is (wrist template, grasp primitive).
        self.template_count = self.base_template_count * self.primitive_count

        self.primitive_approach_bias = torch.as_tensor(
            np.asarray([item.approach_bias for item in self.primitives], dtype=np.float32),
            device=self.torch_device,
        )
        self.primitive_final_bias = torch.as_tensor(
            np.asarray([item.final_bias for item in self.primitives], dtype=np.float32),
            device=self.torch_device,
        )
        self.primitive_close_power = torch.as_tensor(
            np.asarray([item.close_power for item in self.primitives], dtype=np.float32),
            device=self.torch_device,
        )

        # Keep the observation compact and object-local.  The policy is trained
        # per object, so these values describe search-space structure rather than
        # serving as a general visual/object encoder.
        self._obs_vector = torch.as_tensor(
            [
                float(self.base_template_count) / 16.0,
                float(self.primitive_count) / 4.0,
                float(self.config.hand_edit_fraction),
                float(self.config.success_lift_height) / 0.1,
                1.0,
            ],
            device=self.torch_device,
            dtype=torch.float32,
        )
        self.obs_dim = len(self._obs_vector)
        self.last_template_histogram = np.zeros(self.template_count, dtype=np.float64)
        self.last_primitive_histogram = np.zeros(self.primitive_count, dtype=np.float64)

    def _decode_choices(self, choice_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        template_ids = torch.div(choice_ids, self.primitive_count, rounding_mode="floor")
        primitive_ids = torch.remainder(choice_ids, self.primitive_count)
        return template_ids, primitive_ids

    def _hand_controls(
        self,
        template_ids: torch.Tensor,
        primitive_ids: torch.Tensor,
        final_fractions: torch.Tensor,
        step_index: int,
    ) -> torch.Tensor:
        scale = float(self.config.primitive_bias_scale)
        candidate = torch.clamp(
            self.template_candidate_fractions[template_ids]
            + scale * self.primitive_approach_bias[primitive_ids],
            0.0,
            1.0,
        )
        styled_final = torch.clamp(
            final_fractions + scale * self.primitive_final_bias[primitive_ids],
            0.0,
            1.0,
        )

        stage = int(self.stages[step_index].item())
        if stage == STAGE_CODES["approach"]:
            alpha = self.approach_alpha[step_index]
            fractions = self.open_fractions.unsqueeze(0) + alpha * (
                candidate - self.open_fractions.unsqueeze(0)
            )
        elif stage == STAGE_CODES["close"]:
            alpha = torch.pow(
                self.close_alpha[step_index], self.primitive_close_power[primitive_ids]
            ).unsqueeze(1)
            fractions = candidate + alpha * (styled_final - candidate)
        else:
            fractions = styled_final

        return self.hand_low.unsqueeze(0) + fractions * (
            self.hand_high - self.hand_low
        ).unsqueeze(0)

    def _trajectory_from_world(
        self,
        world: int,
        actions: torch.Tensor,
        choice_ids: torch.Tensor,
        template_ids: torch.Tensor,
        primitive_ids: torch.Tensor,
        controls_history: torch.Tensor,
        reward: torch.Tensor,
        max_lift: torch.Tensor,
        final_lift: torch.Tensor,
        tail_max_speed: torch.Tensor,
        tail_max_angular_speed: torch.Tensor,
        success: torch.Tensor,
    ) -> ResidualTrajectory:
        choice_id = int(choice_ids[world].item())
        template_id = int(template_ids[world].item())
        primitive_id = int(primitive_ids[world].item())
        primitive = self.primitives[primitive_id]
        action = actions[world].detach().cpu().numpy().astype(np.float32)
        controls = controls_history[:, world].detach().cpu().numpy().astype(np.float32)
        repeated_action = np.repeat(action[None, :], self.horizon, axis=0)
        reference = self.references[template_id]
        template = self.templates[template_id]
        return ResidualTrajectory(
            object_id=self.object_id,
            source_manifest=str(template.manifest),
            start_stage="approach",
            action_mode="grasp_edit_primitive",
            residual_actions=repeated_action,
            controls=controls,
            initial_qpos=reference.initial_qpos.copy(),
            initial_qvel=reference.initial_qvel.copy(),
            success=bool(success[world].item()),
            episode_return=float(reward[world].item()),
            metadata={
                "control_dt": reference.control_dt,
                "source_seed": self.source_seeds[template_id],
                "choice_id": choice_id,
                "template_id": template_id,
                "template_label": template.label,
                "grasp_primitive_id": primitive_id,
                "grasp_primitive": primitive.name,
                "grasp_primitive_description": primitive.description,
                "template_manifest": str(template.manifest),
                "template_translation_offset": list(template.translation_offset),
                "template_rotation_offset_degrees": list(template.rotation_offset_degrees),
                "template_pre_rl_success": bool(template.success),
                "template_selection_mode": "categorical_template_x_primitive",
                "base_grip_fractions": self.template_grip_fractions[
                    template_id
                ].detach().cpu().tolist(),
                "hand_edit_normalized": action[1:].tolist(),
                "grasp_edit_action": action.tolist(),
                "mjwarp_max_lift": float(max_lift[world].item()),
                "mjwarp_final_lift": float(final_lift[world].item()),
                "mjwarp_tail_max_speed": float(tail_max_speed[world].item()),
                "mjwarp_tail_max_angular_speed": float(
                    tail_max_angular_speed[world].item()
                ),
                "single_step_grasp_edit": True,
                "primitive_conditioned": True,
            },
        )

    def step(self, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        if actions.shape != (self.num_envs, self.action_dim):
            raise ValueError(
                f"Grasp-edit actions must have shape {(self.num_envs, self.action_dim)}, "
                f"got {tuple(actions.shape)}."
            )
        actions = actions.to(self.torch_device).float()
        choice_ids = torch.round(actions[:, 0]).long().clamp(0, self.template_count - 1)
        template_ids, primitive_ids = self._decode_choices(choice_ids)
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
        tail_lift_sum = torch.zeros((self.num_envs,), device=self.torch_device)
        tail_max_speed = torch.zeros((self.num_envs,), device=self.torch_device)
        tail_max_angular_speed = torch.zeros(
            (self.num_envs,), device=self.torch_device
        )
        tail_samples = 0
        controls_history = torch.empty(
            (self.horizon, self.num_envs, self.references[0].action_dim),
            device=self.torch_device,
            dtype=torch.float32,
        )

        for step_index in range(self.horizon):
            target = self.template_controls[template_ids, step_index].clone()
            target[:, self.hand_slice] = self._hand_controls(
                template_ids, primitive_ids, final_fractions, step_index
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
                tail_lift_sum += lift
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
                tail_samples += 1

        final_lift = self.xpos[:, self.object_body_id, 2] - initial_z
        tail_mean_lift = tail_lift_sum / float(max(tail_samples, 1))
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
        tail_mean_progress = torch.clamp(
            tail_mean_lift / self.config.success_lift_height, 0.0, 1.0
        )
        tail_min_progress = torch.clamp(
            tail_min_lift / self.config.success_lift_height, 0.0, 1.0
        )
        speed_ratio = torch.clamp(
            tail_max_speed / self.config.maximum_object_speed, min=0.0, max=5.0
        )
        angular_speed_ratio = torch.clamp(
            tail_max_angular_speed / self.config.maximum_object_angular_speed,
            min=0.0,
            max=5.0,
        )
        hand_cost = hand_edit.square().mean(dim=1)

        # Dense reward should point in the same direction as the formal success
        # criterion. Max lift is kept only as a small exploration signal;
        # sustained tail lift dominates so "hook and drop" cannot win simply by
        # creating a transient upward impulse.
        reward = (
            1.0 * max_progress
            + 3.0 * final_progress
            + 4.0 * tail_mean_progress
            + 5.0 * tail_min_progress
            + 15.0 * success.float()
            - 0.25 * speed_ratio
            - 0.50 * angular_speed_ratio
            - 0.02 * hand_cost
        )

        self.completed_episodes += self.num_envs
        self.last_success_rate = float(success.float().mean().item())
        self.last_mean_max_lift = float(max_lift.mean().item())
        self.last_mean_final_lift = float(final_lift.mean().item())
        self.last_mean_tail_lift = float(tail_mean_lift.mean().item())
        self.last_mean_tail_min_lift = float(tail_min_lift.mean().item())
        counts = torch.bincount(choice_ids, minlength=self.template_count).float()
        self.last_template_histogram = (counts / float(self.num_envs)).detach().cpu().numpy()
        primitive_counts = torch.bincount(primitive_ids, minlength=self.primitive_count).float()
        self.last_primitive_histogram = (
            primitive_counts / float(self.num_envs)
        ).detach().cpu().numpy()

        # Rank diagnostic attempts by sustained physical quality rather than
        # transient max lift. This makes best_attempt useful as a seed for the
        # later arm+hand residual stage.
        attempt_quality = (
            0.25 * max_lift
            + 1.0 * final_lift
            + 1.5 * tail_mean_lift
            + 2.0 * tail_min_lift
            - 0.01 * speed_ratio
            - 0.02 * angular_speed_ratio
        )
        attempt_world = int(torch.argmax(attempt_quality).item())
        attempt_lift = float(max_lift[attempt_world].item())
        attempt_final = float(final_lift[attempt_world].item())
        if (
            attempt_lift > self.best_attempt_lift + 1e-6
            or (
                abs(attempt_lift - self.best_attempt_lift) <= 1e-6
                and attempt_final > self.best_attempt_final_lift
            )
        ):
            self.best_attempt_lift = attempt_lift
            self.best_attempt_final_lift = attempt_final
            self.best_attempt_trajectory = self._trajectory_from_world(
                attempt_world,
                actions,
                choice_ids,
                template_ids,
                primitive_ids,
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
                    choice_ids,
                    template_ids,
                    primitive_ids,
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
        return self._observation(), reward, done, {
            "success_rate": self.last_success_rate,
            "new_successes": int(success.sum().item()),
        }

    def training_metrics(self) -> dict[str, float]:
        result = super().training_metrics()
        # super() reads last_template_histogram, which now indexes categorical
        # choices rather than wrist templates.  Keep the old metric names for
        # HybridPPOTrainer/callback compatibility and add style marginals.
        for index, value in enumerate(self.last_primitive_histogram):
            result[f"primitive_{self.primitives[index].name}_rate"] = float(value)
        result["mean_tail_lift"] = float(getattr(self, "last_mean_tail_lift", 0.0))
        result["mean_tail_min_lift"] = float(
            getattr(self, "last_mean_tail_min_lift", 0.0)
        )
        return result

    def template_summary(self) -> list[dict]:
        rows = []
        for choice_id in range(self.template_count):
            template_id = choice_id // self.primitive_count
            primitive_id = choice_id % self.primitive_count
            template = self.templates[template_id]
            primitive = self.primitives[primitive_id]
            rows.append(
                {
                    "id": choice_id,
                    "template_id": template_id,
                    "primitive_id": primitive_id,
                    "primitive": primitive.name,
                    "primitive_description": primitive.description,
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
            )
        return rows
