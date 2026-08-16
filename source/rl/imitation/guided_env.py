"""Geometry-aware BC-guided residual RL with state-conditioned grasp curriculum."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from source.rl.imitation.bc import BC_TARGET_TYPE, load_bc_policy
from source.rl.imitation.geometry_env import GeometryAwareResidualLiftEnv
from source.rl.residual.env import ResidualLiftConfig
from source.rl.residual.reference import STAGE_CODES
from source.rl.residual.trajectory import ResidualTrajectory


@dataclass(frozen=True)
class GuidedResidualConfig(ResidualLiftConfig):
    action_mode: str = "arm_hand"
    reward_version: int = 3
    maximum_object_speed: float = 0.10
    success_hold_steps: int = 12
    opposition_threshold: float = 0.25
    grasp_ready_steps: int = 3

    bc_approach_blend: float = 0.15
    bc_close_blend: float = 0.85
    bc_hold_blend: float = 0.90
    bc_lift_blend: float = 0.75
    bc_verify_blend: float = 0.75

    arm_approach_gate: float = 0.0
    arm_close_gate: float = 0.10
    arm_hold_gate: float = 0.20
    arm_lift_gate: float = 1.00
    arm_verify_gate: float = 0.50
    locked_arm_lift_gate: float = 0.10

    hand_approach_gate: float = 0.15
    hand_close_gate: float = 0.50
    hand_hold_gate: float = 0.40
    hand_lift_gate: float = 0.30
    hand_verify_gate: float = 0.25
    locked_hand_lift_gate: float = 0.65

    grasp_contact_reward: float = 0.35
    grasp_thumb_reward: float = 0.80
    grasp_opposition_reward: float = 1.60
    opposition_geometry_reward: float = 0.80
    hold_opposition_bonus: float = 0.80
    guided_lift_reward: float = 4.00
    guided_success_reward: float = 15.0
    speed_penalty: float = 1.50

    def validate(self) -> None:
        super().validate()
        if self.action_mode != "arm_hand":
            raise ValueError("BC-guided residual RL requires action_mode='arm_hand'.")
        if not 0.0 <= self.opposition_threshold <= 1.0 or self.grasp_ready_steps <= 0:
            raise ValueError("Invalid opposition threshold/grasp_ready_steps.")
        values = (
            self.bc_approach_blend,
            self.bc_close_blend,
            self.bc_hold_blend,
            self.bc_lift_blend,
            self.bc_verify_blend,
            self.arm_approach_gate,
            self.arm_close_gate,
            self.arm_hold_gate,
            self.arm_lift_gate,
            self.arm_verify_gate,
            self.locked_arm_lift_gate,
            self.hand_approach_gate,
            self.hand_close_gate,
            self.hand_hold_gate,
            self.hand_lift_gate,
            self.hand_verify_gate,
            self.locked_hand_lift_gate,
        )
        if any(not 0.0 <= value <= 1.0 for value in values):
            raise ValueError("BC blend and curriculum gates must lie in [0, 1].")


class BCGuidedResidualLiftEnv(GeometryAwareResidualLiftEnv):
    """Coarse reference + BC hand correction + 13-D PPO residual."""

    def __init__(
        self,
        reference_manifest: str | Path,
        bc_checkpoint: str | Path,
        config: GuidedResidualConfig | None = None,
    ) -> None:
        self.guided_config = config or GuidedResidualConfig()
        super().__init__(reference_manifest, self.guided_config)
        if self.action_dim != self.reference.action_dim or self.reference.arm_action_size != 7:
            raise ValueError(
                "BC-guided training expects 7 arm + 6 hand actions, got "
                f"action_dim={self.action_dim}, arm={self.reference.arm_action_size}."
            )
        self.bc_policy = load_bc_policy(bc_checkpoint, device=self.torch_device)
        if self.bc_policy.obs_dim != self.obs_dim:
            raise ValueError(
                f"BC observation dim {self.bc_policy.obs_dim} does not match env {self.obs_dim}. "
                "Rebuild BC after observation-schema changes."
            )
        self.hand_start = self.reference.arm_action_size
        self.hand_low = self.ctrl_low[self.hand_start :]
        self.hand_high = self.ctrl_high[self.hand_start :]
        self.grasp_ready_streak = torch.zeros(
            self.num_envs, device=self.torch_device, dtype=torch.int32
        )
        self.best_attempt_quality = -np.inf
        self._guided_action_sum = torch.zeros((), device=self.torch_device)
        self._guided_bc_sum = torch.zeros((), device=self.torch_device)
        self._guided_steps = 0
        self._last_guided_metrics = {
            "bc_blend": 0.0,
            "effective_residual_abs": 0.0,
            "grasp_ready_rate": 0.0,
            "best_attempt_quality": 0.0,
        }

    def _reset_all(self) -> None:
        super()._reset_all()
        if hasattr(self, "grasp_ready_streak"):
            self.grasp_ready_streak.zero_()

    def _stage_code(self) -> int:
        index = min(self.step_index, self.reference.horizon - 1)
        return round(float(self.reference.stages[index]))

    def _grasp_state(self):
        contact_signal = self._contact_signal()
        if self.has_digit_contacts:
            thumb = self.digit_flags[:, 4] > 0
            finger = self.digit_flags[:, :4].sum(dim=1) > 0
        else:
            thumb = contact_signal >= 1.0
            finger = contact_signal >= 2.0
        opposition_score = self._opposition_score()
        opposition = thumb & finger & (opposition_score >= self.guided_config.opposition_threshold)
        ready = opposition & (contact_signal >= self.guided_config.minimum_contact_digits)
        return contact_signal, thumb, finger, opposition_score, opposition, ready

    def _bc_blend_values(self, stage: int, ready: torch.Tensor) -> torch.Tensor:
        cfg = self.guided_config
        if stage == STAGE_CODES["approach"]:
            value = cfg.bc_approach_blend
        elif stage == STAGE_CODES["close"]:
            value = cfg.bc_close_blend
        elif stage == STAGE_CODES["hold"]:
            value = cfg.bc_hold_blend
        elif stage == STAGE_CODES["lift"]:
            value = cfg.bc_lift_blend
        elif stage == STAGE_CODES["verify"]:
            value = cfg.bc_verify_blend
        else:
            value = 0.0
        result = torch.full((self.num_envs, 1), float(value), device=self.torch_device)
        if stage in {STAGE_CODES["lift"], STAGE_CODES["verify"]}:
            locked = max(cfg.bc_close_blend, value)
            result = torch.where(ready.unsqueeze(1), result, torch.full_like(result, float(locked)))
        return result

    def _curriculum_gate(self, stage: int, ready: torch.Tensor) -> torch.Tensor:
        cfg = self.guided_config
        if stage == STAGE_CODES["approach"]:
            arm, hand = cfg.arm_approach_gate, cfg.hand_approach_gate
        elif stage == STAGE_CODES["close"]:
            arm, hand = cfg.arm_close_gate, cfg.hand_close_gate
        elif stage == STAGE_CODES["hold"]:
            arm, hand = cfg.arm_hold_gate, cfg.hand_hold_gate
        elif stage == STAGE_CODES["lift"]:
            arm, hand = cfg.arm_lift_gate, cfg.hand_lift_gate
        elif stage == STAGE_CODES["verify"]:
            arm, hand = cfg.arm_verify_gate, cfg.hand_verify_gate
        else:
            arm, hand = 0.0, 0.10
        gate = torch.full((self.num_envs, self.action_dim), float(hand), device=self.torch_device)
        gate[:, : self.hand_start] = float(arm)
        if stage in {STAGE_CODES["lift"], STAGE_CODES["verify"]}:
            gate[:, : self.hand_start] = torch.where(
                ready.unsqueeze(1),
                gate[:, : self.hand_start],
                torch.full_like(gate[:, : self.hand_start], cfg.locked_arm_lift_gate),
            )
            gate[:, self.hand_start :] = torch.where(
                ready.unsqueeze(1),
                gate[:, self.hand_start :],
                torch.full_like(gate[:, self.hand_start :], cfg.locked_hand_lift_gate),
            )
        return gate

    @torch.no_grad()
    def _bc_hand_control(self, observation: torch.Tensor) -> torch.Tensor:
        correction = self.bc_policy(observation)
        coarse = self.coarse_reference_controls[self.step_index, self.hand_start :].unsqueeze(0)
        span = (self.hand_high - self.hand_low).unsqueeze(0)
        return torch.maximum(
            torch.minimum(coarse + correction * span, self.hand_high),
            self.hand_low,
        )

    def _apply_residual(self, actions: torch.Tensor) -> torch.Tensor:
        actions = torch.clamp(actions, -1.0, 1.0)
        stage = self._stage_code()
        _, _, _, _, _, grasp_ready_now = self._grasp_state()
        self.grasp_ready_streak = torch.where(
            grasp_ready_now,
            self.grasp_ready_streak + 1,
            torch.zeros_like(self.grasp_ready_streak),
        )
        ready = self.grasp_ready_streak >= self.guided_config.grasp_ready_steps

        reference = (
            self.coarse_reference_controls[self.step_index].unsqueeze(0).expand(self.num_envs, -1)
        )
        target = reference.clone()
        observation = self._observation()
        blend = self._bc_blend_values(stage, ready)
        bc_hand = self._bc_hand_control(observation)
        target[:, self.hand_start :] = (1.0 - blend) * reference[
            :, self.hand_start :
        ] + blend * bc_hand

        gate = self._curriculum_gate(stage, ready)
        target[:, self.controlled_positions] += actions * self.residual_scale.unsqueeze(0) * gate
        target = torch.maximum(torch.minimum(target, self.ctrl_high), self.ctrl_low)
        self.ctrl[:, self.actuator_ids] = target

        effective = (
            target[:, self.controlled_positions] - reference[:, self.controlled_positions]
        ) / self.residual_scale.unsqueeze(0)
        self.action_history[self.step_index] = effective
        self._guided_steps += 1
        self._guided_action_sum += effective.abs().mean()
        self._guided_bc_sum += blend.mean()
        self._last_guided_metrics["grasp_ready_rate"] = float(ready.float().mean().item())
        return target

    def _reward(self, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        cfg = self.guided_config
        stage = self._stage_code()
        object_position = self.xpos[:, self.object_body_id]
        lift = object_position[:, 2] - self.initial_object_z
        lift_progress = torch.clamp(lift / cfg.success_lift_height, 0.0, 1.0)
        contact_signal, thumb_contact, _, opposition_score, opposition, _ = self._grasp_state()
        contact_progress = torch.clamp(contact_signal / float(cfg.minimum_contact_digits), 0.0, 1.0)
        self.episode_max_lift = torch.maximum(self.episode_max_lift, lift)
        self.episode_opposition_steps += opposition.to(torch.int32)

        object_velocity = self.qvel[:, self.object_qvel_adr : self.object_qvel_adr + 6]
        translational_speed = torch.linalg.vector_norm(object_velocity[:, :3], dim=1)
        stable = (
            (lift >= cfg.success_lift_height)
            & (contact_signal >= cfg.minimum_contact_digits)
            & opposition
            & (translational_speed <= cfg.maximum_object_speed)
        )
        self.success_streak = torch.where(
            stable, self.success_streak + 1, torch.zeros_like(self.success_streak)
        )
        success_now = self.success_streak >= cfg.success_hold_steps
        new_success = success_now & ~self.success_reached
        self.success_reached |= success_now
        dropped = lift < -cfg.drop_margin

        close_or_hold = stage in {STAGE_CODES["close"], STAGE_CODES["hold"]}
        lift_or_verify = stage in {STAGE_CODES["lift"], STAGE_CODES["verify"]}
        grasp_scale = 1.0 if close_or_hold else (0.55 if lift_or_verify else 0.15)
        opposition_scale = 1.0 if close_or_hold else (0.80 if lift_or_verify else 0.15)
        gated_lift = lift_progress * opposition.float()
        speed_excess = torch.relu(translational_speed - cfg.maximum_object_speed)
        unsafe_motion = speed_excess / max(cfg.maximum_object_speed, 1e-6)
        unsafe_motion = unsafe_motion * (1.0 + (~opposition).float())
        action_cost = actions.square().mean(dim=1)
        delta_cost = (actions - self.last_action).square().mean(dim=1)
        reward = (
            grasp_scale * cfg.grasp_contact_reward * contact_progress
            + grasp_scale * cfg.grasp_thumb_reward * thumb_contact.float()
            + opposition_scale * cfg.grasp_opposition_reward * opposition.float()
            + cfg.opposition_geometry_reward * opposition_score * thumb_contact.float()
            + (
                cfg.hold_opposition_bonus * opposition.float()
                if stage == STAGE_CODES["hold"]
                else 0.0
            )
            + (cfg.guided_lift_reward * gated_lift if lift_or_verify else 0.0)
            + cfg.guided_success_reward * new_success.float()
            - cfg.speed_penalty * unsafe_motion
            - cfg.drop_penalty * dropped.float()
            - cfg.action_penalty * action_cost
            - cfg.action_delta_penalty * delta_cost
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

    def _trajectory_for_world(
        self, world: int, success: bool, score: float, quality: float
    ) -> ResidualTrajectory:
        residual = self.action_history[:, world].detach().cpu().numpy()
        controls = self.coarse_reference.controls.copy()
        positions = self.controlled_positions.detach().cpu().numpy()
        physical_delta = residual * self.residual_scale.detach().cpu().numpy()[None, :]
        controls[:, positions] += physical_delta
        controls = np.clip(
            controls, self.reference.ctrl_low[None, :], self.reference.ctrl_high[None, :]
        )
        return ResidualTrajectory(
            object_id=self.reference.object_id,
            source_manifest=str(self.coarse_reference.source_manifest),
            start_stage=self.reference.start_stage,
            action_mode=self.config.action_mode,
            residual_actions=residual,
            controls=controls,
            initial_qpos=self.reference.initial_qpos.copy(),
            initial_qvel=self.reference.initial_qvel.copy(),
            success=success,
            episode_return=score,
            metadata={
                "actuator_names": list(self.reference.actuator_names),
                "controlled_positions": positions.tolist(),
                "residual_scale": self.residual_scale.detach().cpu().numpy().tolist(),
                "success_lift_height": self.config.success_lift_height,
                "success_hold_steps": self.config.success_hold_steps,
                "minimum_contact_digits": self.config.minimum_contact_digits,
                "opposition_threshold": self.guided_config.opposition_threshold,
                "reward_version": self.guided_config.reward_version,
                "observation_schema": self.observation_schema(),
                "control_dt": self.reference.control_dt,
                "source_seed": self.reference.source_seed,
                "input_expert_manifest": str(self.reference.source_manifest),
                "coarse_reference_manifest": str(self.coarse_reference.source_manifest),
                "bc_target_type": BC_TARGET_TYPE,
                "mjwarp_max_lift": float(self.episode_max_lift[world].item()),
                "mjwarp_opposition_steps": int(self.episode_opposition_steps[world].item()),
                "diagnostic_quality": float(quality),
            },
        )

    def _finalize_episode(self) -> None:
        success = self.success_streak >= self.config.success_hold_steps
        returns = self.episode_return
        final_lift = self.xpos[:, self.object_body_id, 2] - self.initial_object_z
        opp_fraction = self.episode_opposition_steps.float() / float(max(self.reference.horizon, 1))
        max_progress = torch.clamp(
            self.episode_max_lift / self.config.success_lift_height, -1.0, 2.0
        )
        final_progress = torch.clamp(final_lift / self.config.success_lift_height, -1.0, 1.5)
        quality = (
            2.0 * opp_fraction
            + 1.5 * final_progress
            + 0.25 * max_progress
            + returns / float(max(self.reference.horizon, 1))
        )
        self.completed_episodes += self.num_envs
        self.last_success_rate = float(success.float().mean().item())
        self.last_mean_return = float(returns.mean().item())

        attempt_world = int(torch.argmax(quality).item())
        attempt_quality = float(quality[attempt_world].item())
        attempt_lift = float(self.episode_max_lift[attempt_world].item())
        attempt_return = float(returns[attempt_world].item())
        if attempt_quality > self.best_attempt_quality + 1e-6:
            self.best_attempt_quality = attempt_quality
            self.best_attempt_lift = attempt_lift
            self.best_attempt_return = attempt_return
            self.best_attempt_trajectory = self._trajectory_for_world(
                attempt_world, bool(success[attempt_world].item()), attempt_return, attempt_quality
            )
            self.best_attempt_version += 1

        if success.any():
            successful = torch.nonzero(success, as_tuple=False).flatten()
            successful_returns = returns[successful]
            world = int(successful[int(torch.argmax(successful_returns).item())].item())
            score = float(returns[world].item())
            if score > self.best_success_return:
                self.best_success_return = score
                self.best_trajectory = self._trajectory_for_world(
                    world, True, score, float(quality[world].item())
                )
                self.best_version += 1

    def training_metrics(self) -> dict[str, float]:
        result = super().training_metrics()
        if self._guided_steps:
            self._last_guided_metrics.update(
                {
                    "bc_blend": float((self._guided_bc_sum / self._guided_steps).item()),
                    "effective_residual_abs": float(
                        (self._guided_action_sum / self._guided_steps).item()
                    ),
                    "best_attempt_quality": float(self.best_attempt_quality)
                    if np.isfinite(self.best_attempt_quality)
                    else 0.0,
                }
            )
            self._guided_steps = 0
            self._guided_bc_sum.zero_()
            self._guided_action_sum.zero_()
        return result | self._last_guided_metrics
