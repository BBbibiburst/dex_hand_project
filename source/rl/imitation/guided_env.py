"""BC-guided, grasp-aware residual RL environment.

The successful-demo BC policy supplies a hand-closure prior.  PPO retains a
13-D arm+hand residual action, but stage-dependent gates prevent the random
initial policy from destroying the reference approach before a grasp can form.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from source.rl.imitation.bc import load_bc_policy
from source.rl.residual.env import MjWarpResidualLiftEnv, ResidualLiftConfig
from source.rl.residual.reference import STAGE_CODES


@dataclass(frozen=True)
class GuidedResidualConfig(ResidualLiftConfig):
    action_mode: str = "arm_hand"
    reward_version: int = 3
    maximum_object_speed: float = 0.10
    success_hold_steps: int = 12

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

    hand_approach_gate: float = 0.15
    hand_close_gate: float = 0.50
    hand_hold_gate: float = 0.40
    hand_lift_gate: float = 0.30
    hand_verify_gate: float = 0.25

    grasp_contact_reward: float = 0.35
    grasp_thumb_reward: float = 0.80
    grasp_opposition_reward: float = 1.60
    hold_opposition_bonus: float = 0.80
    guided_lift_reward: float = 4.00
    guided_success_reward: float = 15.0
    speed_penalty: float = 1.50

    def validate(self) -> None:
        super().validate()
        if self.action_mode != "arm_hand":
            raise ValueError("BC-guided residual RL currently requires action_mode='arm_hand'.")
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
            self.hand_approach_gate,
            self.hand_close_gate,
            self.hand_hold_gate,
            self.hand_lift_gate,
            self.hand_verify_gate,
        )
        if any(not 0.0 <= value <= 1.0 for value in values):
            raise ValueError("BC blend and curriculum gates must lie in [0, 1].")


class BCGuidedResidualLiftEnv(MjWarpResidualLiftEnv):
    """Residual environment whose hand baseline comes from successful demonstrations."""

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
                f"BC observation dim {self.bc_policy.obs_dim} does not match env {self.obs_dim}."
            )
        self.hand_start = self.reference.arm_action_size
        self.hand_positions = torch.arange(
            self.hand_start,
            self.reference.action_dim,
            device=self.torch_device,
            dtype=torch.long,
        )
        self.hand_low = self.ctrl_low[self.hand_start :]
        self.hand_high = self.ctrl_high[self.hand_start :]
        self._guided_action_sum = torch.zeros((), device=self.torch_device)
        self._guided_bc_sum = torch.zeros((), device=self.torch_device)
        self._guided_steps = 0
        self._last_guided_metrics = {"bc_blend": 0.0, "effective_residual_abs": 0.0}

    def _stage_code(self) -> int:
        index = min(self.step_index, self.reference.horizon - 1)
        return int(round(float(self.reference.stages[index])))

    def _bc_blend(self, stage: int) -> float:
        cfg = self.guided_config
        if stage == STAGE_CODES["approach"]:
            return cfg.bc_approach_blend
        if stage == STAGE_CODES["close"]:
            return cfg.bc_close_blend
        if stage == STAGE_CODES["hold"]:
            return cfg.bc_hold_blend
        if stage == STAGE_CODES["lift"]:
            return cfg.bc_lift_blend
        if stage == STAGE_CODES["verify"]:
            return cfg.bc_verify_blend
        return 0.0

    def _curriculum_gate(self, stage: int) -> torch.Tensor:
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
        gate = torch.full((self.action_dim,), float(hand), device=self.torch_device)
        gate[: self.hand_start] = float(arm)
        return gate

    @torch.no_grad()
    def _bc_hand_control(self) -> torch.Tensor:
        normalized = self.bc_policy(self._observation())
        return self.hand_low.unsqueeze(0) + 0.5 * (normalized + 1.0) * (
            self.hand_high - self.hand_low
        ).unsqueeze(0)

    def _apply_residual(self, actions: torch.Tensor) -> torch.Tensor:
        actions = torch.clamp(actions, -1.0, 1.0)
        stage = self._stage_code()
        reference = self.reference_controls[self.step_index].unsqueeze(0).expand(
            self.num_envs, -1
        )
        target = reference.clone()

        blend = self._bc_blend(stage)
        if blend > 0.0:
            bc_hand = self._bc_hand_control()
            target[:, self.hand_start :] = (
                (1.0 - blend) * reference[:, self.hand_start :] + blend * bc_hand
            )

        gate = self._curriculum_gate(stage)
        target[:, self.controlled_positions] += (
            actions * self.residual_scale.unsqueeze(0) * gate.unsqueeze(0)
        )
        target = torch.maximum(torch.minimum(target, self.ctrl_high), self.ctrl_low)
        self.ctrl[:, self.actuator_ids] = target

        # Base _finalize_episode reconstructs controls from action_history. Store
        # the *effective* residual relative to the original reference so the
        # serialized trajectory includes both BC guidance and PPO corrections.
        effective = (
            target[:, self.controlled_positions] - reference[:, self.controlled_positions]
        ) / self.residual_scale.unsqueeze(0)
        self.action_history[self.step_index] = effective
        self._guided_steps += 1
        self._guided_action_sum += effective.abs().mean()
        self._guided_bc_sum += float(blend)
        return target

    def _reward(self, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        cfg = self.guided_config
        stage = self._stage_code()
        object_position = self.xpos[:, self.object_body_id]
        lift = object_position[:, 2] - self.initial_object_z
        lift_progress = torch.clamp(lift / cfg.success_lift_height, 0.0, 1.0)
        contact_signal = self._contact_signal()
        contact_progress = torch.clamp(
            contact_signal / float(cfg.minimum_contact_digits), 0.0, 1.0
        )

        if self.has_digit_contacts:
            thumb_contact = self.digit_flags[:, 4] > 0
            non_thumb_contact = self.digit_flags[:, :4].sum(dim=1) > 0
        else:
            thumb_contact = contact_signal >= 1.0
            non_thumb_contact = contact_signal >= 2.0
        opposition = thumb_contact & non_thumb_contact
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
            stable,
            self.success_streak + 1,
            torch.zeros_like(self.success_streak),
        )
        success_now = self.success_streak >= cfg.success_hold_steps
        new_success = success_now & ~self.success_reached
        self.success_reached |= success_now
        dropped = lift < -cfg.drop_margin

        close_or_hold = stage in {STAGE_CODES["close"], STAGE_CODES["hold"]}
        lift_or_verify = stage in {STAGE_CODES["lift"], STAGE_CODES["verify"]}
        grasp_scale = 1.0 if close_or_hold else (0.55 if lift_or_verify else 0.15)
        opposition_scale = 1.0 if close_or_hold else (0.80 if lift_or_verify else 0.15)

        # Lift receives essentially no credit without a real thumb/finger grasp.
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
            + (cfg.hold_opposition_bonus * opposition.float() if stage == STAGE_CODES["hold"] else 0.0)
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

    def training_metrics(self) -> dict[str, float]:
        result = super().training_metrics()
        if self._guided_steps:
            self._last_guided_metrics = {
                "bc_blend": float((self._guided_bc_sum / self._guided_steps).item()),
                "effective_residual_abs": float(
                    (self._guided_action_sum / self._guided_steps).item()
                ),
            }
            self._guided_steps = 0
            self._guided_bc_sum.zero_()
            self._guided_action_sum.zero_()
        return result | self._last_guided_metrics
