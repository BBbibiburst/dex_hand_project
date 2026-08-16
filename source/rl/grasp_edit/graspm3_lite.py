"""GraspM3-lite temporal trajectory search for the six-drive Dex Hand.

The original GraspM3 optimizes a low-dimensional temporal parameterization and
then filters candidates with dynamic simulation.  This implementation keeps
the same contract while respecting the underactuated hand:

* Wrist targets come from the reachable Wrist Lattice templates.
* Six actuator schedules independently control approach and closure timing.
* Small final actuator edits and a clench increment are optimized jointly.
* The existing IK-valid wrist trajectory is re-timed with smooth sigmoid
  profiles instead of optimizing every frame independently.
* MJWarp is used for batched screening; authoritative C MuJoCo replay is done
  by the single-object driver before a trajectory is accepted.

No ShadowHand joint trajectory is imported or retargeted.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import warp as wp

from source.rl.grasp_edit.env import GraspEditConfig, MjWarpGraspEditEnv
from source.rl.grasp_edit.primitives import (
    available_grasp_primitives,
    resolve_grasp_primitives,
)
from source.rl.grasp_edit.templates import GraspEditTemplate
from source.rl.residual.reference import STAGE_CODES
from source.rl.residual.trajectory import ResidualTrajectory

TEMPORAL_PARAMETER_DIM = 29


@dataclass(frozen=True)
class GraspM3LiteConfig(GraspEditConfig):
    """Search and trajectory parameters for one object's temporal optimizer."""

    population_size: int = 64
    iterations: int = 4
    elite_fraction: float = 0.20
    smoothing: float = 0.70
    verification_candidates: int = 8
    timing_center_min: float = 0.05
    timing_center_max: float = 0.95
    timing_width_min: float = 0.035
    timing_width_max: float = 0.45
    clench_gain_max: float = 0.40
    grasp_modes: tuple[str, ...] = available_grasp_primitives()
    mode_bias_scale: float = 1.0

    def validate(self) -> None:
        super().validate()
        if self.population_size <= 0 or self.num_envs != self.population_size:
            raise ValueError("num_envs must equal a positive population_size.")
        if self.iterations <= 0:
            raise ValueError("iterations must be positive.")
        if not 0.0 < self.elite_fraction <= 1.0:
            raise ValueError("elite_fraction must lie in (0, 1].")
        if not 0.0 <= self.smoothing < 1.0:
            raise ValueError("smoothing must lie in [0, 1).")
        if self.verification_candidates <= 0:
            raise ValueError("verification_candidates must be positive.")
        if not 0.0 < self.timing_center_min < self.timing_center_max <= 1.0:
            raise ValueError("timing center bounds must satisfy 0 < min < max <= 1.")
        if not 0.0 < self.timing_width_min < self.timing_width_max:
            raise ValueError("timing width bounds must satisfy 0 < min < max.")
        if self.clench_gain_max < 0.0:
            raise ValueError("clench_gain_max must be non-negative.")
        modes = resolve_grasp_primitives(self.grasp_modes)
        if self.verification_candidates < len(modes):
            raise ValueError(
                "verification_candidates must be at least the number of requested "
                f"grasp modes ({len(modes)})."
            )
        if self.population_size < len(modes) + 1:
            raise ValueError(
                "population_size must leave one reference schedule in addition to "
                f"the {len(modes)} requested grasp modes."
            )
        if not 0.0 <= self.mode_bias_scale <= 2.0:
            raise ValueError("mode_bias_scale must lie in [0, 2].")


@dataclass(frozen=True)
class TemporalBatch:
    """One CEM population transported into the batched MJWarp environment."""

    parameters: np.ndarray
    template_ids: np.ndarray
    reference_mask: np.ndarray
    mode_ids: np.ndarray | None = None

    def validate(
        self,
        *,
        population_size: int,
        template_count: int,
        mode_count: int = 1,
    ) -> None:
        parameters = np.asarray(self.parameters, dtype=np.float32)
        template_ids = np.asarray(self.template_ids, dtype=np.int64)
        reference_mask = np.asarray(self.reference_mask, dtype=bool)
        if parameters.shape != (population_size, TEMPORAL_PARAMETER_DIM):
            raise ValueError(
                "Temporal parameters must have shape "
                f"({population_size}, {TEMPORAL_PARAMETER_DIM}), got {parameters.shape}."
            )
        if template_ids.shape != (population_size,):
            raise ValueError("template_ids must match the population size.")
        if reference_mask.shape != (population_size,):
            raise ValueError("reference_mask must match the population size.")
        if self.mode_ids is not None:
            mode_ids = np.asarray(self.mode_ids, dtype=np.int64)
            if mode_ids.shape != (population_size,):
                raise ValueError("mode_ids must match the population size.")
            if np.any((mode_ids < 0) | (mode_ids >= mode_count)):
                raise ValueError("mode_ids contains an invalid grasp-mode index.")
        if np.any((template_ids < 0) | (template_ids >= template_count)):
            raise ValueError("template_ids contains an invalid Wrist Lattice index.")
        if not np.all(np.isfinite(parameters)):
            raise ValueError("Temporal parameters contain NaN or infinity.")

    @property
    def approach_centers(self) -> np.ndarray:
        return self.parameters[:, 0:6]

    @property
    def close_centers(self) -> np.ndarray:
        return self.parameters[:, 6:12]

    @property
    def widths(self) -> np.ndarray:
        return self.parameters[:, 12:18]

    @property
    def final_edits(self) -> np.ndarray:
        return self.parameters[:, 18:24]

    @property
    def wrist_approach(self) -> np.ndarray:
        return self.parameters[:, 24:26]

    @property
    def wrist_lift(self) -> np.ndarray:
        return self.parameters[:, 26:28]

    @property
    def clench_gain(self) -> np.ndarray:
        return self.parameters[:, 28]


@dataclass(frozen=True)
class TemporalEvaluation:
    rewards: np.ndarray
    max_lift: np.ndarray
    final_lift: np.ndarray
    tail_min_lift: np.ndarray
    success: np.ndarray
    controls: torch.Tensor


@dataclass(frozen=True)
class TemporalCandidate:
    trajectory: ResidualTrajectory
    score: float
    mjwarp_success: bool
    reference_schedule: bool
    mode_id: int
    mode_name: str


@dataclass(frozen=True)
class TemporalSearchResult:
    best_trajectory: ResidualTrajectory | None
    best_attempt: ResidualTrajectory | None
    verification_pool: tuple[TemporalCandidate, ...]
    history: tuple[dict[str, Any], ...]
    template_probabilities: tuple[float, ...]
    mode_probabilities: tuple[float, ...]


def normalized_sigmoid(progress: np.ndarray, center: float, width: float) -> np.ndarray:
    """Map [0, 1] to [0, 1] with a monotone, endpoint-normalized sigmoid."""
    progress = np.asarray(progress, dtype=np.float64)
    if width <= 0.0:
        raise ValueError("width must be positive.")
    raw = 1.0 / (1.0 + np.exp(-np.clip((progress - center) / width, -60.0, 60.0)))
    start = 1.0 / (1.0 + np.exp(np.clip(center / width, -60.0, 60.0)))
    end = 1.0 / (1.0 + np.exp(-np.clip((1.0 - center) / width, -60.0, 60.0)))
    return np.clip((raw - start) / max(end - start, 1e-8), 0.0, 1.0)


def mode_close_alpha(
    close_alpha_step: torch.Tensor,
    close_power: torch.Tensor,
) -> torch.Tensor:
    """Apply one scalar closure exponent to all six drives in each world."""
    if close_alpha_step.ndim != 2:
        raise ValueError("close_alpha_step must have shape (worlds, actuators).")
    if close_power.shape != (close_alpha_step.shape[0],):
        raise ValueError("close_power must provide one exponent per world.")
    return torch.pow(close_alpha_step, close_power.unsqueeze(1))


def _stage_progress(stages: np.ndarray, code: int) -> np.ndarray:
    result = np.zeros(len(stages), dtype=np.float32)
    indices = np.flatnonzero(np.asarray(stages) == code)
    if len(indices) > 1:
        result[indices] = np.linspace(0.0, 1.0, len(indices), dtype=np.float32)
    elif len(indices) == 1:
        result[indices] = 1.0
    return result


class MjWarpGraspM3LiteEnv(MjWarpGraspEditEnv):
    """Evaluate temporal actuator/wrist schedules in parallel with MJWarp."""

    parameter_dim = TEMPORAL_PARAMETER_DIM

    def __init__(
        self,
        object_id: str,
        templates: tuple[GraspEditTemplate, ...],
        config: GraspM3LiteConfig,
    ) -> None:
        super().__init__(object_id, templates, config)
        self.temporal_config = config
        self.modes = resolve_grasp_primitives(config.grasp_modes)
        self.mode_count = len(self.modes)
        self.mode_names = tuple(item.name for item in self.modes)
        self.reference_mode_id = (
            self.mode_names.index("wrap") if "wrap" in self.mode_names else 0
        )
        self.mode_approach_bias = torch.as_tensor(
            np.asarray([item.approach_bias for item in self.modes], dtype=np.float32),
            device=self.torch_device,
        )
        self.mode_final_bias = torch.as_tensor(
            np.asarray([item.final_bias for item in self.modes], dtype=np.float32),
            device=self.torch_device,
        )
        self.mode_close_power = torch.as_tensor(
            np.asarray([item.close_power for item in self.modes], dtype=np.float32),
            device=self.torch_device,
        )
        self.mode_score_weights = torch.as_tensor(
            np.asarray([item.score_weights for item in self.modes], dtype=np.float32),
            device=self.torch_device,
        )
        self.parameter_bounds_low, self.parameter_bounds_high = self._build_bounds(config)
        stages = np.asarray(self.references[0].stages, dtype=np.int16)
        self._approach_progress = torch.as_tensor(
            _stage_progress(stages, STAGE_CODES["approach"]),
            device=self.torch_device,
        )
        self._close_progress = torch.as_tensor(
            _stage_progress(stages, STAGE_CODES["close"]),
            device=self.torch_device,
        )
        self._hold_progress = torch.as_tensor(
            _stage_progress(stages, STAGE_CODES["hold"]),
            device=self.torch_device,
        )
        self._lift_progress = torch.as_tensor(
            _stage_progress(stages, STAGE_CODES["lift"]),
            device=self.torch_device,
        )
        self._stage_codes = stages
        self._arm_endpoints = self._build_arm_endpoints()

    def mode_summary(self) -> list[dict[str, Any]]:
        return [
            {
                "id": index,
                "name": mode.name,
                "family": mode.mode_family,
                "description": mode.description,
                "objective": mode.objective_name,
                "score_weights": list(mode.score_weights),
                "table_assisted": mode.table_assisted,
                "enclosure_bias": mode.enclosure_bias,
                "support_bias": mode.support_bias,
            }
            for index, mode in enumerate(self.modes)
        ]

    def _batch_mode_ids(self, batch: TemporalBatch) -> np.ndarray:
        if batch.mode_ids is None:
            return np.full(
                self.num_envs,
                self.reference_mode_id,
                dtype=np.int64,
            )
        return np.asarray(batch.mode_ids, dtype=np.int64)

    @staticmethod
    def _build_bounds(config: GraspM3LiteConfig) -> tuple[np.ndarray, np.ndarray]:
        low = np.empty(TEMPORAL_PARAMETER_DIM, dtype=np.float32)
        high = np.empty(TEMPORAL_PARAMETER_DIM, dtype=np.float32)
        low[0:12] = config.timing_center_min
        high[0:12] = config.timing_center_max
        low[12:18] = config.timing_width_min
        high[12:18] = config.timing_width_max
        low[18:24] = -1.0
        high[18:24] = 1.0
        low[24:28] = (config.timing_center_min, config.timing_width_min) * 2
        high[24:28] = (config.timing_center_max, config.timing_width_max) * 2
        low[28] = 0.0
        high[28] = config.clench_gain_max
        return low, high

    def _build_arm_endpoints(self) -> dict[int, tuple[torch.Tensor, torch.Tensor]]:
        endpoints: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        for name in ("approach", "lift"):
            code = STAGE_CODES[name]
            indices = np.flatnonzero(self._stage_codes == code)
            if not len(indices):
                continue
            endpoints[code] = (
                self.template_controls[:, int(indices[0]), : self.arm_action_size],
                self.template_controls[:, int(indices[-1]), : self.arm_action_size],
            )
        return endpoints

    def _normalised_sigmoid_torch(
        self,
        progress: torch.Tensor,
        centers: torch.Tensor,
        widths: torch.Tensor,
    ) -> torch.Tensor:
        x = progress[:, None, None]
        center = centers[None, :, :]
        width = widths[None, :, :]
        raw = torch.sigmoid((x - center) / width)
        start = torch.sigmoid(-center / width)
        end = torch.sigmoid((1.0 - center) / width)
        return torch.clamp((raw - start) / torch.clamp(end - start, min=1e-6), 0.0, 1.0)

    def _normalised_scalar_torch(
        self,
        progress: torch.Tensor,
        centers: torch.Tensor,
        widths: torch.Tensor,
    ) -> torch.Tensor:
        x = progress[:, None]
        center = centers[None, :]
        width = widths[None, :]
        raw = torch.sigmoid((x - center) / width)
        start = torch.sigmoid(-center / width)
        end = torch.sigmoid((1.0 - center) / width)
        return torch.clamp((raw - start) / torch.clamp(end - start, min=1e-6), 0.0, 1.0)

    def _apply_arm_retiming(
        self,
        target: torch.Tensor,
        template_ids: torch.Tensor,
        stage: int,
        progress_value: torch.Tensor,
        center: torch.Tensor,
        width: torch.Tensor,
    ) -> None:
        endpoints = self._arm_endpoints.get(stage)
        if endpoints is None:
            return
        start, end = endpoints
        alpha = self._normalised_scalar_torch(
            progress_value.reshape(1), center, width
        )[0]
        target[:, : self.arm_action_size] = start[template_ids] + alpha[:, None] * (
            end[template_ids] - start[template_ids]
        )

    def _trajectory_from_world(
        self,
        world: int,
        batch: TemporalBatch,
        controls: torch.Tensor,
        reward: torch.Tensor,
        max_lift: torch.Tensor,
        final_lift: torch.Tensor,
        tail_min_lift: torch.Tensor,
        success: torch.Tensor,
    ) -> ResidualTrajectory:
        template_id = int(batch.template_ids[world])
        mode_ids = self._batch_mode_ids(batch)
        mode_id = int(mode_ids[world])
        mode = self.modes[mode_id]
        reference = self.references[template_id]
        template = self.templates[template_id]
        parameters = batch.parameters[world].astype(np.float32)
        repeated = np.repeat(parameters[None, :], self.horizon, axis=0)
        return ResidualTrajectory(
            object_id=self.object_id,
            source_manifest=str(template.manifest),
            start_stage="approach",
            action_mode="graspm3_lite_temporal",
            residual_actions=repeated,
            controls=controls[:, world].detach().cpu().numpy().astype(np.float32),
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
                "temporal_parameter_dim": TEMPORAL_PARAMETER_DIM,
                "temporal_parameter_names": [
                    "approach_centers[6]",
                    "close_centers[6]",
                    "widths[6]",
                    "final_edits[6]",
                    "wrist_approach_center_width[2]",
                    "wrist_lift_center_width[2]",
                    "clench_gain",
                ],
                "temporal_parameters": parameters.tolist(),
                "reference_schedule": bool(batch.reference_mask[world]),
                "grasp_mode": mode.name,
                "grasp_mode_id": mode_id,
                "mode_family": mode.mode_family,
                "mode_description": mode.description,
                "mode_objective": mode.objective_name,
                "mode_score_weights": list(mode.score_weights),
                "table_assisted": bool(mode.table_assisted),
                "mode_enclosure_bias": float(mode.enclosure_bias),
                "mode_support_bias": float(mode.support_bias),
                "mjwarp_success": bool(success[world].item()),
                "mjwarp_max_lift": float(max_lift[world].item()),
                "mjwarp_final_lift": float(final_lift[world].item()),
                "mjwarp_tail_min_lift": float(tail_min_lift[world].item()),
                "graspm3_lite": True,
            },
        )

    def evaluate(self, batch: TemporalBatch) -> TemporalEvaluation:
        batch.validate(
            population_size=self.num_envs,
            template_count=self.template_count,
            mode_count=self.mode_count,
        )
        mode_ids_np = self._batch_mode_ids(batch)
        template_ids = torch.as_tensor(
            batch.template_ids, device=self.torch_device, dtype=torch.long
        )
        mode_ids = torch.as_tensor(
            mode_ids_np, device=self.torch_device, dtype=torch.long
        )
        parameters = torch.as_tensor(
            batch.parameters, device=self.torch_device, dtype=torch.float32
        )
        mode_approach_bias = self.mode_approach_bias[mode_ids]
        mode_final_bias = self.mode_final_bias[mode_ids]
        candidate = torch.clamp(
            self.template_candidate_fractions[template_ids]
            + self.temporal_config.mode_bias_scale * mode_approach_bias,
            0.0,
            1.0,
        )
        final = torch.clamp(
            self.template_grip_fractions[template_ids]
            + self.config.hand_edit_fraction * parameters[:, 18:24],
            0.0,
            1.0,
        )
        final = torch.clamp(
            final + self.temporal_config.mode_bias_scale * mode_final_bias,
            0.0,
            1.0,
        )
        clench = torch.clamp(
            final
            + parameters[:, 28:29]
            * (final - self.open_fractions.unsqueeze(0)),
            0.0,
            1.0,
        )
        approach_alpha = self._normalised_sigmoid_torch(
            self._approach_progress,
            parameters[:, 0:6],
            parameters[:, 12:18],
        )
        close_alpha = self._normalised_sigmoid_torch(
            self._close_progress,
            parameters[:, 6:12],
            parameters[:, 12:18],
        )
        reference_controls = self.template_controls[template_ids]
        reference_mask = torch.as_tensor(
            batch.reference_mask, device=self.torch_device, dtype=torch.bool
        )
        self._initialise_selected_worlds(template_ids)
        initial_z = self.template_object_z[template_ids]
        max_lift = torch.full((self.num_envs,), -float("inf"), device=self.torch_device)
        tail_min_lift = torch.full((self.num_envs,), float("inf"), device=self.torch_device)
        tail_lift_sum = torch.zeros((self.num_envs,), device=self.torch_device)
        tail_samples = 0
        controls_history = torch.empty(
            (self.horizon, self.num_envs, self.references[0].action_dim),
            device=self.torch_device,
            dtype=torch.float32,
        )
        for step_index, stage in enumerate(self._stage_codes):
            target = reference_controls[:, step_index].clone()
            if stage == STAGE_CODES["approach"]:
                fractions = self.open_fractions.unsqueeze(0) + approach_alpha[step_index] * (
                    candidate - self.open_fractions.unsqueeze(0)
                )
                target[:, self.hand_slice] = self.hand_low.unsqueeze(0) + fractions * (
                    self.hand_high - self.hand_low
                ).unsqueeze(0)
                self._apply_arm_retiming(
                    target,
                    template_ids,
                    int(stage),
                    self._approach_progress[step_index],
                    parameters[:, 24],
                    parameters[:, 25],
                )
            elif stage == STAGE_CODES["close"]:
                close_alpha_mode = mode_close_alpha(
                    close_alpha[step_index], self.mode_close_power[mode_ids]
                )
                fractions = candidate + close_alpha_mode * (final - candidate)
                target[:, self.hand_slice] = self.hand_low.unsqueeze(0) + fractions * (
                    self.hand_high - self.hand_low
                ).unsqueeze(0)
            elif stage == STAGE_CODES["hold"]:
                fractions = final + self._hold_progress[step_index] * (clench - final)
                target[:, self.hand_slice] = self.hand_low.unsqueeze(0) + fractions * (
                    self.hand_high - self.hand_low
                ).unsqueeze(0)
            elif stage in (STAGE_CODES["lift"], STAGE_CODES["verify"]):
                target[:, self.hand_slice] = self.hand_low.unsqueeze(0) + clench * (
                    self.hand_high - self.hand_low
                ).unsqueeze(0)
                if stage == STAGE_CODES["lift"]:
                    self._apply_arm_retiming(
                        target,
                        template_ids,
                        int(stage),
                        self._lift_progress[step_index],
                        parameters[:, 26],
                        parameters[:, 27],
                    )
            target[reference_mask] = reference_controls[reference_mask, step_index]
            target = torch.maximum(torch.minimum(target, self.ctrl_high), self.ctrl_low)
            controls_history[step_index] = target
            self.ctrl[:, self.actuator_ids] = target
            self._sync_torch_before_warp()
            for _ in range(self.physics_steps_per_control):
                # The graph is captured for one step and reused for every
                # candidate/frame, matching the existing MJWarp editor.
                wp.capture_launch(self.step_graph)
            self._sync_warp_before_torch()
            lift = self.xpos[:, self.object_body_id, 2] - initial_z
            max_lift = torch.maximum(max_lift, lift)
            if step_index >= self.horizon - self.config.success_tail_steps:
                tail_min_lift = torch.minimum(tail_min_lift, lift)
                tail_lift_sum += lift
                tail_samples += 1

        final_lift = self.xpos[:, self.object_body_id, 2] - initial_z
        tail_mean_lift = tail_lift_sum / float(max(tail_samples, 1))
        object_velocity = self.qvel[:, self.object_qvel_adr : self.object_qvel_adr + 6]
        object_speed = torch.linalg.vector_norm(object_velocity, dim=1)
        success = (tail_min_lift >= self.config.success_lift_height) & (
            object_speed <= self.config.maximum_object_speed
        )
        max_progress = torch.clamp(max_lift / self.config.success_lift_height, 0.0, 1.0)
        final_progress = torch.clamp(final_lift / self.config.success_lift_height, 0.0, 1.0)
        tail_progress = torch.clamp(tail_mean_lift / self.config.success_lift_height, 0.0, 1.0)
        tail_min_progress = torch.clamp(
            tail_min_lift / self.config.success_lift_height, 0.0, 1.0
        )
        edit_cost = parameters[:, 18:24].square().mean(dim=1)
        timing_cost = (
            (parameters[:, 0:12] - 0.55).square().mean(dim=1)
            + (parameters[:, 12:18] - 0.18).square().mean(dim=1)
        )
        edit_cost = torch.where(reference_mask, 0.0, edit_cost)
        timing_cost = torch.where(reference_mask, 0.0, timing_cost)
        score_weights = self.mode_score_weights[mode_ids]
        reward = (
            score_weights[:, 0] * max_progress
            + score_weights[:, 1] * final_progress
            + score_weights[:, 2] * tail_progress
            + score_weights[:, 3] * tail_min_progress
            + 15.0 * success.float()
            - 0.03 * edit_cost
            - 0.01 * timing_cost
        )
        return TemporalEvaluation(
            rewards=reward.detach().cpu().numpy(),
            max_lift=max_lift.detach().cpu().numpy(),
            final_lift=final_lift.detach().cpu().numpy(),
            tail_min_lift=tail_min_lift.detach().cpu().numpy(),
            success=success.detach().cpu().numpy().astype(bool),
            controls=controls_history.detach(),
        )

    def trajectory_from_world(
        self,
        world: int,
        batch: TemporalBatch,
        evaluation: TemporalEvaluation,
    ) -> ResidualTrajectory:
        reward = torch.as_tensor(evaluation.rewards, device=self.torch_device)
        max_lift = torch.as_tensor(evaluation.max_lift, device=self.torch_device)
        final_lift = torch.as_tensor(evaluation.final_lift, device=self.torch_device)
        tail_min_lift = torch.as_tensor(evaluation.tail_min_lift, device=self.torch_device)
        success = torch.as_tensor(evaluation.success, device=self.torch_device)
        return self._trajectory_from_world(
            world,
            batch,
            evaluation.controls,
            reward,
            max_lift,
            final_lift,
            tail_min_lift,
            success,
        )


class TemporalCEMSearch:
    """Cross-entropy search over the compact GraspM3-lite parameterization."""

    def __init__(
        self,
        env: MjWarpGraspM3LiteEnv,
        config: GraspM3LiteConfig,
        *,
        seed: int = 0,
    ) -> None:
        self.env = env
        self.config = config
        self.rng = np.random.default_rng(seed)
        self.mean = np.asarray(
            [
                *([0.55] * 12),
                *([0.18] * 6),
                *([0.0] * 6),
                0.55,
                0.18,
                0.55,
                0.18,
                0.10,
            ],
            dtype=np.float32,
        )
        self.std = np.asarray(
            [
                *([0.20] * 12),
                *([0.10] * 6),
                *([0.35] * 6),
                0.18,
                0.10,
                0.18,
                0.10,
                0.10,
            ],
            dtype=np.float32,
        )
        self.low = env.parameter_bounds_low
        self.high = env.parameter_bounds_high
        self.template_probabilities = np.full(
            len(env.templates), 1.0 / len(env.templates), dtype=np.float64
        )
        self.mode_probabilities = np.full(
            env.mode_count, 1.0 / env.mode_count, dtype=np.float64
        )

    def _sample(self) -> TemporalBatch:
        parameters = self.rng.normal(
            self.mean,
            np.maximum(self.std, 1e-4),
            size=(self.config.population_size, TEMPORAL_PARAMETER_DIM),
        ).astype(np.float32)
        parameters = np.clip(parameters, self.low, self.high)
        template_ids = self.rng.choice(
            len(self.env.templates),
            size=self.config.population_size,
            p=self.template_probabilities,
        ).astype(np.int64)
        mode_ids = self.rng.choice(
            self.env.mode_count,
            size=self.config.population_size,
            p=self.mode_probabilities,
        ).astype(np.int64)
        # Stratify the categorical seed so every requested grasp family is
        # evaluated in every CEM iteration.  This is the cheap analogue of
        # GraspM3's many directional demonstrations and avoids early collapse
        # onto a thumb/finger pinch.
        mode_ids[: self.env.mode_count] = np.arange(self.env.mode_count, dtype=np.int64)
        reference_mask = np.zeros(self.config.population_size, dtype=bool)
        reference_count = min(
            len(self.env.templates),
            max(1, self.config.population_size // 8),
            self.config.population_size - self.env.mode_count,
        )
        if reference_count > 0:
            start = self.config.population_size - reference_count
            template_ids[start:] = np.arange(reference_count, dtype=np.int64)
            mode_ids[start:] = self.env.reference_mode_id
            reference_mask[start:] = True
        return TemporalBatch(parameters, template_ids, reference_mask, mode_ids)

    def _update(self, batch: TemporalBatch, evaluation: TemporalEvaluation) -> np.ndarray:
        eligible = np.flatnonzero(~batch.reference_mask)
        if not len(eligible):
            eligible = np.arange(len(batch.parameters))
        elite_count = max(1, round(self.config.elite_fraction * len(eligible)))
        ordered = eligible[np.argsort(evaluation.rewards[eligible])]
        elite = ordered[-elite_count:]
        elite_parameters = batch.parameters[elite]
        blend = 1.0 - self.config.smoothing
        self.mean = np.clip(
            self.config.smoothing * self.mean + blend * elite_parameters.mean(axis=0),
            self.low,
            self.high,
        )
        self.std = np.maximum(
            0.02,
            self.config.smoothing * self.std + blend * elite_parameters.std(axis=0),
        )
        counts = np.bincount(
            batch.template_ids[elite], minlength=len(self.env.templates)
        ).astype(np.float64)
        counts += 0.25
        updated = counts / counts.sum()
        self.template_probabilities = (
            self.config.smoothing * self.template_probabilities + blend * updated
        )
        self.template_probabilities /= self.template_probabilities.sum()
        mode_ids = self.env._batch_mode_ids(batch)
        mode_counts = np.bincount(mode_ids[elite], minlength=self.env.mode_count).astype(
            np.float64
        )
        mode_counts += 0.25
        mode_updated = mode_counts / mode_counts.sum()
        self.mode_probabilities = (
            self.config.smoothing * self.mode_probabilities
            + blend * mode_updated
        )
        self.mode_probabilities /= self.mode_probabilities.sum()
        return elite

    @staticmethod
    def _candidate_rank(candidate: TemporalCandidate) -> tuple[bool, bool, float]:
        return (
            candidate.mjwarp_success,
            not candidate.reference_schedule,
            candidate.score,
        )

    @staticmethod
    def _candidate_key(candidate: TemporalCandidate) -> tuple[Any, ...]:
        metadata = candidate.trajectory.metadata
        parameters = np.asarray(metadata.get("temporal_parameters", []), dtype=np.float32)
        return (
            candidate.mode_id,
            metadata.get("template_id"),
            bool(candidate.reference_schedule),
            b"" if candidate.reference_schedule else parameters.round(5).tobytes(),
        )

    def _select_verification_pool(
        self,
        pool: list[TemporalCandidate],
    ) -> tuple[TemporalCandidate, ...]:
        ranked = sorted(pool, key=self._candidate_rank, reverse=True)
        deduplicated: list[TemporalCandidate] = []
        seen: set[tuple[Any, ...]] = set()
        for candidate in ranked:
            key = self._candidate_key(candidate)
            if key not in seen:
                seen.add(key)
                deduplicated.append(candidate)

        selected: list[TemporalCandidate] = []
        # One non-reference candidate per requested mode is the minimum useful
        # verification set.  Fill remaining slots globally by physical rank.
        for mode_id in range(self.env.mode_count):
            mode_candidates = [
                candidate
                for candidate in deduplicated
                if candidate.mode_id == mode_id and not candidate.reference_schedule
            ]
            if mode_candidates:
                selected.append(max(mode_candidates, key=self._candidate_rank))
        selected_keys = {self._candidate_key(candidate) for candidate in selected}
        for candidate in deduplicated:
            key = self._candidate_key(candidate)
            if key not in selected_keys:
                selected.append(candidate)
                selected_keys.add(key)
            if len(selected) >= self.config.verification_candidates:
                break
        selected = sorted(selected, key=self._candidate_rank, reverse=True)[
            : self.config.verification_candidates
        ]
        return tuple(selected)

    def run(
        self,
        callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> TemporalSearchResult:
        history: list[dict[str, Any]] = []
        pool: list[TemporalCandidate] = []
        best_temporal: TemporalCandidate | None = None
        best_any: TemporalCandidate | None = None
        for iteration in range(self.config.iterations):
            batch = self._sample()
            batch_mode_ids = self.env._batch_mode_ids(batch)
            evaluation = self.env.evaluate(batch)
            elite = self._update(batch, evaluation)
            temporal_indices = np.flatnonzero(~batch.reference_mask)
            candidate_indices = np.concatenate(
                [temporal_indices, np.flatnonzero(batch.reference_mask)]
            )
            candidate_indices = candidate_indices[
                np.argsort(evaluation.rewards[candidate_indices])[::-1]
            ]
            selected_indices: list[int] = []
            # Preserve the best non-reference trajectory for every grasp mode,
            # even if another family currently dominates the raw reward.
            for mode_id in range(self.env.mode_count):
                mode_indices = temporal_indices[batch_mode_ids[temporal_indices] == mode_id]
                if len(mode_indices):
                    success_order = np.lexsort(
                        (
                            evaluation.rewards[mode_indices],
                            evaluation.success[mode_indices].astype(np.int8),
                        )
                    )
                    selected_indices.append(int(mode_indices[success_order[-1]]))
            for index in candidate_indices:
                if int(index) not in selected_indices:
                    selected_indices.append(int(index))
                if len(selected_indices) >= (
                    self.config.verification_candidates + self.env.mode_count
                ):
                    break

            for index in selected_indices:
                candidate = TemporalCandidate(
                    trajectory=self.env.trajectory_from_world(index, batch, evaluation),
                    score=float(evaluation.rewards[index]),
                    mjwarp_success=bool(evaluation.success[index]),
                    reference_schedule=bool(batch.reference_mask[index]),
                    mode_id=int(batch_mode_ids[index]),
                    mode_name=self.env.mode_names[int(batch_mode_ids[index])],
                )
                pool.append(candidate)
                if best_any is None or self._candidate_rank(
                    candidate
                ) > self._candidate_rank(best_any):
                    best_any = candidate
                if (
                    not candidate.reference_schedule
                    and candidate.mjwarp_success
                    and (best_temporal is None or candidate.score > best_temporal.score)
                ):
                    best_temporal = candidate
            row = {
                "iteration": iteration + 1,
                "population": self.config.population_size,
                "elite": len(elite),
                "mjwarp_successes": int(evaluation.success.sum()),
                "best_reward": float(np.max(evaluation.rewards)),
                "best_max_lift": float(np.max(evaluation.max_lift)),
                "best_final_lift": float(np.max(evaluation.final_lift)),
                "best_tail_min_lift": float(np.max(evaluation.tail_min_lift)),
                "mean_reward": float(np.mean(evaluation.rewards)),
                "template_probabilities": self.template_probabilities.tolist(),
                "mode_probabilities": self.mode_probabilities.tolist(),
                "mode_mjwarp_successes": {
                    self.env.mode_names[mode_id]: int(
                        evaluation.success[batch_mode_ids == mode_id].sum()
                    )
                    for mode_id in range(self.env.mode_count)
                },
            }
            history.append(row)
            if callback is not None:
                callback(dict(row))
        selected = self._select_verification_pool(pool)
        return TemporalSearchResult(
            best_trajectory=None if best_temporal is None else best_temporal.trajectory,
            best_attempt=None if best_any is None else best_any.trajectory,
            verification_pool=selected,
            history=tuple(history),
            template_probabilities=tuple(float(x) for x in self.template_probabilities),
            mode_probabilities=tuple(float(x) for x in self.mode_probabilities),
        )
