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

import mujoco
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
from source.ultradexgrasp.catalog import load_object_geometry
from source.ultradexgrasp.contracts import DemonstrationEpisode

TEMPORAL_PARAMETER_DIM = 30


@dataclass(frozen=True)
class ObjectShapeProfile:
    """Scale-aware coarse geometry used to allocate grasp-search effort."""

    extents: tuple[float, float, float]
    sorted_extents: tuple[float, float, float]
    flatness_ratio: float
    elongation_ratio: float
    family: str

    @property
    def is_flat(self) -> bool:
        return self.family == "flat"


def classify_object_shape(extents: np.ndarray | tuple[float, float, float]) -> ObjectShapeProfile:
    """Classify collision extents without relying on an object-name taxonomy."""
    values = np.asarray(extents, dtype=np.float64)
    if values.shape != (3,) or not np.all(np.isfinite(values)) or np.any(values <= 0.0):
        raise ValueError("Object extents must contain three finite positive values.")
    longest, middle, thickness = np.sort(values)[::-1]
    flatness = float(middle / max(thickness, 1e-9))
    elongation = float(longest / max(middle, 1e-9))
    if flatness >= 1.8 and thickness <= 0.045:
        family = "flat"
    elif elongation >= 2.2:
        family = "elongated"
    else:
        family = "compact"
    return ObjectShapeProfile(
        extents=tuple(float(value) for value in values),
        sorted_extents=(float(longest), float(middle), float(thickness)),
        flatness_ratio=flatness,
        elongation_ratio=elongation,
        family=family,
    )


def geometry_mode_probabilities(
    mode_names: tuple[str, ...],
    profile: ObjectShapeProfile,
) -> np.ndarray:
    """Return a normalized categorical prior while retaining every requested mode."""
    if not mode_names:
        raise ValueError("At least one grasp mode is required.")
    weights = np.ones(len(mode_names), dtype=np.float64)
    if profile.family == "flat":
        preferred = {
            "lateral": 6.0,
            "table_assisted": 4.5,
            "wrap": 3.0,
            "tripod": 1.75,
            "pinch": 1.50,
            "cradle": 1.0,
            "hook": 0.50,
            "spherical": 0.25,
        }
        weights = np.asarray([preferred.get(name, 1.0) for name in mode_names], dtype=np.float64)
    elif profile.family == "elongated":
        preferred = {
            "wrap": 4.0,
            "hook": 3.0,
            "cradle": 2.5,
            "table_assisted": 2.0,
            "lateral": 1.5,
        }
        weights = np.asarray([preferred.get(name, 1.0) for name in mode_names], dtype=np.float64)
    weights = np.maximum(weights, 1e-6)
    return weights / weights.sum()


def template_tracking_probabilities(
    templates: tuple[GraspEditTemplate, ...],
) -> np.ndarray:
    """Prefer dynamically accurate references, not just static IK prechecks."""
    if not templates:
        raise ValueError("At least one Wrist Lattice template is required.")
    weights: list[float] = []
    for template in templates:
        try:
            episode = DemonstrationEpisode.load(template.manifest)
            position_error = float(episode.metadata.get("approach_position_error", 0.025))
            orientation_error = float(episode.metadata.get("approach_orientation_error", 0.22))
        except (OSError, KeyError, TypeError, ValueError):
            position_error = 0.025
            orientation_error = 0.22
        tracking = np.exp(
            -position_error / 0.012 - orientation_error / 0.15
        )
        edit_norm = np.linalg.norm(template.translation_offset) / 0.01 + np.linalg.norm(
            template.rotation_offset_degrees
        ) / 15.0
        weight = float(max(tracking, 0.02) * np.exp(-0.08 * edit_norm))
        if template.success:
            weight *= 3.0
        weights.append(weight)
    probabilities = np.maximum(np.asarray(weights, dtype=np.float64), 1e-6)
    return probabilities / probabilities.sum()


@wp.kernel
def _collect_graspm3_contacts(
    dimensions: wp.array(dtype=wp.int32),
    geoms: wp.array(dtype=wp.vec2i),
    world_ids: wp.array(dtype=wp.int32),
    object_mask: wp.array(dtype=wp.int32),
    robot_mask: wp.array(dtype=wp.int32),
    palm_mask: wp.array(dtype=wp.int32),
    digit_lookup: wp.array(dtype=wp.int32),
    contact_counts: wp.array(dtype=wp.int32),
    digit_flags: wp.array(dtype=wp.int32),
    palm_flags: wp.array(dtype=wp.int32),
):
    contact_index = wp.tid()
    if dimensions[contact_index] <= 0:
        return
    pair = geoms[contact_index]
    geom0 = pair[0]
    geom1 = pair[1]
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
    if palm_mask[robot_geom] != 0:
        wp.atomic_max(palm_flags, world, 1)


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
    ingress_gain_max: float = 0.25
    maximum_object_speed: float = 0.10
    maximum_object_angular_speed: float = 0.10
    success_tail_steps: int = 20
    minimum_tail_contact_fraction: float = 0.70
    minimum_tail_grasp_fraction: float = 0.60
    minimum_flat_thumb_fraction: float = 0.55
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
        if not 0.0 <= self.ingress_gain_max <= 0.5:
            raise ValueError("ingress_gain_max must lie in [0, 0.5].")
        if self.maximum_object_speed <= 0.0:
            raise ValueError("maximum_object_speed must be positive.")
        if self.maximum_object_angular_speed <= 0.0:
            raise ValueError("maximum_object_angular_speed must be positive.")
        for name, value in (
            ("minimum_tail_contact_fraction", self.minimum_tail_contact_fraction),
            ("minimum_tail_grasp_fraction", self.minimum_tail_grasp_fraction),
            ("minimum_flat_thumb_fraction", self.minimum_flat_thumb_fraction),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must lie in [0, 1].")
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

    @property
    def ingress_gain(self) -> np.ndarray:
        return self.parameters[:, 29]


@dataclass(frozen=True)
class TemporalEvaluation:
    rewards: np.ndarray
    max_lift: np.ndarray
    final_lift: np.ndarray
    tail_min_lift: np.ndarray
    tail_max_speed: np.ndarray
    tail_max_angular_speed: np.ndarray
    tail_contact_fraction: np.ndarray
    tail_grasp_fraction: np.ndarray
    tail_thumb_fraction: np.ndarray
    tail_mean_contact_digits: np.ndarray
    success: np.ndarray
    controls: torch.Tensor


@dataclass(frozen=True)
class TemporalWarmStart:
    """One previously evaluated temporal solution injected into every CEM run."""

    parameters: np.ndarray
    template_id: int
    mode_id: int

    def validate(self, *, template_count: int, mode_count: int) -> None:
        parameters = np.asarray(self.parameters, dtype=np.float32)
        if parameters.shape != (TEMPORAL_PARAMETER_DIM,) or not np.all(
            np.isfinite(parameters)
        ):
            raise ValueError(
                f"Warm-start parameters must have shape ({TEMPORAL_PARAMETER_DIM},)."
            )
        if not 0 <= self.template_id < template_count:
            raise ValueError("Warm-start template_id is out of range.")
        if not 0 <= self.mode_id < mode_count:
            raise ValueError("Warm-start mode_id is out of range.")


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
    """Apply scalar or per-actuator closure exponents in each world."""
    if close_alpha_step.ndim != 2:
        raise ValueError("close_alpha_step must have shape (worlds, actuators).")
    if close_power.shape == (close_alpha_step.shape[0],):
        close_power = close_power.unsqueeze(1)
    elif close_power.shape != close_alpha_step.shape:
        raise ValueError(
            "close_power must provide one exponent per world or per world/actuator."
        )
    return torch.pow(close_alpha_step, close_power)


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
            np.asarray(
                [
                    item.close_power_by_actuator
                    if item.close_power_by_actuator is not None
                    else (item.close_power,) * 6
                    for item in self.modes
                ],
                dtype=np.float32,
            ),
            device=self.torch_device,
        )
        self.mode_ingress_scale = torch.as_tensor(
            np.asarray([item.ingress_scale for item in self.modes], dtype=np.float32),
            device=self.torch_device,
        )
        self.mode_score_weights = torch.as_tensor(
            np.asarray([item.score_weights for item in self.modes], dtype=np.float32),
            device=self.torch_device,
        )
        try:
            geometry = load_object_geometry(object_id, surface_points=256)
            self.shape_profile = classify_object_shape(np.ptp(geometry.bounds, axis=0))
        except (OSError, ValueError):
            self.shape_profile = classify_object_shape((1.0, 1.0, 1.0))
        self.mode_prior_probabilities = geometry_mode_probabilities(
            self.mode_names, self.shape_profile
        )
        self.template_prior_probabilities = template_tracking_probabilities(templates)
        self._prepare_contact_lookup()
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
                "close_power_by_actuator": list(
                    mode.close_power_by_actuator
                    if mode.close_power_by_actuator is not None
                    else (mode.close_power,) * 6
                ),
                "ingress_scale": mode.ingress_scale,
            }
            for index, mode in enumerate(self.modes)
        ]

    def shape_summary(self) -> dict[str, Any]:
        return {
            "family": self.shape_profile.family,
            "extents": list(self.shape_profile.extents),
            "sorted_extents": list(self.shape_profile.sorted_extents),
            "flatness_ratio": self.shape_profile.flatness_ratio,
            "elongation_ratio": self.shape_profile.elongation_ratio,
            "mode_prior_probabilities": self.mode_prior_probabilities.tolist(),
            "template_prior_probabilities": self.template_prior_probabilities.tolist(),
        }

    def _batch_mode_ids(self, batch: TemporalBatch) -> np.ndarray:
        if batch.mode_ids is None:
            return np.full(
                self.num_envs,
                self.reference_mode_id,
                dtype=np.int64,
            )
        return np.asarray(batch.mode_ids, dtype=np.int64)

    def _prepare_contact_lookup(self) -> None:
        bindings = self.host_env.task._require_bindings()
        object_mask = np.zeros(self.model.ngeom, dtype=np.int32)
        robot_mask = np.zeros(self.model.ngeom, dtype=np.int32)
        palm_mask = np.zeros(self.model.ngeom, dtype=np.int32)
        digit_lookup = np.full(self.model.ngeom, -1, dtype=np.int32)

        for geom_id in bindings.objects["object"].geom_ids:
            object_mask[int(geom_id)] = 1

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
            if not body_ids:
                return -1
            chains = [ancestors(body_id) for body_id in body_ids]
            common = set(chains[0])
            for chain in chains[1:]:
                common.intersection_update(chain)
            return next((body_id for body_id in chains[0] if body_id in common), -1)

        hand_prefix = str(
            getattr(self.host_env.controller.hand_controller, "hand_prefix", "") or ""
        )

        def resolve_hand_geom(local_name: str) -> int:
            names = [f"{hand_prefix}{local_name}"] if hand_prefix else []
            names.append(local_name)
            for name in names:
                geom_id = mujoco.mj_name2id(
                    self.model, mujoco.mjtObj.mjOBJ_GEOM, name
                )
                if geom_id >= 0:
                    return int(geom_id)
            matches = [
                geom_id
                for geom_id in range(self.model.ngeom)
                if (
                    mujoco.mj_id2name(
                        self.model, mujoco.mjtObj.mjOBJ_GEOM, geom_id
                    )
                    or ""
                ).endswith(local_name)
            ]
            return int(matches[0]) if len(matches) == 1 else -1

        digit_roots: list[int] = []
        for digit in range(5):
            anchors: list[int] = []
            for part in range(3):
                geom_id = resolve_hand_geom(f"skin_{digit}_{part}_p")
                if geom_id >= 0:
                    anchors.append(int(self.model.geom_bodyid[geom_id]))
            digit_roots.append(deepest_common_ancestor(anchors))

        for raw_geom_id in bindings.robot_geom_ids:
            geom_id = int(raw_geom_id)
            robot_mask[geom_id] = 1
            geom_name = (
                mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
                or ""
            )
            body_id = int(self.model.geom_bodyid[geom_id])
            body_name = (
                mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, body_id)
                or ""
            )
            names = f"{geom_name} {body_name}".lower()
            if "palm" in names:
                palm_mask[geom_id] = 1

            direct = next(
                (digit for digit in range(5) if f"skin_{digit}_" in geom_name),
                -1,
            )
            if direct >= 0:
                digit_lookup[geom_id] = direct
                continue
            chain = set(ancestors(body_id))
            candidates = [
                digit
                for digit, root in enumerate(digit_roots)
                if root > 0 and root in chain
            ]
            if len(candidates) == 1:
                digit_lookup[geom_id] = candidates[0]

        self.has_digit_contacts = bool(np.any(digit_lookup >= 0))
        self.object_mask = wp.from_numpy(
            object_mask, dtype=wp.int32, device=self.wp_device
        )
        self.robot_mask = wp.from_numpy(
            robot_mask, dtype=wp.int32, device=self.wp_device
        )
        self.palm_mask = wp.from_numpy(palm_mask, dtype=wp.int32, device=self.wp_device)
        self.digit_lookup = wp.from_numpy(
            digit_lookup, dtype=wp.int32, device=self.wp_device
        )
        self.contact_counts_wp = wp.zeros(
            self.num_envs, dtype=wp.int32, device=self.wp_device
        )
        self.digit_flags_wp = wp.zeros(
            self.num_envs * 5, dtype=wp.int32, device=self.wp_device
        )
        self.palm_flags_wp = wp.zeros(
            self.num_envs, dtype=wp.int32, device=self.wp_device
        )
        self.contact_counts = wp.to_torch(self.contact_counts_wp, requires_grad=False)
        self.digit_flags = wp.to_torch(
            self.digit_flags_wp, requires_grad=False
        ).view(self.num_envs, 5)
        self.palm_flags = wp.to_torch(self.palm_flags_wp, requires_grad=False)

    def _update_contacts(self) -> None:
        self.contact_counts_wp.zero_()
        self.digit_flags_wp.zero_()
        self.palm_flags_wp.zero_()
        wp.launch(
            _collect_graspm3_contacts,
            dim=int(self.data.contact.dim.shape[0]),
            inputs=[
                self.data.contact.dim,
                self.data.contact.geom,
                self.data.contact.worldid,
                self.object_mask,
                self.robot_mask,
                self.palm_mask,
                self.digit_lookup,
                self.contact_counts_wp,
                self.digit_flags_wp,
                self.palm_flags_wp,
            ],
            device=self.wp_device,
        )
        self._sync_warp_before_torch()

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
        low[29] = 0.0
        high[29] = config.ingress_gain_max
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

    def _apply_ingress(
        self,
        target: torch.Tensor,
        template_ids: torch.Tensor,
        mode_ids: torch.Tensor,
        ingress_gain: torch.Tensor,
    ) -> None:
        endpoints = self._arm_endpoints.get(STAGE_CODES["approach"])
        if endpoints is None:
            return
        start, end = endpoints
        approach_delta = end[template_ids] - start[template_ids]
        shape_scale = 1.0 if self.shape_profile.is_flat else 0.35
        scale = (
            shape_scale
            * ingress_gain
            * self.mode_ingress_scale[mode_ids]
        )
        target[:, : self.arm_action_size] += scale[:, None] * approach_delta

    def _trajectory_from_world(
        self,
        world: int,
        batch: TemporalBatch,
        controls: torch.Tensor,
        reward: torch.Tensor,
        max_lift: torch.Tensor,
        final_lift: torch.Tensor,
        tail_min_lift: torch.Tensor,
        tail_max_speed: torch.Tensor,
        tail_max_angular_speed: torch.Tensor,
        tail_contact_fraction: torch.Tensor,
        tail_grasp_fraction: torch.Tensor,
        tail_thumb_fraction: torch.Tensor,
        tail_mean_contact_digits: torch.Tensor,
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
                    "approach_ingress_gain",
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
                "mode_ingress_scale": float(mode.ingress_scale),
                "object_shape_family": self.shape_profile.family,
                "object_extents": list(self.shape_profile.extents),
                "object_flatness_ratio": self.shape_profile.flatness_ratio,
                "mjwarp_success": bool(success[world].item()),
                "mjwarp_max_lift": float(max_lift[world].item()),
                "mjwarp_final_lift": float(final_lift[world].item()),
                "mjwarp_tail_min_lift": float(tail_min_lift[world].item()),
                "mjwarp_tail_max_speed": float(tail_max_speed[world].item()),
                "mjwarp_tail_max_angular_speed": float(
                    tail_max_angular_speed[world].item()
                ),
                "mjwarp_tail_contact_fraction": float(
                    tail_contact_fraction[world].item()
                ),
                "mjwarp_tail_grasp_fraction": float(
                    tail_grasp_fraction[world].item()
                ),
                "mjwarp_tail_thumb_fraction": float(
                    tail_thumb_fraction[world].item()
                ),
                "mjwarp_tail_mean_contact_digits": float(
                    tail_mean_contact_digits[world].item()
                ),
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
        tail_max_speed = torch.zeros((self.num_envs,), device=self.torch_device)
        tail_max_angular_speed = torch.zeros(
            (self.num_envs,), device=self.torch_device
        )
        tail_contact_sum = torch.zeros((self.num_envs,), device=self.torch_device)
        tail_grasp_sum = torch.zeros((self.num_envs,), device=self.torch_device)
        tail_thumb_sum = torch.zeros((self.num_envs,), device=self.torch_device)
        tail_digit_sum = torch.zeros((self.num_envs,), device=self.torch_device)
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
            if stage in (
                STAGE_CODES["close"],
                STAGE_CODES["hold"],
                STAGE_CODES["lift"],
                STAGE_CODES["verify"],
            ):
                self._apply_ingress(
                    target,
                    template_ids,
                    mode_ids,
                    parameters[:, 29],
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
                # Contact semantics only affect the sustained tail criterion.
                # Restricting the contact reduction to these frames avoids a
                # GPU synchronization on every approach/close frame.
                self._update_contacts()
                object_velocity = self.qvel[
                    :, self.object_qvel_adr : self.object_qvel_adr + 3
                ]
                object_angular_velocity = self.qvel[
                    :, self.object_qvel_adr + 3 : self.object_qvel_adr + 6
                ]
                object_speed = torch.linalg.vector_norm(object_velocity, dim=1)
                object_angular_speed = torch.linalg.vector_norm(
                    object_angular_velocity, dim=1
                )
                contact_present = self.contact_counts > 0
                if self.has_digit_contacts:
                    digit_count = self.digit_flags.sum(dim=1).float()
                    thumb_contact = self.digit_flags[:, 4] > 0
                    non_thumb_contact = self.digit_flags[:, :4].sum(dim=1) > 0
                else:
                    digit_count = torch.clamp(self.contact_counts.float(), max=5.0)
                    thumb_contact = self.contact_counts >= 1
                    non_thumb_contact = self.contact_counts >= 2
                palm_contact = self.palm_flags > 0
                grasp_present = (thumb_contact & non_thumb_contact) | (
                    palm_contact & (self.contact_counts >= 2)
                )
                tail_min_lift = torch.minimum(tail_min_lift, lift)
                tail_lift_sum += lift
                tail_max_speed = torch.maximum(tail_max_speed, object_speed)
                tail_max_angular_speed = torch.maximum(
                    tail_max_angular_speed, object_angular_speed
                )
                tail_contact_sum += contact_present.float()
                tail_grasp_sum += grasp_present.float()
                tail_thumb_sum += thumb_contact.float()
                tail_digit_sum += digit_count
                tail_samples += 1

        final_lift = self.xpos[:, self.object_body_id, 2] - initial_z
        tail_mean_lift = tail_lift_sum / float(max(tail_samples, 1))
        tail_contact_fraction = tail_contact_sum / float(max(tail_samples, 1))
        tail_grasp_fraction = tail_grasp_sum / float(max(tail_samples, 1))
        tail_thumb_fraction = tail_thumb_sum / float(max(tail_samples, 1))
        tail_mean_contact_digits = tail_digit_sum / float(max(tail_samples, 1))
        flat_thumb_ok = tail_thumb_fraction >= self.config.minimum_flat_thumb_fraction
        if not self.shape_profile.is_flat:
            flat_thumb_ok = torch.ones_like(flat_thumb_ok)
        success = (
            (tail_min_lift >= self.config.success_lift_height)
            & (tail_max_speed <= self.config.maximum_object_speed)
            & (
                tail_max_angular_speed
                <= self.config.maximum_object_angular_speed
            )
            & (
                tail_contact_fraction
                >= self.config.minimum_tail_contact_fraction
            )
            & (tail_grasp_fraction >= self.config.minimum_tail_grasp_fraction)
            & flat_thumb_ok
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
        ingress_cost = parameters[:, 29].square()
        edit_cost = torch.where(reference_mask, 0.0, edit_cost)
        timing_cost = torch.where(reference_mask, 0.0, timing_cost)
        ingress_cost = torch.where(reference_mask, 0.0, ingress_cost)
        score_weights = self.mode_score_weights[mode_ids]
        lift_score = (
            score_weights[:, 0] * max_progress
            + score_weights[:, 1] * final_progress
            + score_weights[:, 2] * tail_progress
            + score_weights[:, 3] * tail_min_progress
        )
        grasp_quality = 0.15 + 0.85 * tail_grasp_fraction
        speed_excess = torch.clamp(
            tail_max_speed / self.config.maximum_object_speed - 1.0,
            min=0.0,
            max=5.0,
        )
        angular_speed_ratio = torch.clamp(
            tail_max_angular_speed / self.config.maximum_object_angular_speed,
            min=0.0,
            max=5.0,
        )
        angular_speed_excess = torch.clamp(
            angular_speed_ratio - 1.0,
            min=0.0,
            max=5.0,
        )
        flat_thumb_weight = 2.0 if self.shape_profile.is_flat else 0.5
        reward = (
            lift_score * grasp_quality
            + 1.2 * tail_contact_fraction
            + 3.0 * tail_grasp_fraction
            + flat_thumb_weight * tail_thumb_fraction
            + 0.20 * torch.clamp(tail_mean_contact_digits / 3.0, 0.0, 1.0)
            + 15.0 * success.float()
            - 1.5 * speed_excess
            - 0.5 * angular_speed_ratio
            - 1.5 * angular_speed_excess
            - 0.03 * edit_cost
            - 0.01 * timing_cost
            - 0.01 * ingress_cost
        )
        return TemporalEvaluation(
            rewards=reward.detach().cpu().numpy(),
            max_lift=max_lift.detach().cpu().numpy(),
            final_lift=final_lift.detach().cpu().numpy(),
            tail_min_lift=tail_min_lift.detach().cpu().numpy(),
            tail_max_speed=tail_max_speed.detach().cpu().numpy(),
            tail_max_angular_speed=tail_max_angular_speed.detach().cpu().numpy(),
            tail_contact_fraction=tail_contact_fraction.detach().cpu().numpy(),
            tail_grasp_fraction=tail_grasp_fraction.detach().cpu().numpy(),
            tail_thumb_fraction=tail_thumb_fraction.detach().cpu().numpy(),
            tail_mean_contact_digits=tail_mean_contact_digits.detach().cpu().numpy(),
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
        tail_max_speed = torch.as_tensor(evaluation.tail_max_speed, device=self.torch_device)
        tail_max_angular_speed = torch.as_tensor(
            evaluation.tail_max_angular_speed, device=self.torch_device
        )
        tail_contact_fraction = torch.as_tensor(
            evaluation.tail_contact_fraction, device=self.torch_device
        )
        tail_grasp_fraction = torch.as_tensor(
            evaluation.tail_grasp_fraction, device=self.torch_device
        )
        tail_thumb_fraction = torch.as_tensor(
            evaluation.tail_thumb_fraction, device=self.torch_device
        )
        tail_mean_contact_digits = torch.as_tensor(
            evaluation.tail_mean_contact_digits, device=self.torch_device
        )
        success = torch.as_tensor(evaluation.success, device=self.torch_device)
        return self._trajectory_from_world(
            world,
            batch,
            evaluation.controls,
            reward,
            max_lift,
            final_lift,
            tail_min_lift,
            tail_max_speed,
            tail_max_angular_speed,
            tail_contact_fraction,
            tail_grasp_fraction,
            tail_thumb_fraction,
            tail_mean_contact_digits,
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
        warm_starts: tuple[TemporalWarmStart, ...] = (),
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
                0.08,
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
                0.08,
            ],
            dtype=np.float32,
        )
        self.low = env.parameter_bounds_low
        self.high = env.parameter_bounds_high
        self.warm_starts = tuple(warm_starts)
        for warm_start in self.warm_starts:
            warm_start.validate(
                template_count=len(env.templates), mode_count=env.mode_count
            )
        if self.warm_starts:
            self.mean = np.clip(
                np.asarray(self.warm_starts[0].parameters, dtype=np.float32),
                self.low,
                self.high,
            )
        template_prior = getattr(
            env,
            "template_prior_probabilities",
            np.full(len(env.templates), 1.0 / len(env.templates), dtype=np.float64),
        )
        mode_prior = getattr(
            env,
            "mode_prior_probabilities",
            np.full(env.mode_count, 1.0 / env.mode_count, dtype=np.float64),
        )
        self.template_probabilities = np.asarray(
            template_prior, dtype=np.float64
        ).copy()
        self.mode_probabilities = np.asarray(mode_prior, dtype=np.float64).copy()

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
        reference_count = min(
            len(self.env.templates),
            max(1, self.config.population_size // 8),
            self.config.population_size - self.env.mode_count,
        )
        # Stratify the categorical seed so every requested grasp family is
        # evaluated in every CEM iteration.  This is the cheap analogue of
        # GraspM3's many directional demonstrations and avoids early collapse
        # onto a thumb/finger pinch.
        mode_ids[: self.env.mode_count] = np.arange(self.env.mode_count, dtype=np.int64)
        template_order = np.argsort(-self.template_probabilities, kind="stable")
        template_ids[: self.env.mode_count] = template_order[
            np.arange(self.env.mode_count) % len(template_order)
        ]
        parameters[: self.env.mode_count] = self.mean

        # A second deterministic anchor tests meaningful preload and ingress
        # before CEM has learned either from sparse binary success. Random-only
        # populations frequently miss this regime on thin, low-friction boxes.
        available = self.config.population_size - reference_count
        if available >= 2 * self.env.mode_count:
            start = self.env.mode_count
            stop = 2 * self.env.mode_count
            mode_ids[start:stop] = np.arange(self.env.mode_count, dtype=np.int64)
            template_ids[start:stop] = template_order[
                (np.arange(self.env.mode_count) + 1) % len(template_order)
            ]
            parameters[start:stop] = self.mean
            parameters[start:stop, 6:12] = 0.38
            parameters[start:stop, 28] = min(0.30, self.config.clench_gain_max)
            parameters[start:stop, 29] = min(0.16, self.config.ingress_gain_max)
        reference_mask = np.zeros(self.config.population_size, dtype=bool)
        if reference_count > 0:
            start = self.config.population_size - reference_count
            template_ids[start:] = template_order[:reference_count]
            mode_ids[start:] = self.env.reference_mode_id
            reference_mask[start:] = True
        if self.warm_starts:
            available = self.config.population_size - reference_count
            count = min(len(self.warm_starts), max(0, available - self.env.mode_count))
            start = available - count
            for row, warm_start in enumerate(self.warm_starts[:count], start=start):
                parameters[row] = np.clip(warm_start.parameters, self.low, self.high)
                template_ids[row] = warm_start.template_id
                mode_ids[row] = warm_start.mode_id
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
                "best_tail_contact_fraction": float(
                    np.max(evaluation.tail_contact_fraction)
                ),
                "best_tail_grasp_fraction": float(
                    np.max(evaluation.tail_grasp_fraction)
                ),
                "best_tail_thumb_fraction": float(
                    np.max(evaluation.tail_thumb_fraction)
                ),
                "lowest_tail_max_speed": float(
                    np.min(evaluation.tail_max_speed)
                ),
                "lowest_tail_max_angular_speed": float(
                    np.min(evaluation.tail_max_angular_speed)
                ),
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
