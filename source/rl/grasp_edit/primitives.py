"""Low-dimensional grasp-style priors for the six-drive Dex Hand.

The primitives are residual hand synergies around an UltraDexGrasp candidate,
not absolute poses.  They intentionally keep the search close to a validated
prior while exposing qualitatively different grasp styles to PPO.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class GraspPrimitive:
    name: str
    approach_bias: tuple[float, float, float, float, float, float]
    final_bias: tuple[float, float, float, float, float, float]
    close_power: float
    description: str
    # ``mode_family`` is the human-facing macro grasp family.  ``name`` stays
    # short and backwards compatible with the original primitive PPO CLI.
    mode_family: str = "wrap"
    objective_name: str = "sustained_lift"
    # We score (max lift, final lift, tail mean lift, tail minimum lift) with
    # these weights during the cheap MJWarp search.  C MuJoCo remains the
    # authoritative binary label.
    score_weights: tuple[float, float, float, float] = (1.0, 3.0, 4.0, 5.0)
    table_assisted: bool = False
    enclosure_bias: float = 0.0
    support_bias: float = 0.0

    def validate(self) -> None:
        if len(self.approach_bias) != 6 or len(self.final_bias) != 6:
            raise ValueError(f"Primitive {self.name!r} must define six actuator biases.")
        if not np.all(np.isfinite(self.approach_bias)) or not np.all(
            np.isfinite(self.final_bias)
        ):
            raise ValueError(f"Primitive {self.name!r} contains a non-finite bias.")
        if self.close_power <= 0.0:
            raise ValueError(f"Primitive {self.name!r} close_power must be positive.")
        if len(self.score_weights) != 4 or not np.all(np.isfinite(self.score_weights)):
            raise ValueError(f"Primitive {self.name!r} has invalid score weights.")
        if any(value < 0.0 for value in self.score_weights):
            raise ValueError(f"Primitive {self.name!r} score weights must be non-negative.")
        if not np.isfinite(self.enclosure_bias) or not np.isfinite(self.support_bias):
            raise ValueError(f"Primitive {self.name!r} contains a non-finite mode bias.")
        if self.enclosure_bias < 0.0 or self.support_bias < 0.0:
            raise ValueError(f"Primitive {self.name!r} mode biases must be non-negative.")
        if not self.mode_family or not self.objective_name:
            raise ValueError(f"Primitive {self.name!r} must define mode metadata.")


# Physical actuator order used by this project:
# finger0, finger1, finger2, finger3, thumb_rotate, thumb_grasp.
# OPEN_FRACTIONS uses thumb_rotate=1.0, so negative thumb_rotate bias means
# stronger thumb opposition.
GRASP_PRIMITIVES: dict[str, GraspPrimitive] = {
    "wrap": GraspPrimitive(
        name="wrap",
        approach_bias=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        final_bias=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        close_power=1.0,
        description="Power-wrap baseline with multi-finger enclosure and palm contact.",
        mode_family="power_wrap",
        objective_name="enclosure_sustained_lift",
        score_weights=(0.8, 2.5, 4.5, 6.0),
        enclosure_bias=1.0,
    ),
    "pinch": GraspPrimitive(
        name="pinch",
        approach_bias=(0.10, 0.03, -0.10, -0.16, -0.10, 0.04),
        final_bias=(0.16, 0.05, -0.18, -0.24, -0.18, 0.12),
        close_power=1.15,
        description="Thumb-opposed precision bias with one leading finger dominant.",
        mode_family="pinch",
        objective_name="opposition_final_lift",
        score_weights=(1.0, 4.0, 2.5, 3.0),
    ),
    "tripod": GraspPrimitive(
        name="tripod",
        approach_bias=(0.08, 0.02, -0.04, -0.10, -0.08, 0.04),
        final_bias=(0.12, 0.05, -0.06, -0.12, -0.12, 0.10),
        close_power=1.05,
        description="Thumb plus the two leading fingers form a three-point pinch.",
        mode_family="tripod",
        objective_name="three_point_sustained_lift",
        score_weights=(1.0, 3.5, 3.5, 4.0),
    ),
    "spherical": GraspPrimitive(
        name="spherical",
        approach_bias=(0.05, 0.05, 0.05, 0.05, -0.03, -0.03),
        final_bias=(0.04, 0.04, 0.04, 0.04, -0.05, -0.05),
        close_power=0.95,
        description="Balanced radial closure for balls and rounded objects.",
        mode_family="spherical",
        objective_name="multi_direction_enclosure",
        score_weights=(0.8, 2.5, 4.5, 6.0),
        enclosure_bias=1.0,
    ),
    "hook": GraspPrimitive(
        name="hook",
        approach_bias=(0.22, 0.22, 0.22, 0.22, 0.00, -0.12),
        final_bias=(0.12, 0.12, 0.12, 0.12, -0.04, -0.16),
        close_power=0.75,
        description="Early four-finger flexion for side/rim hooking.",
        mode_family="hook",
        objective_name="hook_sustained_lift",
        score_weights=(0.7, 2.0, 4.0, 6.0),
        enclosure_bias=0.6,
    ),
    "cradle": GraspPrimitive(
        name="cradle",
        approach_bias=(0.14, 0.14, 0.14, 0.14, -0.04, -0.08),
        final_bias=(0.08, 0.08, 0.08, 0.08, -0.06, -0.06),
        close_power=1.50,
        description="Four-finger cradle/support with delayed thumb side-limiting.",
        mode_family="cradle",
        objective_name="support_containment_tail_lift",
        score_weights=(0.5, 1.5, 4.0, 8.0),
        enclosure_bias=0.8,
        support_bias=1.0,
    ),
    "lateral": GraspPrimitive(
        name="lateral",
        approach_bias=(0.10, 0.08, 0.02, -0.04, -0.16, 0.06),
        final_bias=(0.12, 0.10, 0.02, -0.06, -0.20, 0.10),
        close_power=1.20,
        description="Thumb-side edge clamp for thin plates and box edges.",
        mode_family="lateral",
        objective_name="edge_clamp_sustained_lift",
        score_weights=(1.0, 3.0, 3.5, 5.0),
    ),
    "table_assisted": GraspPrimitive(
        name="table_assisted",
        approach_bias=(0.18, 0.18, 0.16, 0.12, -0.02, -0.08),
        final_bias=(0.10, 0.10, 0.08, 0.06, -0.05, -0.08),
        close_power=1.35,
        description=(
            "Contact-rich pre-grasp seed that may push/roll an object before enclosure; "
            "it is not a full object-repositioning planner."
        ),
        mode_family="table_assisted",
        objective_name="support_containment_tail_lift",
        score_weights=(0.3, 1.0, 4.0, 9.0),
        table_assisted=True,
        enclosure_bias=0.8,
        support_bias=1.0,
    ),
}

# Canonical names intentionally stay compatible with the original four-style
# primitive PPO.  These aliases make the taxonomy terminology pleasant to use
# without creating duplicate categorical modes.
GRASP_PRIMITIVE_ALIASES: dict[str, str] = {
    "power_wrap": "wrap",
    "support": "cradle",
}

_CANONICAL_NAMES: tuple[str, ...] = (
    "wrap",
    "pinch",
    "tripod",
    "spherical",
    "hook",
    "cradle",
    "lateral",
    "table_assisted",
)

for _primitive in GRASP_PRIMITIVES.values():
    _primitive.validate()


def available_grasp_primitives() -> tuple[str, ...]:
    return _CANONICAL_NAMES


def resolve_grasp_primitives(
    names: tuple[str, ...] | list[str] | str,
) -> tuple[GraspPrimitive, ...]:
    if isinstance(names, str):
        names = tuple(part.strip() for part in names.split(",") if part.strip())
    else:
        names = tuple(names)
    if not names:
        raise ValueError("At least one grasp primitive is required.")
    if names == ("all",):
        names = available_grasp_primitives()
    names = tuple(GRASP_PRIMITIVE_ALIASES.get(name, name) for name in names)
    unknown = [name for name in names if name not in GRASP_PRIMITIVES]
    if unknown:
        raise ValueError(
            f"Unknown grasp primitive(s): {unknown}. Available: {available_grasp_primitives()}"
        )
    unique = tuple(dict.fromkeys(names))
    return tuple(GRASP_PRIMITIVES[name] for name in unique)
