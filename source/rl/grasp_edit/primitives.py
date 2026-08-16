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

    def validate(self) -> None:
        if len(self.approach_bias) != 6 or len(self.final_bias) != 6:
            raise ValueError(f"Primitive {self.name!r} must define six actuator biases.")
        if not np.all(np.isfinite(self.approach_bias)) or not np.all(
            np.isfinite(self.final_bias)
        ):
            raise ValueError(f"Primitive {self.name!r} contains a non-finite bias.")
        if self.close_power <= 0.0:
            raise ValueError(f"Primitive {self.name!r} close_power must be positive.")


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
        description="Ultra-like power/wrap grasp; exact baseline when used alone.",
    ),
    "pinch": GraspPrimitive(
        name="pinch",
        approach_bias=(0.10, 0.03, -0.10, -0.16, -0.10, 0.04),
        final_bias=(0.16, 0.05, -0.18, -0.24, -0.18, 0.12),
        close_power=1.15,
        description="Thumb-opposed precision bias with one leading finger dominant.",
    ),
    "support": GraspPrimitive(
        name="support",
        approach_bias=(0.14, 0.14, 0.14, 0.14, -0.04, -0.08),
        final_bias=(0.08, 0.08, 0.08, 0.08, -0.06, -0.06),
        close_power=1.50,
        description="Four-finger cradle pre-shape with delayed thumb closure.",
    ),
    "hook": GraspPrimitive(
        name="hook",
        approach_bias=(0.22, 0.22, 0.22, 0.22, 0.00, -0.12),
        final_bias=(0.12, 0.12, 0.12, 0.12, -0.04, -0.16),
        close_power=0.75,
        description="Early four-finger flexion for side/rim hooking.",
    ),
}

for _primitive in GRASP_PRIMITIVES.values():
    _primitive.validate()


def available_grasp_primitives() -> tuple[str, ...]:
    return tuple(GRASP_PRIMITIVES)


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
    unknown = [name for name in names if name not in GRASP_PRIMITIVES]
    if unknown:
        raise ValueError(
            f"Unknown grasp primitive(s): {unknown}. Available: {available_grasp_primitives()}"
        )
    unique = tuple(dict.fromkeys(names))
    return tuple(GRASP_PRIMITIVES[name] for name in unique)
