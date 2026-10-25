"""Deployable wrist-height supervisor for bounded Cartesian Lift execution."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class LiftPhase(str, Enum):
    LIFT_ACTIVE = "LIFT_ACTIVE"
    TARGET_REACHED = "TARGET_REACHED"
    HOLD_STABILIZE = "HOLD_STABILIZE"
    TERMINATED = "TERMINATED"
    TIMEOUT = "TIMEOUT"
    CONTROL_FAULT = "CONTROL_FAULT"
    SUSPECTED_SLIP = "SUSPECTED_SLIP"


@dataclass(frozen=True)
class LiftSupervisorConfig:
    target_rise_m: float = 0.085
    target_tolerance_m: float = 0.0015
    hold_steps: int = 80
    maximum_active_steps: int = 180


class LiftSupervisor:
    """Latch a task-level wrist-height goal and transition once in one direction."""

    def __init__(self, config: LiftSupervisorConfig):
        if config.target_rise_m <= 0 or config.hold_steps < 1:
            raise ValueError("Lift target and hold duration must be positive.")
        self.config = config
        self.phase = LiftPhase.LIFT_ACTIVE
        self.start_z: float | None = None
        self.goal_z: float | None = None
        self.active_steps = 0
        self.hold_steps = 0

    def reset(self, wrist_z: float) -> None:
        self.start_z = float(wrist_z)
        self.goal_z = self.start_z + self.config.target_rise_m
        self.phase = LiftPhase.LIFT_ACTIVE
        self.active_steps = 0
        self.hold_steps = 0

    def clamp_delta_z(self, anchor_z: float, requested_delta_z: float) -> float:
        if self.goal_z is None:
            raise RuntimeError("reset() must be called before use.")
        if self.phase is not LiftPhase.LIFT_ACTIVE:
            return 0.0
        return min(float(requested_delta_z), self.goal_z - float(anchor_z))

    def observe(self, wrist_z: float, *, control_ok: bool = True) -> LiftPhase:
        if self.goal_z is None:
            raise RuntimeError("reset() must be called before use.")
        if not control_ok:
            self.phase = LiftPhase.CONTROL_FAULT
            return self.phase
        if self.phase is LiftPhase.LIFT_ACTIVE:
            self.active_steps += 1
            if float(wrist_z) >= self.goal_z - self.config.target_tolerance_m:
                self.phase = LiftPhase.TARGET_REACHED
            elif self.active_steps >= self.config.maximum_active_steps:
                self.phase = LiftPhase.TIMEOUT
        elif self.phase is LiftPhase.TARGET_REACHED:
            self.phase = LiftPhase.HOLD_STABILIZE
        elif self.phase is LiftPhase.HOLD_STABILIZE:
            self.hold_steps += 1
            if self.hold_steps >= self.config.hold_steps:
                self.phase = LiftPhase.TERMINATED
        return self.phase

    @property
    def terminal(self) -> bool:
        return self.phase in {
            LiftPhase.TERMINATED,
            LiftPhase.TIMEOUT,
            LiftPhase.CONTROL_FAULT,
            LiftPhase.SUSPECTED_SLIP,
        }
