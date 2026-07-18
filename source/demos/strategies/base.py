"""Reusable phase-state machine for absolute-IK scripted policies."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any

import numpy as np

from source.geometry import normalize_quat


class PhaseResult(Enum):
    CONTINUE = auto()
    NEXT = auto()
    RETRY = auto()
    RESTART = auto()
    ABORT = auto()


@dataclass(frozen=True)
class PhaseContext:
    observation: dict[str, Any]
    info: dict[str, Any]
    step: int
    phase_step: int
    memory: dict[str, Any]
    env: Any


@dataclass(frozen=True)
class ActionContext:
    ee_target_position: np.ndarray | None = None
    ee_target_quaternion_wxyz: np.ndarray | None = None
    hand_target: np.ndarray | None = None


class TaskStrategy(ABC):
    """Generate bounded absolute IK actions through a sequence of phases."""

    phase_position_speeds: dict[str, float] = {}
    phase_orientation_speeds: dict[str, float] = {}

    def __init__(
        self,
        *,
        max_position_step: float = 0.015,
        max_orientation_step: float = 0.20,
    ) -> None:
        if max_position_step <= 0:
            raise ValueError("max_position_step must be positive.")
        if max_orientation_step <= 0:
            raise ValueError("max_orientation_step must be positive.")
        self.max_position_step = float(max_position_step)
        self.max_orientation_step = float(max_orientation_step)
        self.phase_index = 0
        self.phase_step = 0
        self.aborted = False
        self.memory: dict[str, Any] = {}
        self._command: np.ndarray | None = None

    @property
    @abstractmethod
    def phases(self) -> tuple[str, ...]:
        """Ordered phase names."""

    @abstractmethod
    def execute_phase(
        self, phase_index: int, context: PhaseContext
    ) -> tuple[PhaseResult, ActionContext]:
        """Return the current phase result and desired absolute targets."""

    def reset(self) -> None:
        self.phase_index = 0
        self.phase_step = 0
        self.aborted = False
        self.memory.clear()
        self._command = None

    @property
    def finished(self) -> bool:
        return self.phase_index >= len(self.phases)

    @property
    def phase_name(self) -> str:
        if self.finished:
            return "done"
        return self.phases[self.phase_index]

    @property
    def phase_prompt(self) -> str:
        """Human-readable instruction for the active phase."""
        return "strategy complete" if self.finished else self.phase_name

    def tick(
        self,
        observation: dict[str, Any],
        info: dict[str, Any],
        step: int,
        env,
    ) -> tuple[np.ndarray, ActionContext]:
        measured = env.controller.current_ik_action(env.model, env.data)
        if self.aborted or self.finished:
            return measured, ActionContext()
        current = measured if self._command is None else self._command
        context = PhaseContext(
            observation=observation,
            info=info,
            step=step,
            phase_step=self.phase_step,
            memory=self.memory,
            env=env,
        )
        result, action_context = self.execute_phase(self.phase_index, context)
        action = self._build_action(current, action_context, env)
        self._command = action.astype(np.float64).copy()
        self.phase_step += 1
        if result is PhaseResult.NEXT:
            self.phase_index += 1
            self.phase_step = 0
        elif result is PhaseResult.RETRY:
            self.phase_step = 0
        elif result is PhaseResult.RESTART:
            self.phase_index = 0
            self.phase_step = 0
            self.memory.clear()
        elif result is PhaseResult.ABORT:
            self.aborted = True
        return action, action_context

    def _build_action(
        self,
        current: np.ndarray,
        context: ActionContext,
        env,
    ) -> np.ndarray:
        action = current.astype(np.float64).copy()
        if context.ee_target_position is not None:
            target = np.asarray(context.ee_target_position, dtype=np.float64)
            delta = target - action[:3]
            distance = float(np.linalg.norm(delta))
            control_dt = float(env.config.control_dt)
            phase_speed = self.phase_position_speeds.get(self.phase_name)
            max_position_step = self.max_position_step
            if phase_speed is not None:
                max_position_step = min(max_position_step, phase_speed * control_dt)
            if distance > max_position_step:
                delta *= max_position_step / distance
            action[:3] += delta
        if context.ee_target_quaternion_wxyz is not None:
            source = normalize_quat(action[3:7])
            target = normalize_quat(
                np.asarray(context.ee_target_quaternion_wxyz, dtype=np.float64)
            )
            dot = float(np.clip(np.dot(source, target), -1.0, 1.0))
            if dot < 0.0:
                target = -target
                dot = -dot
            angle = 2.0 * np.arccos(np.clip(dot, 0.0, 1.0))
            max_orientation_step = self.max_orientation_step
            phase_speed = self.phase_orientation_speeds.get(self.phase_name)
            if phase_speed is not None:
                max_orientation_step = min(
                    max_orientation_step,
                    phase_speed * float(env.config.control_dt),
                )
            blend = (
                1.0
                if angle <= max_orientation_step
                else max_orientation_step / angle
            )
            action[3:7] = normalize_quat((1.0 - blend) * source + blend * target)
        if context.hand_target is not None:
            hand_size = env.controller.hand_controller.action_size
            target = np.asarray(context.hand_target, dtype=np.float64).reshape(-1)
            if target.shape != (hand_size,):
                raise ValueError(
                    f"Scripted hand target must have shape {(hand_size,)}, got {target.shape}."
                )
            action[-hand_size:] = target
        return np.clip(action, env.action_space.low, env.action_space.high).astype(np.float32)

    def status(self) -> dict[str, Any]:
        return {
            "phase_index": self.phase_index,
            "phase_name": self.phase_name,
            "phase_step": self.phase_step,
            "finished": self.finished,
            "aborted": self.aborted,
        }
