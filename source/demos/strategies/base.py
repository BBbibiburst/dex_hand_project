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

    def __init__(self, *, max_position_step: float = 0.015) -> None:
        if max_position_step <= 0:
            raise ValueError("max_position_step must be positive.")
        self.max_position_step = float(max_position_step)
        self.phase_index = 0
        self.phase_step = 0
        self.aborted = False
        self.memory: dict[str, Any] = {}

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

    @property
    def finished(self) -> bool:
        return self.phase_index >= len(self.phases)

    @property
    def phase_name(self) -> str:
        if self.finished:
            return "done"
        return self.phases[self.phase_index]

    def tick(
        self,
        observation: dict[str, Any],
        info: dict[str, Any],
        step: int,
        env,
    ) -> tuple[np.ndarray, ActionContext]:
        current = env.controller.current_ik_action(env.model, env.data)
        if self.aborted or self.finished:
            return current, ActionContext()
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
            if distance > self.max_position_step:
                delta *= self.max_position_step / distance
            action[:3] += delta
        if context.ee_target_quaternion_wxyz is not None:
            action[3:7] = normalize_quat(
                np.asarray(context.ee_target_quaternion_wxyz, dtype=np.float64)
            )
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
