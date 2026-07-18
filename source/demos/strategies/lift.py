"""Scripted approach, grasp, and lift policy for the built-in lift task."""

from __future__ import annotations

import numpy as np

from source.demos.strategies.base import ActionContext, PhaseContext, PhaseResult, TaskStrategy


class LiftStrategy(TaskStrategy):
    phases = ("open", "approach", "descend", "grasp", "lift", "verify")

    def __init__(
        self,
        *,
        approach_height: float = 0.12,
        grasp_height: float = 0.035,
        lift_height: float = 0.16,
        position_tolerance: float = 0.008,
        settle_steps: int = 12,
        max_position_step: float = 0.012,
        grasp_quaternion_wxyz: tuple[float, float, float, float] | None = None,
    ) -> None:
        super().__init__(max_position_step=max_position_step)
        self.approach_height = float(approach_height)
        self.grasp_height = float(grasp_height)
        self.lift_height = float(lift_height)
        self.position_tolerance = float(position_tolerance)
        self.settle_steps = int(settle_steps)
        self.grasp_quaternion_wxyz = (
            None
            if grasp_quaternion_wxyz is None
            else np.asarray(grasp_quaternion_wxyz, dtype=np.float64)
        )

    def execute_phase(
        self, phase_index: int, context: PhaseContext
    ) -> tuple[PhaseResult, ActionContext]:
        phase = self.phases[phase_index]
        env = context.env
        cube = np.asarray(context.observation["cube_pos"], dtype=np.float64)
        current = env.controller.current_ik_action(env.model, env.data)
        ee_position = current[:3]
        hand_size = env.controller.hand_controller.action_size
        hand_low = np.asarray(env.action_space.low[-hand_size:], dtype=np.float64)
        hand_high = np.asarray(env.action_space.high[-hand_size:], dtype=np.float64)
        open_hand = hand_high
        closed_hand = hand_low

        if phase == "open":
            if "grasp_quaternion" not in context.memory:
                context.memory["grasp_quaternion"] = (
                    current[3:7].copy()
                    if self.grasp_quaternion_wxyz is None
                    else self.grasp_quaternion_wxyz.copy()
                )
            result = (
                PhaseResult.NEXT
                if context.phase_step >= self.settle_steps
                else PhaseResult.CONTINUE
            )
            return result, ActionContext(hand_target=open_hand)

        grasp_quaternion = np.asarray(context.memory["grasp_quaternion"])

        if phase == "approach":
            target = cube + np.asarray([0.0, 0.0, self.approach_height])
            reached = np.linalg.norm(target - ee_position) <= self.position_tolerance
            result = PhaseResult.NEXT if reached else PhaseResult.CONTINUE
            return result, ActionContext(
                ee_target_position=target,
                ee_target_quaternion_wxyz=grasp_quaternion,
                hand_target=open_hand,
            )

        if phase == "descend":
            target = cube + np.asarray([0.0, 0.0, self.grasp_height])
            reached = np.linalg.norm(target - ee_position) <= self.position_tolerance
            result = PhaseResult.NEXT if reached else PhaseResult.CONTINUE
            if reached:
                context.memory["grasp_position"] = target.copy()
            return result, ActionContext(
                ee_target_position=target,
                ee_target_quaternion_wxyz=grasp_quaternion,
                hand_target=open_hand,
            )

        if phase == "grasp":
            target = np.asarray(context.memory.get("grasp_position", ee_position))
            result = (
                PhaseResult.NEXT
                if context.phase_step >= self.settle_steps
                else PhaseResult.CONTINUE
            )
            return result, ActionContext(
                ee_target_position=target,
                ee_target_quaternion_wxyz=grasp_quaternion,
                hand_target=closed_hand,
            )

        if phase == "lift":
            grasp_position = np.asarray(context.memory.get("grasp_position", ee_position))
            target = grasp_position.copy()
            target[2] = max(target[2] + self.lift_height, cube[2] + self.lift_height)
            reached = np.linalg.norm(target - ee_position) <= self.position_tolerance
            result = PhaseResult.NEXT if reached else PhaseResult.CONTINUE
            return result, ActionContext(
                ee_target_position=target,
                ee_target_quaternion_wxyz=grasp_quaternion,
                hand_target=closed_hand,
            )

        if context.info.get("task_success", False):
            return PhaseResult.NEXT, ActionContext(hand_target=closed_hand)
        if context.phase_step >= self.settle_steps:
            return PhaseResult.RESTART, ActionContext(hand_target=open_hand)
        return PhaseResult.CONTINUE, ActionContext(hand_target=closed_hand)
