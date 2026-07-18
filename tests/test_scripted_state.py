"""Typed scripted-strategy state checks."""

import numpy as np

from source.scripted.lift import LiftStrategy, LiftStrategyState


def test_lift_strategy_state_reset() -> None:
    state = LiftStrategyState(
        lift_stable_steps=5,
        verify_success_steps=3,
        hold_wrist_position=np.ones(3),
        verified_success=True,
    )

    state.reset()

    assert state.lift_stable_steps == 0
    assert state.verify_success_steps == 0
    assert state.hold_wrist_position is None
    assert state.verified_success is False


def test_lift_strategy_reset_clears_state() -> None:
    strategy = LiftStrategy()
    strategy.state.lift_stable_steps = 5
    strategy.state.verified_success = True

    strategy.reset()

    assert strategy.state.lift_stable_steps == 0
    assert strategy.state.verified_success is False
