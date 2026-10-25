import pytest

from source.imitation.lift_supervisor import LiftPhase, LiftSupervisor, LiftSupervisorConfig


def test_supervisor_clamps_and_latches_hold():
    supervisor = LiftSupervisor(
        LiftSupervisorConfig(target_rise_m=0.08, target_tolerance_m=0.001, hold_steps=2)
    )
    supervisor.reset(0.5)
    assert supervisor.clamp_delta_z(0.57, 0.02) == pytest.approx(0.01)
    assert supervisor.observe(0.5795) is LiftPhase.TARGET_REACHED
    assert supervisor.clamp_delta_z(0.5795, 0.01) == 0.0
    assert supervisor.observe(0.5795) is LiftPhase.HOLD_STABILIZE
    assert supervisor.observe(0.5795) is LiftPhase.HOLD_STABILIZE
    assert supervisor.observe(0.5795) is LiftPhase.TERMINATED
    assert supervisor.terminal


def test_supervisor_times_out_without_target():
    supervisor = LiftSupervisor(
        LiftSupervisorConfig(target_rise_m=0.08, hold_steps=2, maximum_active_steps=2)
    )
    supervisor.reset(0.5)
    assert supervisor.observe(0.51) is LiftPhase.LIFT_ACTIVE
    assert supervisor.observe(0.52) is LiftPhase.TIMEOUT
