"""Fast contracts for full-robot Lift prechecks."""

from source.grasping.robot_lift_validator import _ik_waypoint_is_reachable
from source.scripted.lift import LiftStrategy


def test_robot_lift_precheck_rejects_unreachable_ik() -> None:
    assert _ik_waypoint_is_reachable(0.02, 0.1)
    assert not _ik_waypoint_is_reachable(1.0, 0.1)
    assert not _ik_waypoint_is_reachable(0.02, 2.0)


def test_single_candidate_validator_can_detect_first_restart() -> None:
    strategy = LiftStrategy(grasp_candidate_index=0)
    strategy._advance_grasp_candidate()
    assert strategy.restart_count == 1
    assert strategy.archive_candidate_index == 0
