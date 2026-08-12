"""Fast contracts for full-robot Lift prechecks."""

from source.grasping.robot_lift_validator import _ik_waypoint_is_reachable


def test_robot_lift_precheck_rejects_unreachable_ik() -> None:
    assert _ik_waypoint_is_reachable(0.02, 0.1)
    assert not _ik_waypoint_is_reachable(1.0, 0.1)
    assert not _ik_waypoint_is_reachable(0.02, 2.0)
