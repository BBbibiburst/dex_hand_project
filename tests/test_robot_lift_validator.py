"""Fast contracts for full-robot Lift prechecks."""

from types import SimpleNamespace

import numpy as np

from source.envs.manipulation.placement import FixedTablePlacementSampler
from source.execution.robot_lift import (
    _ik_waypoint_is_reachable,
    task_scene_schedule,
)
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


def test_task_scene_schedule_rotates_before_pulling_object() -> None:
    scenes = task_scene_schedule(
        seed=3,
        scene_attempts=10,
        rotations_per_distance=4,
        pull_step=0.05,
        maximum_pull=0.10,
    )
    assert [scene["pull_toward_robot"] for scene in scenes] == [
        0.0,
        0.0,
        0.0,
        0.0,
        0.05,
        0.05,
        0.05,
        0.05,
        0.10,
        0.10,
    ]
    assert np.isclose(scenes[4]["object_xy"][0], scenes[0]["object_xy"][0] - 0.05)


def test_fixed_table_placement_uses_explicit_pose() -> None:
    sampler = FixedTablePlacementSampler(xy=(-0.1, 0.02), yaw=np.pi / 2)
    obj = SimpleNamespace(name="object", bottom_offset=0.03)
    placement = sampler.sample(
        [obj], rng=np.random.default_rng(0), reference_pos=np.array([0.55, 0.0, 0.5])
    )["object"]
    assert np.allclose(placement[0], [0.45, 0.02, 0.532])
    assert np.allclose(placement[1], [np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)])
