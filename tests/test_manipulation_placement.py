from __future__ import annotations

from collections.abc import Callable

import mujoco
import numpy as np
import pytest

from source.envs.manipulation import make_nut_assembly_env, make_pick_place_env
from source.envs.manipulation.lift import LiftTask
from source.envs.manipulation.nut_assembly import NutAssemblyTask
from source.envs.manipulation.objects import FreeBoxSpec
from source.envs.manipulation.pick_place import PickPlaceTask
from source.envs.manipulation.placement import (
    DEFAULT_REACHABLE_REGION,
    NUT_ASSEMBLY_REACHABLE_REGION,
    PICK_PLACE_SOURCE_REGION,
    PUSH_OBJECT_REACHABLE_REGION,
    PUSH_TARGET_REACHABLE_REGION,
    STACK_REACHABLE_REGION,
    FixedTablePlacementSampler,
    PlacementRegion,
    UniformTablePlacementSampler,
)
from source.envs.manipulation.push import PushTask
from source.envs.manipulation.stack import StackTask


def _assert_inside_region(points: np.ndarray, region: PlacementRegion) -> None:
    assert np.all(points[:, 0] >= region.x_range[0])
    assert np.all(points[:, 0] <= region.x_range[1])
    assert np.all(points[:, 1] >= region.y_range[0])
    assert np.all(points[:, 1] <= region.y_range[1])


def test_object_center_sampler_does_not_shrink_region_by_object_radius() -> None:
    region = PlacementRegion(x_range=(-0.02, 0.03), y_range=(-0.01, 0.04))
    sampler = UniformTablePlacementSampler.for_object_centers(region)
    wide_object = FreeBoxSpec(
        name="wide",
        half_size=(0.20, 0.20, 0.02),
        rgba=(1.0, 1.0, 1.0, 1.0),
    )

    samples = []
    for seed in range(128):
        placements = sampler.sample(
            (wide_object,),
            rng=np.random.default_rng(seed),
            reference_pos=np.zeros(3),
        )
        samples.append(placements[wide_object.name][0][:2])

    _assert_inside_region(np.asarray(samples), region)
    assert sampler.ensure_object_boundary_in_range is False


@pytest.mark.parametrize(
    ("task_factory", "region"),
    (
        (LiftTask, DEFAULT_REACHABLE_REGION),
        (PickPlaceTask, PICK_PLACE_SOURCE_REGION),
        (StackTask, STACK_REACHABLE_REGION),
        (NutAssemblyTask, NUT_ASSEMBLY_REACHABLE_REGION),
        (PushTask, PUSH_OBJECT_REACHABLE_REGION),
    ),
)
def test_default_task_placements_stay_in_reachable_center_regions(
    task_factory: Callable[[], object],
    region: PlacementRegion,
) -> None:
    task = task_factory()
    sampler = task.placement_sampler
    samples = []

    for seed in range(512):
        placements = sampler.sample(
            task.objects,
            rng=np.random.default_rng(seed),
            reference_pos=task.table_offset,
        )
        points = np.asarray(
            [placements[obj.name][0][:2] - task.table_offset[:2] for obj in task.objects]
        )
        samples.extend(points)
        for first in range(len(points)):
            for second in range(first + 1, len(points)):
                assert np.linalg.norm(points[first] - points[second]) >= (
                    sampler.min_separation - 1e-12
                )

    _assert_inside_region(np.asarray(samples), region)
    assert sampler.ensure_object_boundary_in_range is False


def test_push_target_stays_in_reachable_region_and_across_the_table() -> None:
    task = PushTask()
    object_points = []
    target_points = []

    for seed in range(512):
        rng = np.random.default_rng(seed)
        placements = task.placement_sampler.sample(
            task.objects,
            rng=rng,
            reference_pos=task.table_offset,
        )
        object_xy = placements[task.objects[0].name][0][:2]
        target_xy = task.target_region.sample_xy(rng, task.table_offset)
        object_points.append(object_xy - task.table_offset[:2])
        target_points.append(target_xy - task.table_offset[:2])
        assert target_xy[1] - object_xy[1] >= 0.20 - 1e-12

    _assert_inside_region(np.asarray(object_points), PUSH_OBJECT_REACHABLE_REGION)
    _assert_inside_region(np.asarray(target_points), PUSH_TARGET_REACHABLE_REGION)


def test_nut_defaults_do_not_spawn_inside_pegs_or_each_other() -> None:
    env = make_nut_assembly_env(enable_tactile_sensors=False)
    try:
        object_geom_sets = [set(binding.geom_ids) for binding in env.task.bindings.objects.values()]
        peg_geom_ids = {
            geom_id
            for geom_id in range(env.model.ngeom)
            if (mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or "").startswith(
                "peg"
            )
        }

        for seed in range(128):
            env.reset(seed=seed)
            for contact_index in range(env.data.ncon):
                contact = env.data.contact[contact_index]
                pair = {int(contact.geom1), int(contact.geom2)}
                assert not (pair & peg_geom_ids and pair & set.union(*object_geom_sets))
                assert not (pair & object_geom_sets[0] and pair & object_geom_sets[1])
    finally:
        env.close()


def test_fixed_placement_replay_is_unchanged() -> None:
    sampler = FixedTablePlacementSampler(xy=(-0.1, 0.02), yaw=np.pi / 2.0)
    obj = FreeBoxSpec(
        name="object",
        half_size=(0.02, 0.02, 0.03),
        rgba=(1.0, 1.0, 1.0, 1.0),
    )
    reference = np.asarray([0.55, 0.0, 0.5])

    position, quaternion = sampler.sample(
        (obj,),
        rng=np.random.default_rng(0),
        reference_pos=reference,
    )[obj.name]

    np.testing.assert_allclose(position, [0.45, 0.02, 0.532])
    np.testing.assert_allclose(quaternion, [np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)])


def test_fixed_placement_can_replay_complete_world_pose() -> None:
    expected_quaternion = np.asarray([0.5, 0.5, -0.5, 0.5])
    sampler = FixedTablePlacementSampler(
        xy=(-0.1, 0.02),
        quaternion_wxyz=expected_quaternion,
        world_z=0.537,
    )
    obj = FreeBoxSpec(
        name="object",
        half_size=(0.02, 0.02, 0.03),
        rgba=(1.0, 1.0, 1.0, 1.0),
    )

    position, quaternion = sampler.sample(
        (obj,), rng=np.random.default_rng(0), reference_pos=np.asarray([0.55, 0.0, 0.5])
    )["object"]

    np.testing.assert_allclose(position, [0.45, 0.02, 0.537])
    np.testing.assert_allclose(quaternion, expected_quaternion)


def test_pick_place_regions_are_painted_sites_not_fake_collision_walls() -> None:
    env = make_pick_place_env(
        task_config={"object_id": "ycb:005_tomato_soup_can"},
        enable_tactile_sensors=False,
    )
    try:
        geom_names = {
            mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_GEOM, index)
            for index in range(env.model.ngeom)
        }
        site_names = {
            mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_SITE, index)
            for index in range(env.model.nsite)
        }
        assert not any(name and "source_bin" in name for name in geom_names)
        assert not any(name and "target_bin" in name for name in geom_names)
        assert {"source_region", "target_region"} <= site_names
    finally:
        env.close()
