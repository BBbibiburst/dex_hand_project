"""Geometry contracts for the procedural fixed-nut benchmark."""

from __future__ import annotations

import mujoco
import numpy as np
import pytest

from source.envs.manipulation import make_nut_assembly_env
from source.envs.manipulation.nut_assembly import procedural_nut_variant
from source.verification.task_stability import replay_nut_terminal_stability


def test_procedural_nut_catalogue_covers_ten_shapes_and_ten_families() -> None:
    variants = [procedural_nut_variant(index) for index in range(100)]

    assert {variant.polygon_sides for variant in variants} == {
        0,
        3,
        4,
        5,
        6,
        7,
        8,
        10,
        12,
        16,
    }
    family_dimensions = {
        (
            round(
                variant.hole_apothem
                / (1.0 if variant.polygon_sides == 0 else np.cos(np.pi / variant.polygon_sides)),
                6,
            ),
            variant.tube_radius,
        )
        for variant in variants
    }
    assert len(family_dimensions) == 10
    assert all(variant.peg_radius < variant.ring_radius for variant in variants)


@pytest.mark.parametrize("sides", (3, 5, 6, 7))
def test_corner_prone_profiles_receive_additional_running_clearance(sides: int) -> None:
    variants = [
        procedural_nut_variant(index)
        for index in range(10)
        if procedural_nut_variant(index).polygon_sides == sides
    ]

    assert len(variants) == 1
    assert variants[0].clearance == pytest.approx(0.0045)


def test_procedural_nut_dimensions_are_human_hand_scale() -> None:
    variants = [procedural_nut_variant(index) for index in range(100)]

    hole_diameters = [2.0 * variant.hole_apothem for variant in variants]
    shaft_diameters = [2.0 * variant.peg_radius for variant in variants]
    assert min(hole_diameters) >= 0.017
    assert max(hole_diameters) <= 0.042
    assert min(shaft_diameters) >= 0.024
    assert max(shaft_diameters) <= 0.038


def test_fixed_nut_has_physical_lower_walls_and_upper_guides() -> None:
    env = make_nut_assembly_env(
        task_config={"variant_index": 95},
        enable_tactile_sensors=False,
    )
    try:
        geom_names = {
            mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_GEOM, index)
            for index in range(env.model.ngeom)
        }
        assert {f"fixed_nut_wall_{index:02d}" for index in range(7)} <= geom_names
        assert {f"fixed_nut_guide_{index:02d}" for index in range(7)} <= geom_names

        bolt = env.task.objects[0]
        assert bolt.bottom_offset == pytest.approx(0.049)
        assert bolt.handle_center_z - bolt.handle_half_height == pytest.approx(
            bolt.shoulder_center_z + bolt.shoulder_half_height
        )
        assert bolt.shoulder_radius == pytest.approx(max(0.030, bolt.shaft_radius + 0.006))
        assert 2.0 * bolt.handle_radius == pytest.approx(0.048)
        assert 2.0 * bolt.handle_half_height == pytest.approx(0.072)
        assert 2.0 * env.task.arena.wall_half_height == pytest.approx(0.050)

        socket_top = env.task.table_top_z + 2.0 * env.task.arena.wall_half_height
        seated_body_z = socket_top - bolt.shoulder_center_z + bolt.shoulder_half_height - 0.001
        seated_tip_z = seated_body_z - bolt.bottom_offset
        assert seated_tip_z >= env.task.table_top_z + 0.002
        assert socket_top - seated_tip_z >= 0.038
    finally:
        env.close()


def test_nut_stability_replay_validates_window_arguments(tmp_path) -> None:
    with pytest.raises(ValueError, match="tail_steps"):
        replay_nut_terminal_stability(tmp_path, hold_steps=20, tail_steps=21)


def test_nut_stability_replay_rejects_another_task(tmp_path) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        '{"success": true, "metadata": {"task": "stack"}}',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="nut_assembly"):
        replay_nut_terminal_stability(manifest)
