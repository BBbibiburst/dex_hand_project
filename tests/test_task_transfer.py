"""Contracts for transferring reusable Lift grasps to downstream tasks."""

from __future__ import annotations

import json

import numpy as np
import pytest

from source.grasping.task_transfer import (
    PickPlaceTransferConfig,
    _combine_arrays,
    _safe_descent_distance,
)
from tools.grasping.transfer_lift_to_pick_place import discover_lift_manifests
from tools.grasping.batch_pick_place_transfer import _selection_ids, _source_roots


def test_pick_place_transfer_requires_positive_clearance() -> None:
    with pytest.raises(ValueError, match="positive"):
        PickPlaceTransferConfig(clearance_height=0.0).validate()


def test_supported_transport_reserves_an_extra_descent_segment() -> None:
    aerial = PickPlaceTransferConfig()
    supported = PickPlaceTransferConfig(table_supported_transport=True)

    assert supported.appended_steps == aerial.appended_steps + aerial.descend_steps


def test_descent_stops_before_hand_reaches_table() -> None:
    distance, limited = _safe_descent_distance(
        object_position_z=0.70,
        object_bottom_offset=0.04,
        table_top_z=0.50,
        desired_object_clearance=0.002,
        hand_min_z=0.58,
        hand_table_clearance=0.006,
    )

    assert distance == pytest.approx(0.074)
    assert limited


def test_descent_places_object_when_hand_has_clearance() -> None:
    distance, limited = _safe_descent_distance(
        object_position_z=0.70,
        object_bottom_offset=0.04,
        table_top_z=0.50,
        desired_object_clearance=0.002,
        hand_min_z=0.75,
        hand_table_clearance=0.006,
    )

    assert distance == pytest.approx(0.158)
    assert not limited


def test_transfer_array_combination_preserves_order() -> None:
    result = _combine_arrays({"action": np.asarray([[1]])}, {"action": np.asarray([[2]])})
    assert result["action"].tolist() == [[1], [2]]


def test_discovery_keeps_only_successful_episode_manifests(tmp_path) -> None:
    good = tmp_path / "good" / "manifest.json"
    bad = tmp_path / "bad" / "manifest.json"
    good.parent.mkdir()
    bad.parent.mkdir()
    good.write_text(
        json.dumps({"success": True, "object_id": "ycb:test", "candidate": {}}),
        encoding="utf-8",
    )
    bad.write_text(
        json.dumps({"success": False, "object_id": "ycb:test", "candidate": {}}),
        encoding="utf-8",
    )

    assert discover_lift_manifests(None, [tmp_path], object_id="ycb:test") == (good.resolve(),)


def test_discovery_accepts_successful_ppo_trajectory_manifest(tmp_path) -> None:
    path = tmp_path / "best_trajectory" / "manifest.json"
    path.parent.mkdir()
    path.write_text(
        json.dumps(
            {
                "success": True,
                "object_id": "gso:test",
                "action_mode": "grasp_edit_hybrid",
            }
        ),
        encoding="utf-8",
    )

    assert discover_lift_manifests(None, [tmp_path], object_id="gso:test") == (path.resolve(),)


def test_discovery_prioritizes_the_most_retained_lift(tmp_path) -> None:
    weak = tmp_path / "weak" / "manifest.json"
    strong = tmp_path / "strong" / "manifest.json"
    for path, lift in ((weak, 0.04), (strong, 0.08)):
        path.parent.mkdir()
        path.write_text(
            json.dumps(
                {
                    "success": True,
                    "object_id": "ycb:test",
                    "candidate": {},
                    "metadata": {"object_lift": lift},
                }
            ),
            encoding="utf-8",
        )

    assert discover_lift_manifests(None, [tmp_path], object_id="ycb:test") == (
        strong.resolve(),
        weak.resolve(),
    )


def test_batch_selection_and_source_roots(tmp_path) -> None:
    selection = tmp_path / "selection.json"
    selection.write_text(json.dumps({"objects": [{"object_id": "ycb:test"}]}), encoding="utf-8")
    lattice = tmp_path / "lattice"
    direct = lattice / "ycb_test"
    recovery = lattice / "recovery_lift_085mm" / "ycb_test"
    rl = tmp_path / "rl"
    grasp = tmp_path / "grasp"
    direct.mkdir(parents=True)
    recovery.mkdir(parents=True)

    assert _selection_ids(selection) == ("ycb:test",)
    assert _source_roots(lattice, rl, grasp, "ycb:test") == (direct, recovery)
