"""Contracts for transferring reusable Lift grasps to downstream tasks."""

from __future__ import annotations

import json

import numpy as np
import pytest

from source.grasping.task_transfer import PickPlaceTransferConfig, _combine_arrays
from tools.grasping.transfer_lift_to_pick_place import discover_lift_manifests


def test_pick_place_transfer_requires_bin_clearance() -> None:
    with pytest.raises(ValueError, match="bin walls"):
        PickPlaceTransferConfig(clearance_height=0.10).validate()


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

    assert discover_lift_manifests(None, [tmp_path], object_id="ycb:test") == (
        good.resolve(),
    )
