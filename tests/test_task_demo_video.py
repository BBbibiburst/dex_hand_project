"""Tests for generic five-task demonstration video inputs."""

from __future__ import annotations

import json

import numpy as np
import pytest

from source.viz.task_demo_video import load_recorded_task_episode


def _manifest(tmp_path, *, task: str = "pick_place"):
    arrays = tmp_path / "episode.npz"
    np.savez_compressed(
        arrays,
        qpos=np.zeros((3, 2)),
        qvel=np.zeros((3, 2)),
        ctrl=np.zeros((3, 1)),
        stage=np.asarray([0, 1, 1]),
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "object_id": "ycb:test",
                "arrays": arrays.name,
                "metadata": {
                    "task": task,
                    "task_config": {"reward_shaping": True},
                    "stage_codes": {"settle": 0, "move": 1},
                },
            }
        ),
        encoding="utf-8",
    )
    return manifest


def test_loader_infers_task_object_and_stage_names(tmp_path) -> None:
    episode = load_recorded_task_episode(_manifest(tmp_path))

    assert episode.task == "pick_place"
    assert episode.env_config == {}
    assert episode.object_id == "ycb:test"
    assert episode.camera is None
    assert episode.task_config == {
        "reward_shaping": True,
        "object_id": "ycb:test",
        "terminate_on_success": False,
    }
    assert episode.stage_names[0] == "settle"
    assert episode.stage_names[1] == "move"
    assert episode.stage_names[8] == "transport"


def test_loader_allows_task_and_config_override(tmp_path) -> None:
    episode = load_recorded_task_episode(
        _manifest(tmp_path),
        task="push",
        task_config_override={"reward_shaping": False},
    )

    assert episode.task == "push"
    assert episode.task_config["reward_shaping"] is False


def test_loader_rejects_unknown_task(tmp_path) -> None:
    with pytest.raises(ValueError, match="Unsupported task"):
        load_recorded_task_episode(_manifest(tmp_path, task="unknown"))


def test_loader_accepts_standalone_teleop_npz(tmp_path) -> None:
    path = tmp_path / "raw_0000.npz"
    metadata = {
        "schema_version": 1,
        "task": "lift",
        "object_id": "ycb:test",
        "task_config": {"object_id": "ycb:test"},
        "env_config": {"control_dt": 0.05},
    }
    np.savez_compressed(
        path,
        metadata_json=np.asarray(json.dumps(metadata)),
        observed_qpos=np.zeros((2, 3)),
        observed_qvel=np.zeros((2, 2)),
        observed_ctrl=np.zeros((2, 1)),
    )

    episode = load_recorded_task_episode(path)

    assert episode.task == "lift"
    assert episode.arrays["qpos"].shape == (2, 3)
    assert episode.env_config == {"control_dt": 0.05}
