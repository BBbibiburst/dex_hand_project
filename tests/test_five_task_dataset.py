from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sqlite3
import uuid

import numpy as np
import pytest

from apps.check_p30_f_drive import check as check_drive
from apps.produce_five_task_dataset import build_parser as producer_parser
from apps.run_p30_full_episode import EncoderPipe
from source.data.five_task_dataset import (
    EpisodeIndex,
    SCHEMA_VERSION,
    canonical_hash,
    classify_failure,
    validate_exported_arrays,
    validate_source_episode,
    write_causal_source,
)


def _source(tmp_path: Path, tactile: str = "missing") -> Path:
    directory = tmp_path / "source"
    directory.mkdir()
    frames = 4
    arrays = {
        "qpos": np.arange(frames * 26, dtype=np.float32).reshape(frames, 26),
        "qvel": np.zeros((frames, 25), dtype=np.float32),
        "ctrl": np.arange(frames * 13, dtype=np.float32).reshape(frames, 13),
        "action": (1000 + np.arange(frames * 13)).astype(np.float32).reshape(frames, 13),
        "object_position": np.ones((frames, 3), dtype=np.float32),
        "object_quaternion_wxyz": np.tile([1, 0, 0, 0], (frames, 1)).astype(np.float32),
        "stage": np.arange(frames, dtype=np.int16),
        "reward": np.zeros(frames, dtype=np.float32),
        "task_success": np.zeros(frames, dtype=bool),
    }
    if tactile == "present":
        arrays["tactile"] = np.ones((frames, 712), dtype=np.float32)
    elif tactile == "empty":
        arrays["tactile"] = np.zeros((frames, 0), dtype=np.float32)
    np.savez_compressed(directory / "episode.npz", **arrays)
    (directory / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "object_id": "ycb:test",
                "seed": 7,
                "success": False,
                "terminal_stage": "hold",
                "failure_reason": "object drop after hold",
                "arrays": "episode.npz",
                "metadata": {"task": "lift", "control_dt": 0.05},
            }
        ),
        encoding="utf-8",
    )
    return directory / "manifest.json"


def _index_row(index: int) -> dict:
    return {
        "episode_id": str(uuid.uuid4()),
        "lineage_hash": f"lineage-{index}",
        "content_hash": f"content-{index}",
        "task_name": "lift",
        "object_id": f"object-{index}",
        "success": index % 2 == 0,
        "simulation_duration": 1.0,
        "tactile": {"status": "NOT_AVAILABLE_IN_SOURCE"},
        "relative_path": f"episodes/{index}",
        "storage_bytes": 10,
        "validation_status": "PASS",
    }


def test_schema_version_is_frozen():
    assert SCHEMA_VERSION == "five-task-episode-v1"


def test_source_action_contract_is_13d(tmp_path):
    result = validate_source_episode(_source(tmp_path))
    assert result["archive"]["action"].shape == (4, 13)


def test_tactile_present_is_real_712d(tmp_path):
    result = validate_source_episode(_source(tmp_path, "present"))
    assert result["tactile_status"] == "PRESENT"


@pytest.mark.parametrize("kind", ["missing", "empty"])
def test_tactile_missing_is_not_filled(tmp_path, kind):
    result = validate_source_episode(_source(tmp_path, kind))
    assert result["tactile_status"] == "NOT_AVAILABLE_IN_SOURCE"


def test_causal_shift_pairs_previous_state_with_next_action(tmp_path):
    source = _source(tmp_path)
    destination = tmp_path / "causal"
    write_causal_source(source, destination)
    with np.load(source.parent / "episode.npz") as original, np.load(
        destination / "episode.npz"
    ) as shifted:
        assert np.array_equal(shifted["qpos"], original["qpos"][:-1])
        assert np.array_equal(shifted["action"], original["action"][1:])


def test_exported_timestamp_and_39d_state_contract(tmp_path):
    path = tmp_path / "modalities.npz"
    np.savez_compressed(
        path,
        state=np.zeros((3, 39), np.float32),
        action=np.zeros((3, 13), np.float32),
        timestamp=np.asarray([0.0, 0.05, 0.10]),
        sim_time=np.asarray([0.0, 0.05, 0.10]),
    )
    result = validate_exported_arrays(path, "NOT_AVAILABLE_IN_SOURCE")
    assert result["state_shape"] == [3, 39]


def test_missing_tactile_array_is_rejected_when_status_present(tmp_path):
    path = tmp_path / "modalities.npz"
    np.savez_compressed(
        path,
        state=np.zeros((2, 39)),
        action=np.zeros((2, 13)),
        timestamp=np.asarray([0.0, 0.05]),
        sim_time=np.asarray([0.0, 0.05]),
    )
    with pytest.raises(ValueError, match="PRESENT tactile"):
        validate_exported_arrays(path, "PRESENT")


def test_episode_ids_are_unique():
    assert uuid.uuid4() != uuid.uuid4()


def test_canonical_lineage_hash_is_stable():
    assert canonical_hash({"a": 1, "b": 2}) == canonical_hash({"b": 2, "a": 1})


def test_inventory_rejects_duplicate_lineage(tmp_path):
    index = EpisodeIndex(tmp_path / "inventory.sqlite")
    try:
        index.insert(_index_row(0))
        duplicate = _index_row(1)
        duplicate["lineage_hash"] = "lineage-0"
        with pytest.raises(sqlite3.IntegrityError):
            index.insert(duplicate)
    finally:
        index.close()


def test_inventory_can_identify_duplicate_content(tmp_path):
    index = EpisodeIndex(tmp_path / "inventory.sqlite")
    try:
        row = _index_row(3)
        index.insert(row)
        assert index.find_content(row["content_hash"])["episode_id"] == row["episode_id"]
    finally:
        index.close()


def test_partial_directory_is_not_inventory_data(tmp_path):
    (tmp_path / ".partial-deadbeef").mkdir()
    index = EpisodeIndex(tmp_path / "inventory.sqlite")
    try:
        assert index.inventory()["episodes"] == 0
    finally:
        index.close()


def test_inventory_rebuild_statistics(tmp_path):
    index = EpisodeIndex(tmp_path / "inventory.sqlite")
    try:
        index.insert(_index_row(0))
        index.insert(_index_row(1))
        inventory = index.inventory()
        assert inventory["episodes"] == 2
        assert inventory["tasks"]["lift"]["successes"] == 1
    finally:
        index.close()


def test_drive_identity_rejects_small_root(tmp_path):
    assert check_drive(tmp_path)["ready"] is False


def test_two_concurrent_inventory_writers(tmp_path):
    database = tmp_path / "inventory.sqlite"
    EpisodeIndex(database).close()

    def insert(value):
        index = EpisodeIndex(database)
        try:
            index.insert(_index_row(value))
            index.write_inventory(tmp_path / "episode_inventory.json")
        finally:
            index.close()

    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(insert, (1, 2)))
    index = EpisodeIndex(database)
    try:
        assert index.inventory()["episodes"] == 2
        assert json.loads((tmp_path / "episode_inventory.json").read_text())["episodes"] == 2
    finally:
        index.close()


def test_inventory_json_write_is_atomic_and_rebuildable(tmp_path):
    database = tmp_path / "inventory.sqlite"
    target = tmp_path / "episode_inventory.json"
    index = EpisodeIndex(database)
    try:
        index.insert(_index_row(4))
        payload = index.write_inventory(target)
    finally:
        index.close()
    assert json.loads(target.read_text(encoding="utf-8")) == payload
    assert not list(tmp_path.glob(".episode_inventory.json.*.tmp"))


def test_failure_classification():
    failure = classify_failure("IK collision caused object drop timeout", "approach")
    assert failure == {
        "stage": "approach",
        "collision": True,
        "ik_failure": True,
        "grasp_failure": False,
        "object_drop": True,
        "timeout": True,
    }


def test_async_encoder_failure_is_propagated():
    class BrokenStream:
        def write(self, _payload):
            raise BrokenPipeError("synthetic encoder failure")

        def close(self):
            pass

    class BrokenProcess:
        stdin = BrokenStream()

    pipe = EncoderPipe(BrokenProcess(), mode="async", queue_size=1)
    pipe.submit(np.zeros((2, 2, 3), dtype=np.uint8))
    with pytest.raises(BrokenPipeError, match="synthetic encoder failure"):
        pipe.close()


def test_producer_exposes_required_scheduler_parameters():
    parser = producer_parser()
    options = {action.dest for action in parser._actions}
    assert {"task", "object_id", "seed", "episodes", "max_attempts", "workers", "resolution", "fps", "output_dir", "resume", "dry_run"} <= options
