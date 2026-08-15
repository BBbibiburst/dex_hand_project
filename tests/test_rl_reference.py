import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from source.rl.reference import EpisodeRecord, ReferenceTrajectory, STAGE_CODES, resolve_reference_manifest
from source.rl.trajectory import ResidualTrajectory


def _arrays() -> dict[str, np.ndarray]:
    frames = 6
    return {
        "qpos": np.arange(frames * 5, dtype=np.float32).reshape(frames, 5),
        "qvel": np.arange(frames * 4, dtype=np.float32).reshape(frames, 4),
        "ctrl": np.arange(frames * 13, dtype=np.float32).reshape(frames, 13),
        "action": np.zeros((frames, 13), dtype=np.float32),
        "object_position": np.stack(
            [np.asarray([0.5, 0.0, 0.6 + 0.01 * i], dtype=np.float32) for i in range(frames)]
        ),
        "object_quaternion_wxyz": np.tile(np.asarray([1.0, 0.0, 0.0, 0.0]), (frames, 1)),
        "stage": np.asarray(
            [
                STAGE_CODES["pregrasp"],
                STAGE_CODES["approach"],
                STAGE_CODES["approach"],
                STAGE_CODES["close"],
                STAGE_CODES["lift"],
                STAGE_CODES["verify"],
            ],
            dtype=np.int16,
        ),
        "reward": np.zeros(frames, dtype=np.float32),
        "task_success": np.zeros(frames, dtype=bool),
    }


def _write_episode(directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    arrays = _arrays()
    np.savez_compressed(directory / "episode.npz", **arrays)
    manifest = directory / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "pipeline": "ultradexgrasp-rm75b-dex-hand-v1",
                "object_id": "ycb:test",
                "seed": 0,
                "success": False,
                "terminal_stage": "verify",
                "arrays": "episode.npz",
                "candidate": {},
                "metadata": {"control_dt": 0.05},
            }
        ),
        encoding="utf-8",
    )
    return manifest


def _fake_env():
    arm = SimpleNamespace(
        actuator_ids=np.arange(7, dtype=np.int32),
        ctrl_low=np.full(7, -2.0, dtype=np.float32),
        ctrl_high=np.full(7, 2.0, dtype=np.float32),
        action_size=7,
    )
    hand = SimpleNamespace(
        actuator_ids=np.arange(7, 13, dtype=np.int32),
        ctrl_low=np.zeros(6, dtype=np.float32),
        ctrl_high=np.ones(6, dtype=np.float32),
    )
    controller = SimpleNamespace(
        arm_controller=arm,
        hand_controller=hand,
        actuator_names=tuple(f"a{i}" for i in range(13)),
    )
    return SimpleNamespace(
        controller=controller,
        model=SimpleNamespace(nq=5, nv=4, nu=13),
        config=SimpleNamespace(control_dt=0.05),
    )


def test_reference_extracts_low_level_ctrl_from_first_approach_frame(tmp_path) -> None:
    manifest = _write_episode(tmp_path / "episode")
    episode = EpisodeRecord.load(manifest)
    reference = ReferenceTrajectory.from_episode(
        episode,
        _fake_env(),
        source_manifest=manifest,
        start_stage="approach",
    )
    assert reference.horizon == 5
    assert reference.action_dim == 13
    assert reference.hand_action_size == 6
    np.testing.assert_array_equal(reference.initial_qpos, episode.arrays["qpos"][0])
    np.testing.assert_array_equal(reference.controls[0], episode.arrays["ctrl"][1])


def test_residual_trajectory_round_trip(tmp_path) -> None:
    trajectory = ResidualTrajectory(
        object_id="ycb:test",
        source_manifest="episode/manifest.json",
        start_stage="approach",
        action_mode="hand",
        residual_actions=np.zeros((4, 6), dtype=np.float32),
        controls=np.zeros((4, 13), dtype=np.float32),
        initial_qpos=np.zeros(5, dtype=np.float32),
        initial_qvel=np.zeros(4, dtype=np.float32),
        success=True,
        episode_return=12.5,
        metadata={"note": "test"},
    )
    manifest = trajectory.save(tmp_path / "trajectory")
    loaded = ResidualTrajectory.load(manifest)
    assert loaded.success
    assert loaded.episode_return == 12.5
    np.testing.assert_array_equal(loaded.residual_actions, trajectory.residual_actions)
    np.testing.assert_array_equal(loaded.controls, trajectory.controls)


def test_reference_resolver_selects_full_failed_attempt(tmp_path) -> None:
    output = tmp_path / "ultra"
    attempt = output / "attempts" / "rank_00_seed_000"
    manifest = _write_episode(attempt)
    resolved = resolve_reference_manifest(output)
    assert resolved == manifest
