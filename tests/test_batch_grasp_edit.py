"""Scheduling and ETA contracts for the full grasp-edit catalogue runner."""

from __future__ import annotations

import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from types import SimpleNamespace

import pytest

from source.rl.grasp_edit import templates as grasp_edit_templates
from tools.grasping.batch_grasp_edit import (
    _dataset_ids,
    _format_duration,
    _format_progress,
    _gpu_ids,
    _init_object_worker,
    _resource_plan,
    _runtime_estimate,
    _wall_clock_estimate,
    _worker_gpu_identity,
    _write_summary,
    build_parser,
)


def test_historical_benchmark_selection_is_stable_at_127_objects() -> None:
    selected = _dataset_ids("original127")

    assert len(selected) == 127
    assert sum(item.startswith("ycb:") for item in selected) == 78
    assert sum(item.startswith("egad:") for item in selected) == 49
    assert not any(item.startswith("gso:") for item in selected)


def test_benchmark_parser_defaults_to_historical_127() -> None:
    args = build_parser().parse_args([])

    assert args.dataset == "original127"
    assert args.expect_count == 127


def test_ultra_discovery_accepts_batch_generate_layout(tmp_path, monkeypatch) -> None:
    manifest = (
        tmp_path
        / "objects"
        / "ycb_008_pudding_box"
        / "seed_0000"
        / "manifest.json"
    )
    manifest.parent.mkdir(parents=True)
    manifest.touch()
    episode = SimpleNamespace(
        success=False,
        metadata={
            "object_lift": 0.01,
            "approach_position_error": 0.0,
            "approach_orientation_error": 0.0,
        },
    )
    monkeypatch.setattr(grasp_edit_templates, "_full_episode", lambda path, object_id: episode)

    rows = grasp_edit_templates.discover_ultra_attempts(
        "ycb:008_pudding_box", roots=(tmp_path,), maximum=1
    )

    assert rows == [(manifest.resolve(), episode)]


def _summary_args() -> SimpleNamespace:
    return SimpleNamespace(
        dataset="all",
        gpus="auto",
        workers_per_gpu="auto",
        gpu_jobs_per_gpu="auto",
        ppo_jobs_per_gpu="auto",
        num_envs=64,
        initial_updates=5,
        mid_updates=10,
        max_updates=15,
        ultra_seed_count=100,
        ultra_generate_seeds=3,
        base_candidates=3,
        lattice_max_templates=12,
        lattice_max_executions=32,
        lattice_root="lattice",
        promising_lift_mm=20.0,
        promising_success_rate=0.01,
        early_fail_lift_mm=10.0,
        continue_lift_mm=20.0,
        progress_gain_mm=5.0,
        train_ultra_success=True,
        train_lattice_success=True,
    )


def test_auto_gpu_selection_prefers_cuda_visible_devices(monkeypatch) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,5")

    assert _gpu_ids("auto", device="cuda:0") == ("2", "5")


def test_auto_gpu_selection_falls_back_to_device_index(monkeypatch) -> None:
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)

    assert _gpu_ids("auto", device="cuda:3") == ("3",)
    assert _gpu_ids("auto", device="cpu") == ("cpu",)


def test_auto_gpu_selection_rejects_hidden_cuda_devices(monkeypatch) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "-1")

    with pytest.raises(ValueError, match="hides every GPU"):
        _gpu_ids("auto", device="cuda:0")


def test_resource_plan_uses_two_pipeline_workers_on_a_24_gib_gpu(monkeypatch) -> None:
    monkeypatch.setattr("tools.grasping.batch_grasp_edit.os.cpu_count", lambda: 16)
    slots, plan = _resource_plan(
        ("0",),
        workers_per_gpu="auto",
        gpu_jobs_per_gpu="auto",
        ppo_jobs_per_gpu="auto",
        num_envs=64,
        resource_rows={
            "0": {
                "total_memory_mb": 24576.0,
                "free_memory_mb": 22500.0,
                "utilization_percent": 99.0,
            }
        },
    )

    assert slots == ("0", "0")
    assert plan["gpu_details"]["0"]["workers"] == 2
    assert plan["gpu_details"]["0"]["gpu_jobs"] == 1
    assert plan["gpu_details"]["0"]["ppo_jobs"] == 1
    assert plan["gpu_details"]["0"]["estimated_worker_memory_mb"] == 2048.0


def test_resource_plan_runs_two_gpu_jobs_when_a_24_gib_gpu_is_underused(
    monkeypatch,
) -> None:
    monkeypatch.setattr("tools.grasping.batch_grasp_edit.os.cpu_count", lambda: 16)
    slots, plan = _resource_plan(
        ("0",),
        workers_per_gpu="auto",
        gpu_jobs_per_gpu="auto",
        ppo_jobs_per_gpu="auto",
        num_envs=64,
        resource_rows={
            "0": {
                "total_memory_mb": 24576.0,
                "free_memory_mb": 22500.0,
                "utilization_percent": 8.0,
            }
        },
    )

    assert slots == ("0", "0")
    assert plan["gpu_job_limits"] == {"0": 2}
    assert plan["ppo_job_limits"] == {"0": 1}


def test_resource_plan_reduces_workers_when_memory_is_tight() -> None:
    slots, plan = _resource_plan(
        ("0",),
        workers_per_gpu="auto",
        gpu_jobs_per_gpu="auto",
        ppo_jobs_per_gpu="auto",
        num_envs=64,
        resource_rows={
            "0": {
                "total_memory_mb": 4096.0,
                "free_memory_mb": 3000.0,
                "utilization_percent": 0.0,
            }
        },
    )

    assert slots == ("0",)
    assert plan["gpu_details"]["0"]["workers"] == 1
    assert plan["gpu_details"]["0"]["gpu_jobs"] == 1
    assert plan["gpu_details"]["0"]["ppo_jobs"] == 1


def test_resource_plan_interleaves_multiple_gpus() -> None:
    slots, _ = _resource_plan(
        ("0", "1"),
        workers_per_gpu="2",
        gpu_jobs_per_gpu="1",
        ppo_jobs_per_gpu="1",
        num_envs=64,
        resource_rows={},
    )

    assert slots == ("0", "1", "0", "1")


def test_eta_uses_observed_object_runtime_and_parallel_slot_count() -> None:
    rows = ({"runtime_sec": 340.725}, {"runtime_sec": 345.769})

    average, single_gpu_eta, samples = _runtime_estimate(
        rows,
        remaining=125,
        worker_count=1,
    )
    _, pipelined_eta, _ = _runtime_estimate(rows, remaining=125, worker_count=2)

    assert average == pytest.approx(343.247)
    assert single_gpu_eta == pytest.approx(42905.875)
    assert pipelined_eta == pytest.approx(21452.9375)
    assert samples == 2
    assert _format_duration(single_gpu_eta) == "11h55m"


def test_wall_clock_eta_waits_for_pipeline_warmup() -> None:
    assert _wall_clock_estimate(
        elapsed=300.0,
        completed=1,
        remaining=10,
        warmup_completions=2,
    ) == (None, None, 1)

    average, eta, samples = _wall_clock_estimate(
        elapsed=600.0,
        completed=2,
        remaining=10,
        warmup_completions=2,
    )
    assert average == 300.0
    assert eta == 3000.0
    assert samples == 2


def test_progress_line_includes_gpu_runtime_average_and_eta() -> None:
    line = _format_progress(
        3,
        127,
        {
            "object_id": "ycb:test",
            "status": "RL_SUCCESS",
            "ultra_success": False,
            "lattice_templates": 12,
            "rl_best_success_rate": 0.047,
            "rl_updates": 5,
            "rl_best_lift_mm": 64.0,
            "runtime_sec": 340.7,
            "gpu": "0",
        },
        average_runtime=343.247,
        eta=21452.9375,
    )

    assert "gpu=0" in line
    assert "object=5m41s" in line
    assert "avg=5m43s/obj" in line
    assert "eta=5h57m" in line


def test_summary_persists_partial_progress_and_eta(tmp_path) -> None:
    rows = [
        {"object_id": "a", "status": "RL_SUCCESS", "runtime_sec": 300.0},
        {"object_id": "b", "status": "DIRECT_FAILED", "runtime_sec": 420.0},
    ]
    _write_summary(
        tmp_path,
        rows=rows,
        args=_summary_args(),
        signature="test",
        source_hashes={},
        total_count=10,
        worker_count=2,
        gpus=("0",),
        resource_plan={"worker_slots": ["0", "0"]},
    )

    payload = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert payload["schema_version"] == 2
    assert payload["selected_objects"] == 10
    assert payload["completed_objects"] == 2
    assert payload["progress"]["remaining_objects"] == 8
    assert payload["progress"]["worker_slots"] == 2
    assert payload["progress"]["average_runtime_sec"] == 360.0
    assert payload["progress"]["estimated_remaining_sec"] == 1440.0


def test_parallel_parser_defaults_to_auto_resource_planning() -> None:
    args = build_parser().parse_args([])

    assert args.gpus == "auto"
    assert args.workers_per_gpu == "auto"
    assert args.gpu_jobs_per_gpu == "auto"
    assert args.ppo_jobs_per_gpu == "auto"


def test_spawned_worker_receives_one_fixed_gpu_slot() -> None:
    context = multiprocessing.get_context("spawn")
    with context.Manager() as manager:
        queue = manager.Queue()
        queue.put("7")
        semaphores = {"7": manager.BoundedSemaphore(1)}
        ppo_semaphores = {"7": manager.BoundedSemaphore(1)}
        with ProcessPoolExecutor(
            max_workers=1,
            mp_context=context,
            initializer=_init_object_worker,
            initargs=(queue, semaphores, ppo_semaphores),
        ) as executor:
            assert executor.submit(_worker_gpu_identity).result(timeout=10) == ("7", "7")
