"""Regression tests for resumable UltraDexGrasp catalogue generation."""

from __future__ import annotations

import json
from pathlib import Path

from tools.ultradexgrasp import batch_generate


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_catalog_selection_uses_all_lift_objects(monkeypatch) -> None:
    monkeypatch.setattr(
        batch_generate,
        "lift_object_ids",
        lambda: ("ycb:a", "ycb:b", "egad:C0", "egad:C1"),
    )

    assert batch_generate._selected_object_ids(
        dataset="all", requested=None, limit=None
    ) == ["ycb:a", "ycb:b", "egad:C0", "egad:C1"]
    assert batch_generate._selected_object_ids(
        dataset="egad", requested=None, limit=1
    ) == ["egad:C0"]
    assert batch_generate._selected_object_ids(
        dataset="original127", requested=None, limit=None
    ) == ["ycb:a", "ycb:b", "egad:C0", "egad:C1"]


def test_success_manifest_is_recovered_after_parent_interruption(tmp_path: Path) -> None:
    attempt = tmp_path / "seed_0003"
    _write_json(
        attempt / "manifest.json",
        {
            "object_id": "ycb:test",
            "seed": 3,
            "success": True,
        },
    )

    recovered = batch_generate._completed_attempt_from_disk(
        attempt,
        object_id="ycb:test",
        seed=3,
    )

    assert recovered is not None
    assert recovered["status"] == "success"
    assert recovered["recovered"] is True


def test_structured_failure_type_is_recovered(tmp_path: Path) -> None:
    attempt = tmp_path / "seed_0004"
    _write_json(
        attempt / "run.json",
        {
            "object_id": "ycb:test",
            "seed": 4,
            "success": False,
            "failure_type": "ik_unreachable",
            "attempts": [],
        },
    )

    recovered = batch_generate._completed_attempt_from_disk(
        attempt,
        object_id="ycb:test",
        seed=4,
    )

    assert recovered is not None
    assert recovered["status"] == "ik_unreachable"


def test_no_valid_candidate_falls_back_to_candidates_archive(tmp_path: Path) -> None:
    attempt = tmp_path / "seed_0005"
    _write_json(
        attempt / "candidates.json",
        {
            "candidate_count": 32,
            "valid_candidate_count": 0,
        },
    )

    status, manifest, error = batch_generate._classify_child_result(
        attempt,
        object_id="ycb:test",
        seed=5,
        return_code=2,
        timed_out=False,
    )

    assert status == "no_valid_candidates"
    assert manifest is None
    assert error is None


def test_timeout_has_priority_over_partial_files(tmp_path: Path) -> None:
    attempt = tmp_path / "seed_0006"
    _write_json(
        attempt / "run.json",
        {
            "object_id": "ycb:test",
            "seed": 6,
            "success": False,
            "failure_type": "execution_failed",
        },
    )

    status, _, _ = batch_generate._classify_child_result(
        attempt,
        object_id="ycb:test",
        seed=6,
        return_code=None,
        timed_out=True,
    )

    assert status == "timeout"


def test_summary_reports_object_coverage_and_failure_types() -> None:
    state = {
        "objects": [
            {
                "object_id": "ycb:a",
                "status": "success",
                "attempts": [
                    {"seed": 0, "status": "no_valid_candidates"},
                    {"seed": 1, "status": "success"},
                ],
            },
            {
                "object_id": "egad:C0",
                "status": "failed",
                "attempts": [
                    {"seed": 0, "status": "ik_unreachable"},
                    {"seed": 1, "status": "execution_failed"},
                ],
            },
            {
                "object_id": "egad:C1",
                "status": "pending",
                "attempts": [],
            },
        ]
    }

    summary = batch_generate._state_summary(state)

    assert summary["selected_objects"] == 3
    assert summary["completed_objects"] == 2
    assert summary["successful_objects"] == 1
    assert summary["failed_objects"] == 1
    assert summary["pending_objects"] == 1
    assert summary["object_coverage_rate"] == 1 / 3
    assert summary["total_attempts"] == 4
    assert summary["final_failure_types"] == {"execution_failed": 1}
