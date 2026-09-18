"""Task-valid selection and five-task reporting contracts."""

from __future__ import annotations

import json
from pathlib import Path

from source.workflows.benchmark_report import catalogue_task_metrics, fixed_domain_metrics
from source.workflows.task_domains import (
    assess_task_domain,
    build_task_domain_manifest,
    is_prevalidated_task_manifest,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def geometry(**overrides):
    values = {
        "extents_m": (0.04, 0.05, 0.08),
        "sphericity": 0.75,
        "convexity": 0.90,
        "axis_ratio": 2.0,
    }
    values.update(overrides)
    return values


def stability(**overrides):
    values = {
        "stable": True,
        "error": None,
        "horizontal_displacement_m": 0.001,
        "orientation_change_deg": 1.0,
    }
    values.update(overrides)
    return values


def test_stack_and_push_rules_are_independent_of_policy_outcomes() -> None:
    stack = assess_task_domain("stack", "sample", geometry(), stability())
    ball = assess_task_domain("push", "ball", geometry(sphericity=0.98), stability())

    assert stack.eligible
    assert not ball.eligible
    assert "too_round_for_quasi_static_face_push" in ball.reasons


def test_manifest_keeps_auditable_exclusions() -> None:
    selection = {
        "objects": [
            {"object_id": "box", "geometry": geometry()},
            {"object_id": "ball", "geometry": geometry(sphericity=0.99)},
        ]
    }
    report = {
        "selection": "source.json",
        "results": [
            {"object_id": "box", **stability()},
            {"object_id": "ball", **stability()},
        ],
    }

    manifest = build_task_domain_manifest("stack", selection, report)

    assert manifest["count"] == 1
    assert manifest["excluded_count"] == 1
    assert manifest["excluded"][0]["object_id"] == "ball"
    assert is_prevalidated_task_manifest(manifest, "stack")
    assert not is_prevalidated_task_manifest(manifest, "push")


def test_committed_domains_have_intended_population_sizes() -> None:
    stack = json.loads((PROJECT_ROOT / "configs/stack_task_valid_v1.json").read_text())
    push = json.loads((PROJECT_ROOT / "configs/push_task_valid_v1.json").read_text())

    assert stack["source_count"] == 100
    assert 30 <= stack["count"] <= 50
    assert push["source_count"] == 100
    assert 70 <= push["count"] <= 90


def test_report_separates_task_valid_success_from_catalogue_coverage() -> None:
    domain = {
        "source_count": 4,
        "excluded_count": 2,
        "objects": [{"object_id": "a"}, {"object_id": "b"}],
        "excluded": [{"object_id": "c"}, {"object_id": "d"}],
    }
    valid_summary = {
        "results": [
            {"object_id": "a", "status": "SUCCESS"},
            {"object_id": "b", "status": "FAILED"},
        ]
    }
    full_summary = {
        "results": [
            {"object_id": "a", "status": "SUCCESS"},
            {"object_id": "b", "status": "FAILED"},
            {"object_id": "c", "status": "SUCCESS"},
            {"object_id": "d", "status": "FAILED"},
        ]
    }

    metrics = catalogue_task_metrics(
        domain, valid_summary, full_catalogue_summary=full_summary
    )

    assert metrics["task_valid_success_rate"] == 0.5
    assert metrics["full_catalogue_coverage"] == 0.5
    assert fixed_domain_metrics(
        {"count": 2, "results": [{"status": "LATTICE_SUCCESS"}, {"status": "FAILED"}]},
        domain="top100_catalogue",
    )["successes"] == 1
