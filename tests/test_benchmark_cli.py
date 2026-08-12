"""CLI preset and benchmark progress formatting contracts."""

from tools.grasping.benchmark_catalog import _apply_full_pipeline_preset
from source.workflows.grasp_benchmark import _format_duration


def test_full_pipeline_keeps_explicit_parallelism() -> None:
    values = {"jobs": 4, "evolution_jobs": 1}

    _apply_full_pipeline_preset(values, {"jobs", "evolution_jobs"})

    assert values["jobs"] == 4
    assert values["evolution_jobs"] == 1
    assert values["generator"] == "graspqp"


def test_duration_format_is_compact() -> None:
    assert _format_duration(19.7) == "20s"
    assert _format_duration(125) == "2m05s"
    assert _format_duration(7500) == "2h05m"
