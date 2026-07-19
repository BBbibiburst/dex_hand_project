"""Regression coverage for publishing dynamically validated grasp configs."""

from __future__ import annotations

from pathlib import Path

import pytest

from source.grasping import grasp_config_search
from source.grasping.standalone_validator import StandaloneValidationResult


def _validation(*, stable: bool) -> StandaloneValidationResult:
    return StandaloneValidationResult(
        stable=stable,
        initial_displacement=0.0,
        position_drift=0.0 if stable else 0.02,
        rotation_drift=0.0,
        vertical_drop=0.0,
        initial_contacts=4,
        final_contacts=4 if stable else 0,
        simulated_seconds=0.1,
    )


def test_validated_grasp_publishes_first_stable_candidate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "grasp.json"
    searched_seeds: list[int] = []
    validations = iter((_validation(stable=False), _validation(stable=True)))

    def fake_search_grasp_config(*, output: Path, seed: int, **kwargs):
        _ = kwargs
        searched_seeds.append(seed)
        Path(output).write_text(f'{{"seed": {seed}}}', encoding="utf-8")

    monkeypatch.setattr(
        grasp_config_search,
        "search_grasp_config",
        fake_search_grasp_config,
    )
    monkeypatch.setattr(
        grasp_config_search,
        "validate_grasp_config",
        lambda *args, **kwargs: next(validations),
    )

    result = grasp_config_search.generate_validated_grasp_config(
        "ycb:test",
        output=output,
        attempts=3,
        seed=7,
    )

    assert searched_seeds == [7, 8]
    assert result.selected_seed == 8
    assert result.attempts_used == 2
    assert output.read_text(encoding="utf-8") == '{"seed": 8}'
    assert not output.with_suffix(".json.candidate").exists()


def test_failed_validation_preserves_existing_grasp(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "grasp.json"
    output.write_text('{"known": "stable"}', encoding="utf-8")

    def fake_search_grasp_config(*, output: Path, **kwargs):
        _ = kwargs
        Path(output).write_text('{"candidate": true}', encoding="utf-8")

    monkeypatch.setattr(
        grasp_config_search,
        "search_grasp_config",
        fake_search_grasp_config,
    )
    monkeypatch.setattr(
        grasp_config_search,
        "validate_grasp_config",
        lambda *args, **kwargs: _validation(stable=False),
    )

    with pytest.raises(RuntimeError, match="No dynamically stable grasp"):
        grasp_config_search.generate_validated_grasp_config(
            "ycb:test",
            output=output,
            attempts=2,
        )

    assert output.read_text(encoding="utf-8") == '{"known": "stable"}'
    assert not output.with_suffix(".json.candidate").exists()



def test_grasp_config_directories_are_end_effector_scoped() -> None:
    dex = grasp_config_search.grasp_config_directory("dex_hand")
    pika = grasp_config_search.grasp_config_directory("pika_gripper")

    assert dex == grasp_config_search.PROJECT_ROOT / "configs" / "grasps" / "dex_hand"
    assert pika == grasp_config_search.PROJECT_ROOT / "configs" / "grasps" / "pika_gripper"
    assert grasp_config_search.grasp_config_directory(
        "dex_hand", benchmark=True
    ) == dex / "benchmark"
    assert grasp_config_search.grasp_benchmark_report_path(
        "pika_gripper"
    ) == pika / "grasp_catalog_benchmark.json"
