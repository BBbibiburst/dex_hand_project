from pathlib import Path
from types import SimpleNamespace
import json

import numpy as np

from apps import collect_scripted_lerobot
from apps.collect_scripted_lerobot import _yaw_from_quaternion
from source.scripted.lift import LiftStrategy


def test_yaw_from_quaternion() -> None:
    angle = 1.25
    quaternion = np.asarray([np.cos(angle / 2.0), 0.0, 0.0, np.sin(angle / 2.0)])
    assert np.isclose(_yaw_from_quaternion(quaternion), angle)


def test_lift_strategy_accepts_explicit_grasp_config() -> None:
    path = Path("configs/grasps/dex_hand/example.json")
    strategy = LiftStrategy(grasp_config_path=path)
    assert strategy.grasp_config_override == path


def test_catalog_evaluation_writes_per_object_success_rate(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source = tmp_path / "grasps.json"
    output = tmp_path / "lift.json"
    source.write_text(
        json.dumps(
            {
                "objects": [
                    {
                        "object_id": "ycb:test",
                        "status": "stable",
                        "config": "grasp.json",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    env = SimpleNamespace(close=lambda: None)
    monkeypatch.setattr(collect_scripted_lerobot, "lift_object_ids", lambda: ("ycb:test",))
    monkeypatch.setattr(collect_scripted_lerobot, "_make_env", lambda *args, **kwargs: env)
    monkeypatch.setattr(collect_scripted_lerobot, "create_strategy", lambda *args, **kwargs: object())
    monkeypatch.setattr(collect_scripted_lerobot, "scripted_grasp_search_options", lambda args: {})
    monkeypatch.setattr(
        collect_scripted_lerobot,
        "_evaluate_episode",
        lambda env, strategy, *, seed, max_steps: {
            "seed": seed,
            "success": seed % 2 == 0,
            "final_phase": "verify",
        },
    )
    args = SimpleNamespace(
        task="lift",
        trials_per_object=2,
        no_tactile=False,
        limit=None,
        grasp_benchmark_report=source,
        object_ids=None,
        dataset="all",
        evaluation_output=output,
        resume_evaluation=False,
        seed=10,
        max_steps=20,
        fps=20,
    )

    collect_scripted_lerobot._run_catalog_evaluation(args)

    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["summary"]["completed_objects"] == 1
    assert report["summary"]["total_episodes"] == 2
    assert report["objects"][0]["success_rate"] == 0.5
