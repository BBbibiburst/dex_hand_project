"""Create one report for the five task-specific benchmark domains."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from source.workflows.batch import atomic_write_json
from source.workflows.benchmark_report import catalogue_task_metrics, fixed_domain_metrics


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lift-summary", type=Path, required=True)
    parser.add_argument("--pick-place-summary", type=Path, required=True)
    parser.add_argument("--stack-domain", type=Path, required=True)
    parser.add_argument("--stack-summary", type=Path, required=True)
    parser.add_argument("--stack-full-summary", type=Path)
    parser.add_argument("--push-domain", type=Path, required=True)
    parser.add_argument("--push-summary", type=Path, required=True)
    parser.add_argument("--push-full-summary", type=Path)
    parser.add_argument("--nut-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = {
        "schema_version": 1,
        "tasks": {
            "lift": fixed_domain_metrics(_json(args.lift_summary), domain="top100_catalogue"),
            "pick_place": fixed_domain_metrics(
                _json(args.pick_place_summary), domain="top100_catalogue"
            ),
            "stack": catalogue_task_metrics(
                _json(args.stack_domain),
                _json(args.stack_summary),
                full_catalogue_summary=(
                    None if args.stack_full_summary is None else _json(args.stack_full_summary)
                ),
            ),
            "push": catalogue_task_metrics(
                _json(args.push_domain),
                _json(args.push_summary),
                full_catalogue_summary=(
                    None if args.push_full_summary is None else _json(args.push_full_summary)
                ),
            ),
            "nut_assembly": fixed_domain_metrics(
                _json(args.nut_summary), domain="procedural_mating_variants"
            ),
        },
    }
    atomic_write_json(args.output, report)
    for task, metrics in report["tasks"].items():
        rate = metrics.get("task_valid_success_rate", metrics.get("success_rate", 0.0))
        print(f"{task:<13} primary_success_rate={rate:.1%}")
    print(f"report={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
