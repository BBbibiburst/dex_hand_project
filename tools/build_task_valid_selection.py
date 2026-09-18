"""Build a versioned Stack or Push task-valid object selection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from source.workflows.batch import atomic_write_json
from source.workflows.task_domains import build_task_domain_manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("stack", "push"), required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--stability", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    stability = json.loads(args.stability.read_text(encoding="utf-8"))
    manifest = build_task_domain_manifest(args.task, selection, stability)
    atomic_write_json(args.output, manifest)
    print(
        f"task={args.task} accepted={manifest['count']}/{manifest['source_count']} "
        f"excluded={manifest['excluded_count']} output={args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
