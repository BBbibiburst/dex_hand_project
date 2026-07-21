"""Render a grasp catalogue benchmark report as a PNG dashboard."""

from __future__ import annotations

import argparse
from pathlib import Path

from source.viz.grasp_benchmark import render_grasp_benchmark_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--sort", choices=("catalog", "name", "status", "drift", "time"), default="catalog"
    )
    parser.add_argument("--top-failures", type=int, default=15)
    parser.add_argument("--max-labels", type=int, default=24)
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = render_grasp_benchmark_report(
        args.report,
        output=args.output,
        sort_mode=args.sort,
        top_failures=args.top_failures,
        max_labels=args.max_labels,
        dpi=args.dpi,
        show=args.show,
    )
    print(f"Loaded report: {args.report}")
    print(f"Saved visualization: {output}")


if __name__ == "__main__":
    main()
