"""Shared scripted-demo options for production grasp search."""

from __future__ import annotations

import argparse


def add_scripted_grasp_search_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--reuse-grasp-config",
        action="store_true",
        help="Reuse configs/grasps cache instead of searching at startup.",
    )
    parser.add_argument("--grasp-search-attempts", type=int, default=3)
    parser.add_argument("--grasp-points", type=int, default=2048)
    parser.add_argument("--grasp-joint-candidates", type=int, default=128)
    parser.add_argument("--grasp-surface-anchors", type=int, default=24)
    parser.add_argument("--grasp-rolls-per-anchor", type=int, default=8)
    parser.add_argument("--grasp-coarse-keep", type=int, default=24)
    parser.add_argument("--grasp-top-k", type=int, default=8)
    parser.add_argument("--grasp-support-margin", type=float, default=0.008)
    parser.add_argument("--grasp-target-size", type=float, default=0.09)


def scripted_grasp_search_options(args: argparse.Namespace) -> dict:
    return {
        "attempts": args.grasp_search_attempts,
        "points": args.grasp_points,
        "joint_candidates": args.grasp_joint_candidates,
        "surface_anchors": args.grasp_surface_anchors,
        "rolls_per_anchor": args.grasp_rolls_per_anchor,
        "coarse_keep": args.grasp_coarse_keep,
        "top_k": args.grasp_top_k,
        "support_margin": args.grasp_support_margin,
        "target_size": args.grasp_target_size,
        "seed": args.seed,
    }


def validate_scripted_grasp_search_args(args: argparse.Namespace) -> None:
    for name in (
        "grasp_search_attempts",
        "grasp_points",
        "grasp_joint_candidates",
        "grasp_surface_anchors",
        "grasp_rolls_per_anchor",
        "grasp_coarse_keep",
        "grasp_top_k",
    ):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive.")
    if args.grasp_support_margin < 0.0 or args.grasp_target_size <= 0.0:
        raise ValueError("Grasp support margin/target size are invalid.")
