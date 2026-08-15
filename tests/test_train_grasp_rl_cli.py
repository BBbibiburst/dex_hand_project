from pathlib import Path

import pytest

from apps.train_grasp_rl import _slug, build_parser


def test_object_mode_needs_no_reference_path():
    args = build_parser().parse_args(["--object-id", "ycb:003_cracker_box"])
    assert args.object_ids == ["ycb:003_cracker_box"]
    assert args.reference is None
    assert args.ultra_seeds == 3


def test_dataset_mode_is_one_command():
    args = build_parser().parse_args(["--dataset", "all", "--limit", "2"])
    assert args.dataset == "all"
    assert args.limit == 2


def test_reference_remains_debug_override():
    args = build_parser().parse_args(["--reference", "episode/manifest.json"])
    assert args.reference == Path("episode/manifest.json")


def test_source_modes_are_mutually_exclusive():
    with pytest.raises(SystemExit):
        build_parser().parse_args(
            ["--object-id", "ycb:003_cracker_box", "--dataset", "all"]
        )


def test_slug_is_output_safe():
    assert _slug("ycb:003_cracker_box") == "ycb_003_cracker_box"
