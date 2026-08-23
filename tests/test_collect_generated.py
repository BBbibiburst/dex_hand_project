from pathlib import Path

import pytest

from apps.collect_generated_lerobot import build_parser, discover_trajectory_manifests


def _manifest(directory: Path) -> Path:
    directory.mkdir(parents=True)
    path = directory / "manifest.json"
    path.write_text("{}", encoding="utf-8")
    return path.resolve()


def test_discovers_verified_result_layouts_and_deduplicates(tmp_path: Path) -> None:
    best = _manifest(tmp_path / "objects" / "can" / "best_trajectory")
    attempt = _manifest(tmp_path / "rl" / "mug" / "best_attempt")

    found = discover_trajectory_manifests([best], [tmp_path])

    assert found == tuple(sorted((best, attempt)))


def test_explicit_trajectory_directory_resolves_manifest(tmp_path: Path) -> None:
    directory = tmp_path / "trajectory"
    manifest = _manifest(directory)

    assert discover_trajectory_manifests([directory], None) == (manifest,)


def test_missing_input_root_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Input root"):
        discover_trajectory_manifests(None, [tmp_path / "missing"])


def test_cli_accepts_multiple_trajectory_sources() -> None:
    args = build_parser().parse_args(
        ["--trajectory", "one", "--trajectory", "two", "--input-root", "outputs"]
    )

    assert args.trajectories == [Path("one"), Path("two")]
    assert args.input_roots == [Path("outputs")]
