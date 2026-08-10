"""Project asset path constants used by environment builders and demos."""

from __future__ import annotations

from pathlib import Path

PathLike = str | Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ASSETS_DIR = PROJECT_ROOT / "assets"

DEX_HAND_DIR = ASSETS_DIR / "grippers" / "dex_hand"
DEX_HAND_XML_PATH = DEX_HAND_DIR / "dex_hand.xml"
DEX_HAND_MESH_DIR = DEX_HAND_DIR / "meshes"
PIKA_GRIPPER_DIR = ASSETS_DIR / "grippers" / "pika_gripper"
PIKA_GRIPPER_XML_PATH = PIKA_GRIPPER_DIR / "pika_gripper.xml"


def asset_path(*parts: str) -> Path:
    """Return a path under the project assets directory."""
    return ASSETS_DIR.joinpath(*parts)


def resolve_path(path: PathLike | None, default_path: Path) -> Path:
    """Resolve an optional path; fall back to the default when None is passed."""
    return Path(path) if path is not None else default_path
