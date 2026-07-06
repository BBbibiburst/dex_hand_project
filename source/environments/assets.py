# -*- coding: utf-8 -*-
"""Project asset path constants used by environment builders and demos."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union


PathLike = Union[str, Path]

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
ASSETS_DIR = PROJECT_ROOT / "assets"

ROBOTS_DIR = ASSETS_DIR / "robots"
BASES_DIR = ASSETS_DIR / "bases"
GRIPPERS_DIR = ASSETS_DIR / "grippers"

RM75B_XML_PATH = ROBOTS_DIR / "rm75b" / "rm75b.xml"
DEX_HAND_DIR = GRIPPERS_DIR / "dex_hand"
DEX_HAND_XML_PATH = DEX_HAND_DIR / "dex_hand.xml"
DEX_HAND_MESH_DIR = DEX_HAND_DIR / "meshes"
DEFAULT_BASE_XML_PATH = BASES_DIR / "rethink_minimal_mount.xml"


def resolve_path(path: Optional[PathLike], default_path: Path) -> Path:
    """Resolve an optional path; fall back to the default when None is passed."""
    return Path(path) if path is not None else default_path
