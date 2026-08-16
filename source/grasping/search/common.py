"""Shared paths and logging for grasp search."""

from __future__ import annotations

import logging
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
ROOT = PROJECT_ROOT
MANIFEST = PROJECT_ROOT / "assets" / "maniskill" / "manifest.json"
LOGGER = logging.getLogger("source.grasping.search")

def progress(message: str) -> None:
    """Emit internal diagnostics without polluting normal CLI output."""
    LOGGER.debug(message)
