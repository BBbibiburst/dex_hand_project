"""Architecture checks for the shared Vive + glove collection runtime."""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COLLECTORS = (
    ROOT / "apps" / "collect_teleop_lerobot.py",
    ROOT / "apps" / "collect_teleop_trajectory.py",
)


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    result: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            result.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            result.add(node.module)
    return result


def test_collectors_share_one_teleop_session_runtime() -> None:
    for path in COLLECTORS:
        imports = _imports(path)
        assert "source.teleop.session" in imports
        assert "source.teleop.devices" not in imports
        assert "source.teleop.mapping" not in imports
        assert "source.viz.teleop_dashboard" not in imports


def test_shared_session_owns_hardware_mapping_and_dashboard() -> None:
    imports = _imports(ROOT / "source" / "teleop" / "session.py")
    assert "source.teleop.devices" in imports
    assert "source.teleop.mapping" in imports
    assert "source.teleop.glove_processing" in imports
    assert "source.teleop.vive.coordinates" in imports
    assert "source.viz.teleop_dashboard" in imports
