"""Architecture regression tests for package dependency direction."""

from __future__ import annotations

import ast
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "source"
FORBIDDEN_PREFIXES = ("apps", "examples", "tools", "source.demos")


def imported_modules(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    modules: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.append(node.module)
    return modules


def test_source_does_not_depend_on_entrypoint_layers() -> None:
    violations: list[str] = []
    for path in SOURCE_ROOT.rglob("*.py"):
        for module in imported_modules(path):
            if module.startswith(FORBIDDEN_PREFIXES):
                violations.append(f"{path.relative_to(PROJECT_ROOT)} -> {module}")
    assert not violations, "Invalid source dependency direction:\n" + "\n".join(violations)


def test_legacy_demos_package_is_removed() -> None:
    assert not (SOURCE_ROOT / "demos").exists()
