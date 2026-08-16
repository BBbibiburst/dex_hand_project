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



def _source_package(module: str) -> str | None:
    parts = module.split(".")
    if len(parts) < 2 or parts[0] != "source":
        return None
    return ".".join(parts[:2])


def test_source_package_dependency_graph_is_acyclic() -> None:
    graph: dict[str, set[str]] = {}
    for path in SOURCE_ROOT.rglob("*.py"):
        owner = _source_package(".".join(path.with_suffix("").relative_to(PROJECT_ROOT).parts))
        if owner is None:
            continue
        graph.setdefault(owner, set())
        for module in imported_modules(path):
            dependency = _source_package(module)
            if dependency is not None and dependency != owner:
                graph[owner].add(dependency)

    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(node: str, stack: list[str]) -> None:
        if node in visited:
            return
        if node in visiting:
            start = stack.index(node)
            cycle = stack[start:] + [node]
            raise AssertionError("Source package dependency cycle: " + " -> ".join(cycle))
        visiting.add(node)
        stack.append(node)
        for dependency in sorted(graph.get(node, ())):
            visit(dependency, stack)
        stack.pop()
        visiting.remove(node)
        visited.add(node)

    for package in sorted(graph):
        visit(package, [])


def test_layering_regressions_stay_removed() -> None:
    forbidden_edges = {
        "source.robots": {"source.control"},
        "source.grasping": {"source.scripted", "source.execution", "source.workflows"},
        "source.sensors": {"source.viz"},
        "source.teleop": {"source.imitation"},
        "source.data": {
            "source.teleop",
            "source.imitation",
            "source.scripted",
            "source.grasping",
            "source.rl",
        },
    }
    violations: list[str] = []
    for path in SOURCE_ROOT.rglob("*.py"):
        owner = _source_package(".".join(path.with_suffix("").relative_to(PROJECT_ROOT).parts))
        if owner not in forbidden_edges:
            continue
        for module in imported_modules(path):
            dependency = _source_package(module)
            if dependency in forbidden_edges[owner]:
                violations.append(f"{path.relative_to(PROJECT_ROOT)} -> {module}")
    assert not violations, "Layering regression:\n" + "\n".join(violations)


def test_refactored_legacy_modules_do_not_regrow_implementations() -> None:
    assert not (SOURCE_ROOT / "grasping" / "robot_lift_validator.py").exists()
    assert not (SOURCE_ROOT / "sensors" / "tactile" / "surface_fitting.py").exists()
    for legacy_rl_module in (
        "ppo.py",
        "reference.py",
        "trajectory.py",
        "replay.py",
        "mjwarp_env.py",
        "grasp_edit_templates.py",
        "mjwarp_grasp_edit_env.py",
        "grasp_edit_hybrid_ppo.py",
    ):
        assert not (SOURCE_ROOT / "rl" / legacy_rl_module).exists()

    facade = SOURCE_ROOT / "grasping" / "grasp_config_search.py"
    assert len(facade.read_text(encoding="utf-8").splitlines()) <= 25
