"""Small, domain-neutral helpers for resumable batch workflows."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def object_slug(object_id: str) -> str:
    """Return a filesystem-safe directory name for a catalogue object ID."""
    return object_id.replace(":", "_").replace("/", "_")


def load_selection_ids(path: Path) -> tuple[str, ...]:
    """Load unique object IDs from either a JSON list or an ``objects`` field."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("objects", payload) if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        raise ValueError(f"Selection has no object list: {path}")
    result = tuple(str(row["object_id"] if isinstance(row, dict) else row) for row in rows)
    if len(set(result)) != len(result):
        raise ValueError("Selection contains duplicate object IDs.")
    return result


def atomic_write_json(path: Path, payload: Any) -> None:
    """Write JSON through a sibling temporary file and atomically replace it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def lift_source_roots(
    lattice_root: Path,
    rl_root: Path,
    grasp_root: Path,
    object_id: str,
    extra_roots: tuple[Path, ...] = (),
) -> tuple[Path, ...]:
    """Return existing per-object Lift artifact roots in precedence order."""
    slug = object_slug(object_id)
    candidates = (
        grasp_root / slug,
        lattice_root / slug,
        lattice_root / "recovery_lift_085mm" / slug,
        rl_root / slug,
        *(root / slug for root in extra_roots),
    )
    return tuple(path for path in candidates if path.is_dir())


def memory_safe_worker_count(requested: int, *, gib_per_worker: float = 1.5) -> int:
    """Cap concurrent heavyweight workers using currently available host RAM."""
    if requested <= 0:
        raise ValueError("requested workers must be positive.")
    if gib_per_worker <= 0.0:
        raise ValueError("gib_per_worker must be positive.")
    try:
        fields = {
            line.split(":", 1)[0]: int(line.split()[1])
            for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines()
            if ":" in line
        }
        available_gib = fields["MemAvailable"] / (1024**2)
    except (OSError, KeyError, ValueError):
        pages = os.sysconf("SC_AVPHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        available_gib = pages * page_size / (1024**3)
    safe = max(1, int((available_gib - 1.5) // gib_per_worker))
    return min(requested, safe)
