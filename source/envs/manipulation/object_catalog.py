"""Project-local manipulation-object catalogue."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np

from source.assets import asset_path

MANIFEST_PATH = asset_path("maniskill", "manifest.json")
DEFAULT_LIFT_OBJECT = "ycb:002_master_chef_can"
DEFAULT_PICK_PLACE_OBJECT = "ycb:025_mug"
DEFAULT_PUSH_OBJECT = "ycb:006_mustard_bottle"
DEFAULT_STACK_OBJECTS = ("ycb:070-a_colored_wood_blocks", "ycb:070-b_colored_wood_blocks")
DEFAULT_MAX_HORIZONTAL_DIAMETER = 0.075

PICK_PLACE_EXCLUDED = frozenset(
    {
        "ycb:059_chain",
        "ycb:063-a_marbles",
        "ycb:063-b_marbles",
        "ycb:071_nine_hole_peg_test",
    }
)
PUSH_EXCLUDED = PICK_PLACE_EXCLUDED | frozenset(
    {
        "ycb:053_mini_soccer_ball",
        "ycb:054_softball",
        "ycb:055_baseball",
        "ycb:056_tennis_ball",
        "ycb:057_racquetball",
        "ycb:058_golf_ball",
    }
)
STACK_OBJECTS = (
    "ycb:008_pudding_box",
    "ycb:009_gelatin_box",
    "ycb:010_potted_meat_can",
    "ycb:029_plate",
    "ycb:036_wood_block",
    "ycb:061_foam_brick",
    "ycb:062_dice",
    "ycb:070-a_colored_wood_blocks",
    "ycb:070-b_colored_wood_blocks",
    "ycb:073-a_lego_duplo",
    "ycb:073-b_lego_duplo",
    "ycb:073-c_lego_duplo",
    "ycb:073-d_lego_duplo",
    "ycb:073-e_lego_duplo",
    "ycb:073-f_lego_duplo",
    "ycb:077_rubiks_cube",
    "egad:C0",
    "egad:C1",
    "egad:D0",
    "egad:D1",
    "egad:E0",
    "egad:F0",
    "egad:F1",
    "egad:G0",
    "egad:G1",
)


@lru_cache(maxsize=1)
def object_records() -> dict[str, dict]:
    if not MANIFEST_PATH.is_file():
        raise FileNotFoundError(
            f"Manipulation object manifest is missing: {MANIFEST_PATH}. "
            "Run `python tools/download_maniskill_objects.py` first."
        )
    payload = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    records: dict[str, dict] = {}
    for record in payload.get("objects", []):
        key = f"{record['dataset']}:{record['object_id']}"
        records[key] = record
    if not records:
        raise RuntimeError(f"No object records found in {MANIFEST_PATH}")
    return records


def object_ids(dataset: str | None = None) -> tuple[str, ...]:
    keys = object_records()
    return tuple(key for key in keys if dataset is None or key.startswith(f"{dataset}:"))


def lift_object_ids() -> tuple[str, ...]:
    return object_ids()


def pick_place_object_ids() -> tuple[str, ...]:
    return tuple(key for key in object_ids() if key not in PICK_PLACE_EXCLUDED)


def push_object_ids() -> tuple[str, ...]:
    """Objects that remain controllable in a planar pushing task."""
    return tuple(key for key in object_ids() if key not in PUSH_EXCLUDED)


def stack_object_ids() -> tuple[str, ...]:
    available = object_records()
    return tuple(key for key in STACK_OBJECTS if key in available)


def resolve_record(object_id: str) -> dict:
    try:
        return object_records()[object_id]
    except KeyError as exc:
        available = ", ".join(object_ids()[:8])
        raise ValueError(f"Unknown object_id {object_id!r}. Examples: {available}") from exc


def resolve_record_path(record: dict, field: str) -> Path:
    value = Path(record[field])
    return value if value.is_absolute() else asset_path().parent / value


def record_scale_to_meters(record: dict) -> float:
    """Return the mesh-unit to metre conversion for a catalogue record.

    Schema-1 manifests did not store units. YCB OBJ vertices are metres while
    the official EGAD evaluation meshes are millimetres, so retain a narrow
    compatibility fallback for those manifests.
    """
    value = record.get("scale_to_meters")
    if value is None:
        value = 0.001 if record.get("dataset") == "egad" else 1.0
    scale = float(value)
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError(f"Invalid scale_to_meters for {record.get('object_id')!r}: {value!r}")
    return scale


def limited_mesh_scale(
    record: dict,
    extent: np.ndarray,
    *,
    target_size: float | None = None,
    maximum_horizontal_diameter: float | None = DEFAULT_MAX_HORIZONTAL_DIAMETER,
) -> float:
    """Return one uniform mesh scale with an optional horizontal size cap."""
    extent = np.asarray(extent, dtype=np.float64)
    if extent.shape != (3,) or np.any(~np.isfinite(extent)) or np.any(extent <= 0.0):
        raise ValueError("extent must contain three positive finite values.")
    if target_size is not None and target_size <= 0.0:
        raise ValueError("target_size must be positive when provided.")
    if maximum_horizontal_diameter is not None and maximum_horizontal_diameter <= 0.0:
        raise ValueError("maximum_horizontal_diameter must be positive when provided.")
    scale = (
        record_scale_to_meters(record)
        if target_size is None
        else float(target_size) / float(extent.max())
    )
    if maximum_horizontal_diameter is not None:
        horizontal = float(extent[:2].max()) * scale
        if horizontal > maximum_horizontal_diameter:
            scale *= float(maximum_horizontal_diameter) / horizontal
    return float(scale)
