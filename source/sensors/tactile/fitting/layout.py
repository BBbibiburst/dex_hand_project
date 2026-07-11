"""Dex-hand tactile patch layout and fitting-strategy metadata."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_DEX_HAND_MESH_DIR = PROJECT_ROOT / "assets" / "grippers" / "dex_hand" / "meshes"
DEFAULT_PLOT_PATCHES = ("skin_0_0_p", "skin_0_2_p", "skin_palm_p")


def dex_hand_patch_layout() -> tuple[tuple[str, int, int, str], ...]:
    """Return the dex-hand tactile patch layout and fitting strategy names."""
    layout: list[tuple[str, int, int, str]] = []
    for finger_id in range(5):
        layout.append((f"skin_{finger_id}_0_p", 4, 8, "segment"))
        layout.append((f"skin_{finger_id}_1_p", 4, 8, "segment"))
        layout.append((f"skin_{finger_id}_2_p", 7, 8, "fingertip-ellipsoid"))
    layout.append(("skin_palm_p", 7, 16, "mesh-uv"))
    return tuple(layout)


DEX_HAND_PATCH_LAYOUT = dex_hand_patch_layout()
_DEX_HAND_PATCH_INFO = {
    mesh_name: (rows, cols, kind) for mesh_name, rows, cols, kind in DEX_HAND_PATCH_LAYOUT
}


def dex_hand_patch_info() -> dict[str, tuple[int, int, str]]:
    """Return immutable-by-convention patch metadata without rebuilding it."""
    return _DEX_HAND_PATCH_INFO
