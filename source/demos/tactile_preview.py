# -*- coding: utf-8 -*-
"""Preview generated dex-hand tactile taxel positions in the MuJoCo viewer."""

from __future__ import annotations

import argparse

import numpy as np

from source.environments.robot_builder import DEFAULT_HAND_PREFIX, build_combined_model
from source.environments.tactile_layout import DEX_HAND_TACTILE_PATCHES, tactile_site_name


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preview dex-hand tactile taxel sites.")
    parser.add_argument(
        "--patch",
        type=str,
        default="",
        help="Only draw one skin patch, e.g. skin_0_0_p, skin_4_2_p, or skin_palm_p.",
    )
    parser.add_argument("--radius", type=float, default=0.0025)
    parser.add_argument("--no-prefix", action="store_true")
    return parser.parse_args()


def _patch_color(mesh_name: str) -> tuple[float, float, float, float]:
    if mesh_name == "skin_palm_p":
        return (1.0, 0.2, 0.1, 0.75)
    if mesh_name.endswith("_0_p"):
        return (0.0, 0.9, 1.0, 0.75)
    if mesh_name.endswith("_1_p"):
        return (0.1, 1.0, 0.25, 0.75)
    return (1.0, 0.85, 0.05, 0.75)


def _add_sphere(handle, pos: np.ndarray, radius: float, rgba) -> None:
    import mujoco

    if handle.user_scn.ngeom >= handle.user_scn.maxgeom:
        return

    geom = handle.user_scn.geoms[handle.user_scn.ngeom]
    mujoco.mjv_initGeom(
        geom,
        mujoco.mjtGeom.mjGEOM_SPHERE,
        np.asarray([radius, radius, radius], dtype=np.float64),
        np.asarray(pos, dtype=np.float64),
        np.eye(3, dtype=np.float64).reshape(9),
        np.asarray(rgba, dtype=np.float32),
    )
    handle.user_scn.ngeom += 1


def _collect_sites(model, *, prefix: str, patch_filter: str) -> list[tuple[int, str]]:
    import mujoco

    sites: list[tuple[int, str]] = []
    for patch in DEX_HAND_TACTILE_PATCHES:
        if patch_filter and patch.mesh_name != patch_filter:
            continue
        for row in range(patch.rows):
            for col in range(patch.cols):
                name = prefix + tactile_site_name(patch.mesh_name, row, col)
                site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
                if site_id < 0:
                    raise ValueError(f"Missing tactile site {name!r}.")
                sites.append((site_id, patch.mesh_name))
    return sites


def main() -> None:
    args = _parse_args()
    prefix = "" if args.no_prefix else DEFAULT_HAND_PREFIX

    import mujoco
    from mujoco import viewer

    model, data = build_combined_model(add_scene=True)
    mujoco.mj_forward(model, data)

    sites = _collect_sites(model, prefix=prefix, patch_filter=args.patch)
    print(f"Drawing {len(sites)} tactile sites.")
    print("Colors: proximal=cyan, middle=green, fingertip=yellow, palm=red")

    with viewer.launch_passive(model, data) as handle:
        while handle.is_running():
            mujoco.mj_step(model, data)
            handle.user_scn.ngeom = 0
            for site_id, mesh_name in sites:
                _add_sphere(
                    handle,
                    data.site_xpos[site_id],
                    args.radius,
                    _patch_color(mesh_name),
                )
            handle.sync()


if __name__ == "__main__":
    main()
