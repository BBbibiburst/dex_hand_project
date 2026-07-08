# -*- coding: utf-8 -*-
"""Preview generated dex-hand tactile taxel positions in the MuJoCo viewer.

This demo is intentionally dex-hand specific: it imports
``DexHandTouchSensor`` directly rather than going through the generic
``TactileSensorBase`` interface, because it wants to draw per-patch colors
by mesh name — a detail only the dex hand's implementation knows about.
"""

from __future__ import annotations

import argparse

import mujoco
from mujoco import viewer
import numpy as np

from source.environments.overlays import draw_sphere_marker
from source.environments.robot_builder import build_robot_model
from source.robots.defaults import DEFAULT_HAND
from source.robots.hands.dex_hand_tactile import DexHandTouchSensor, site_name


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


def _collect_sites(
    model: mujoco.MjModel,
    sensor: DexHandTouchSensor,
    *,
    prefix: str,
    patch_filter: str,
) -> list[tuple[int, str]]:
    sites: list[tuple[int, str]] = []
    for mesh_name, rows, cols, _kind in sensor.patch_layout:
        if patch_filter and mesh_name != patch_filter:
            continue
        for row in range(rows):
            for col in range(cols):
                name = prefix + site_name(mesh_name, row, col)
                site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
                if site_id < 0:
                    raise ValueError(f"Missing tactile site {name!r}.")
                sites.append((site_id, mesh_name))
    return sites


def main() -> None:
    args = _parse_args()

    tactile_sensor = DexHandTouchSensor()
    hand_prefix = "" if args.no_prefix else DEFAULT_HAND.default_prefix

    model, data = build_robot_model(add_scene=True, tactile_sensor=tactile_sensor)
    mujoco.mj_forward(model, data)

    sites = _collect_sites(model, tactile_sensor, prefix=hand_prefix, patch_filter=args.patch)
    print(f"Drawing {len(sites)} tactile sites.")
    print("Colors: proximal=cyan, middle=green, fingertip=yellow, palm=red")

    with viewer.launch_passive(model, data) as handle:
        while handle.is_running():
            mujoco.mj_step(model, data)
            handle.user_scn.ngeom = 0
            for site_id, mesh_name in sites:
                draw_sphere_marker(
                    handle,
                    data.site_xpos[site_id],
                    radius=args.radius,
                    rgba=_patch_color(mesh_name),
                )
            handle.sync()


if __name__ == "__main__":
    main()
