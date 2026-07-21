# -*- coding: utf-8 -*-
"""Interactive probe demo for any site-based tactile sensor backend.

The demo builds the configured robot, injects a small free probe sphere before
model compilation, and shows tactile readings both in the MuJoCo viewer and as
2D patch heatmaps. Use the MuJoCo viewer's built-in Ctrl + left mouse drag to
grab and move the free probe sphere.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

import mujoco
import numpy as np
from mujoco import viewer

from source.cli.robot_config import add_robot_config_args, load_configured_robot
from source.robots.builder import build_robot_spec
from source.robots.config import (
    descriptors_from_robot_config,
    optional_tuple,
)
from source.robots.scene import add_preview_scene
from source.sensors.base import TactileSensorBase
from source.sensors.tactile.probe import (
    PROBE_GEOM_NAME,
    add_probe_to_spec,
    set_probe_pose,
)



@dataclass(frozen=True)
class TaxelSite:
    site_id: int
    flat_index: int
    patch_name: str


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test tactile matrices with a movable probe.")
    parser.add_argument(
        "--backend",
        default=None,
        help="Tactile backend. Defaults to tactile_backend in the robot config.",
    )
    parser.add_argument(
        "--patch",
        type=str,
        default="",
        help="Only display/test one patch exposed by the configured tactile backend.",
    )
    parser.add_argument("--probe-radius", type=float, default=0.006)
    parser.add_argument("--no-probe-gravity-comp", action="store_true")
    parser.add_argument(
        "--probe-pos",
        type=float,
        nargs=3,
        default=None,
        metavar=("X", "Y", "Z"),
        help="Initial probe world position. Defaults near the selected patch.",
    )
    parser.add_argument(
        "--force-max",
        type=float,
        default=0.0,
        help="Heatmap upper limit; 0 enables per-frame auto scaling.",
    )
    parser.add_argument(
        "--heatmap-gamma",
        type=float,
        default=0.5,
        help="Display-only gamma below 1 brightens low-amplitude neighboring taxels.",
    )
    parser.add_argument("--fps", type=float, default=60.0)
    parser.add_argument("--heatmap-interval", type=float, default=0.05)
    parser.add_argument("--heatmap-cell-size", type=int, default=32)
    parser.add_argument("--debug-tactile", action="store_true")
    parser.add_argument("--debug-interval", type=float, default=0.5)
    parser.add_argument("--no-heatmap", action="store_true")
    parser.add_argument(
        "--show-scene-heat",
        action="store_true",
        help="Draw an extra colored heat overlay on tactile sites in the MuJoCo scene.",
    )
    parser.add_argument("--no-scene", action="store_true")
    add_robot_config_args(
        parser,
        include_device_overrides=False,
        include_tactile_toggle=False,
    )
    return parser.parse_args()




def _build_model_with_probe(
    args: argparse.Namespace,
) -> tuple[mujoco.MjModel, mujoco.MjData, TactileSensorBase, dict[str, Any]]:
    config = load_configured_robot(args)
    config["enable_tactile_sensors"] = True
    if args.backend is not None:
        config["tactile_backend"] = args.backend

    arm_descriptor, hand_descriptor, base_descriptor = descriptors_from_robot_config(config)
    if hand_descriptor.tactile_sensor_factory is None:
        raise ValueError(f"End effector {hand_descriptor.name!r} has no tactile backend.")
    tactile_sensor = hand_descriptor.tactile_sensor_factory(
        str(config.get("tactile_backend", "simple_box")),
        **dict(config.get("tactile_options") or {}),
    )

    spec = build_robot_spec(
        arm_descriptor=arm_descriptor,
        hand_descriptor=hand_descriptor,
        base_descriptor=base_descriptor,
        rot_xyz_deg=optional_tuple(config, "hand_attach_rot_xyz_deg"),
        attach_point_name=config.get("attach_point_name"),
        base_mount_site_name=config.get("base_mount_site_name"),
        hand_prefix=config.get("hand_prefix"),
        tactile_sensor=tactile_sensor,
        add_tactile_sensors=True,
    )
    if bool(config.get("add_preview_scene", True)) and not args.no_scene:
        add_preview_scene(spec)

    initial = np.asarray(args.probe_pos if args.probe_pos is not None else [0.35, 0.0, 1.0])
    add_probe_to_spec(
        spec,
        radius=args.probe_radius,
        initial_pos=initial,
        gravity_comp=not args.no_probe_gravity_comp,
    )

    model = spec.compile()
    data = mujoco.MjData(model)
    tactile_sensor.bind(model, data)
    return model, data, tactile_sensor, config


def _collect_taxel_sites(
    model: mujoco.MjModel,
    sensor: TactileSensorBase,
    *,
    patch_filter: str,
) -> list[TaxelSite]:
    taxels: list[TaxelSite] = []
    refs = sensor.visualization_sites()
    for ref in refs:
        if patch_filter and ref.patch != patch_filter:
            continue
        full_name = getattr(sensor, "name_prefix", "") + ref.name
        site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, full_name)
        if site_id < 0:
            raise ValueError(f"Missing tactile site {full_name!r}.")
        taxels.append(TaxelSite(site_id, ref.flat_index, ref.patch))
    if not taxels:
        raise ValueError(f"No tactile taxels matched patch filter {patch_filter!r}.")
    return taxels


def _initial_probe_position(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    taxels: list[TaxelSite],
    *,
    explicit_pos: Optional[list[float]],
    radius: float,
) -> np.ndarray:
    if explicit_pos is not None:
        return np.asarray(explicit_pos, dtype=np.float64)

    first_patch = taxels[0].patch_name
    placement_taxels = [item for item in taxels if item.patch_name == first_patch]
    positions = np.asarray(
        [data.site_xpos[item.site_id] for item in placement_taxels], dtype=np.float64
    )
    center = positions.mean(axis=0)
    normals = np.asarray(
        [data.site_xmat[item.site_id].reshape(3, 3)[:, 2] for item in placement_taxels]
    )
    normal = normals.mean(axis=0)
    norm = float(np.linalg.norm(normal))
    if norm < 1e-8:
        normal = normals[0]
    else:
        normal /= norm
    return center + max(5.0 * radius, 0.035) * normal






def _geom_name(model: mujoco.MjModel, geom_id: int) -> str:
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(geom_id))
    return "" if name is None else name


def _debug_tactile_contacts(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    values: np.ndarray,
    taxels: list[TaxelSite],
) -> str:
    probe_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, PROBE_GEOM_NAME)
    probe_contacts = 0
    tactile_body_contacts = 0
    tactile_body_ids = {int(model.site_bodyid[item.site_id]) for item in taxels}
    min_dist = np.inf
    other_names: list[str] = []

    for contact_index in range(data.ncon):
        contact = data.contact[contact_index]
        if contact.geom1 != probe_geom_id and contact.geom2 != probe_geom_id:
            continue
        probe_contacts += 1
        other_id = contact.geom2 if contact.geom1 == probe_geom_id else contact.geom1
        other_name = _geom_name(model, int(other_id))
        other_names.append(other_name)
        if int(model.geom_bodyid[other_id]) in tactile_body_ids:
            tactile_body_contacts += 1
        min_dist = min(min_dist, float(contact.dist))

    min_text = "none" if not np.isfinite(min_dist) else f"{min_dist:.6g}"
    sample_names = ", ".join(name for name in other_names[:4] if name)
    return (
        f"max_tactile={float(np.max(values)):.6g} "
        f"contacts={data.ncon} probe_contacts={probe_contacts} "
        f"tactile_body_contacts={tactile_body_contacts} "
        f"min_probe_dist={min_text} other=[{sample_names}]"
    )


def _color_for_force(value: float, force_max: float) -> tuple[float, float, float, float]:
    t = float(np.clip(value / max(force_max, 1e-9), 0.0, 1.0))
    return (
        0.08 + 0.92 * t,
        0.18 + 0.55 * (1.0 - abs(t - 0.45) / 0.45 if t < 0.9 else 0.0),
        1.0 - 0.95 * t,
        0.18 + 0.72 * t,
    )


def _draw_heat_taxels(
    handle: viewer.Handle,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    taxels: list[TaxelSite],
    values: np.ndarray,
    *,
    force_max: float,
    radius: float,
) -> None:
    scene = handle.user_scn
    marker_radius = max(0.0012, 0.35 * radius)
    for taxel in taxels:
        if scene.ngeom >= scene.maxgeom:
            return
        value = float(values[taxel.flat_index])
        mujoco.mjv_initGeom(
            scene.geoms[scene.ngeom],
            mujoco.mjtGeom.mjGEOM_SPHERE,
            np.asarray([marker_radius, marker_radius, marker_radius], dtype=np.float64),
            np.asarray(data.site_xpos[taxel.site_id], dtype=np.float64),
            np.eye(3, dtype=np.float64).reshape(9),
            np.asarray(_color_for_force(value, force_max), dtype=np.float32),
        )
        scene.ngeom += 1


def _create_heatmap_window(
    sensor: TactileSensorBase,
    *,
    patch_filter: str,
    force_max: float,
    cell_size: int,
    heatmap_gamma: float,
):
    import cv2

    window = f"{type(sensor).__name__} tactile heatmap"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    return {
        "cv2": cv2,
        "window": window,
        "patch_filter": patch_filter,
        "force_max": force_max,
        "display_max": max(force_max, 1e-6),
        "gamma": heatmap_gamma,
        "cell_size": max(8, int(cell_size)),
    }


def _colormap_id(cv2_module) -> int:
    return getattr(cv2_module, "COLORMAP_INFERNO", cv2_module.COLORMAP_JET)


def _heatmap_tile(
    cv2_module,
    values: np.ndarray,
    *,
    title: str,
    force_max: float,
    cell_size: int,
    gamma: float,
    min_width: int = 0,
) -> np.ndarray:
    normalized = np.clip(values / max(force_max, 1e-9), 0.0, 1.0)
    normalized = normalized ** max(float(gamma), 1e-6)
    image = np.rint(np.flipud(normalized) * 255.0).astype(np.uint8)
    height = max(1, int(values.shape[0]) * cell_size)
    width = max(1, int(values.shape[1]) * cell_size)
    image = cv2_module.resize(image, (width, height), interpolation=cv2_module.INTER_NEAREST)
    color = cv2_module.applyColorMap(image, _colormap_id(cv2_module))

    if min_width > color.shape[1]:
        pad = min_width - color.shape[1]
        left = pad // 2
        right = pad - left
        color = cv2_module.copyMakeBorder(
            color,
            0,
            0,
            left,
            right,
            cv2_module.BORDER_CONSTANT,
            value=(24, 24, 24),
        )

    title_height = 24
    tile = np.full(
        (color.shape[0] + title_height, color.shape[1], 3),
        24,
        dtype=np.uint8,
    )
    tile[title_height:, :, :] = color
    cv2_module.putText(
        tile,
        title,
        (6, 17),
        cv2_module.FONT_HERSHEY_SIMPLEX,
        0.45,
        (230, 230, 230),
        1,
        cv2_module.LINE_AA,
    )
    return tile


def _pad_to_shape(
    cv2_module,
    image: np.ndarray,
    *,
    height: int,
    width: int,
) -> np.ndarray:
    bottom = max(0, height - image.shape[0])
    right = max(0, width - image.shape[1])
    return cv2_module.copyMakeBorder(
        image,
        0,
        bottom,
        0,
        right,
        cv2_module.BORDER_CONSTANT,
        value=(24, 24, 24),
    )


def _compose_heatmap_panel(
    cv2_module,
    patches: Dict[str, np.ndarray],
    *,
    patch_filter: str,
    force_max: float,
    cell_size: int,
    gamma: float,
) -> np.ndarray:
    if patch_filter:
        if patch_filter not in patches:
            raise ValueError(
                f"Unknown tactile patch {patch_filter!r}; available: {sorted(patches)}."
            )
        return _heatmap_tile(
            cv2_module,
            patches[patch_filter],
            title=patch_filter,
            force_max=force_max,
            cell_size=cell_size,
            gamma=gamma,
        )

    if not patches:
        return np.full((64, 320, 3), 24, dtype=np.uint8)

    # Keep dense films usable without making sparse arrays microscopic.
    largest_dimension = max(max(values.shape) for values in patches.values())
    effective_cell = max(2, min(cell_size, 720 // max(1, largest_dimension)))
    tiles = [
        _heatmap_tile(
            cv2_module,
            values,
            title=name,
            force_max=force_max,
            cell_size=effective_cell,
            gamma=gamma,
        )
        for name, values in patches.items()
    ]
    columns = max(1, int(np.ceil(np.sqrt(len(tiles)))))
    rows = int(np.ceil(len(tiles) / columns))
    tile_height = max(tile.shape[0] for tile in tiles)
    tile_width = max(tile.shape[1] for tile in tiles)
    gap = 10
    blank = np.full((tile_height, tile_width, 3), 24, dtype=np.uint8)
    padded = [
        _pad_to_shape(cv2_module, tile, height=tile_height, width=tile_width) for tile in tiles
    ]
    padded.extend(blank.copy() for _ in range(rows * columns - len(padded)))
    row_images = []
    separator = np.full((tile_height, gap, 3), 24, dtype=np.uint8)
    for row_index in range(rows):
        row_tiles = padded[row_index * columns : (row_index + 1) * columns]
        row_image = row_tiles[0]
        for tile in row_tiles[1:]:
            row_image = np.hstack((row_image, separator, tile))
        row_images.append(row_image)
    panel = row_images[0]
    horizontal = np.full((gap, panel.shape[1], 3), 24, dtype=np.uint8)
    for row_image in row_images[1:]:
        panel = np.vstack((panel, horizontal, row_image))
    return panel


def _update_heatmaps(sensor, values: np.ndarray, heatmap) -> None:
    cv2_module = heatmap["cv2"]
    patches = sensor.patches_from_values(values)
    configured_max = float(heatmap["force_max"])
    if configured_max > 0.0:
        display_max = configured_max
    else:
        current_max = max(float(np.max(values, initial=0.0)), 1e-6)
        display_max = max(current_max, 0.95 * float(heatmap["display_max"]))
        heatmap["display_max"] = display_max
    panel = _compose_heatmap_panel(
        cv2_module,
        patches,
        patch_filter=heatmap["patch_filter"],
        force_max=display_max,
        cell_size=heatmap["cell_size"],
        gamma=heatmap["gamma"],
    )
    cv2_module.imshow(heatmap["window"], panel)
    cv2_module.waitKey(1)


def run_demo(args: argparse.Namespace) -> None:
    if args.force_max < 0.0:
        raise ValueError("--force-max must be non-negative; use 0 for auto scaling.")
    if args.heatmap_gamma <= 0.0:
        raise ValueError("--heatmap-gamma must be positive.")
    model, data, sensor, config = _build_model_with_probe(args)
    mujoco.mj_forward(model, data)
    taxels = _collect_taxel_sites(model, sensor, patch_filter=args.patch)

    probe_pos = _initial_probe_position(
        model,
        data,
        taxels,
        explicit_pos=args.probe_pos,
        radius=args.probe_radius,
    )
    set_probe_pose(
        model,
        data,
        pos=probe_pos,
        quat=np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
    )
    mujoco.mj_forward(model, data)

    heatmap = None
    if not args.no_heatmap:
        try:
            heatmap = _create_heatmap_window(
                sensor,
                patch_filter=args.patch,
                force_max=args.force_max,
                cell_size=args.heatmap_cell_size,
                heatmap_gamma=args.heatmap_gamma,
            )
        except ImportError as exc:
            print(f"OpenCV unavailable; continuing without heatmap: {exc}")

    backend = config.get("tactile_backend", "simple_box")
    print(f"tactile backend: {backend}")
    print(f"taxels displayed: {len(taxels)}")
    print("probe control: Ctrl + left mouse drag in the MuJoCo viewer")

    render_dt = 1.0 / max(args.fps, 1.0)
    last_heatmap = 0.0
    last_debug = 0.0
    viewer_context = viewer.launch_passive(model, data)

    try:
        with viewer_context as handle:
            while handle.is_running():
                loop_start = time.perf_counter()
                mujoco.mj_step(model, data)
                values = np.asarray(sensor.read(model, data), dtype=np.float32).reshape(-1)
                now = time.perf_counter()

                handle.user_scn.ngeom = 0
                if args.show_scene_heat:
                    scene_force_max = (
                        args.force_max
                        if args.force_max > 0.0
                        else max(float(np.max(values, initial=0.0)), 1e-6)
                    )
                    _draw_heat_taxels(
                        handle,
                        model,
                        data,
                        taxels,
                        values,
                        force_max=scene_force_max,
                        radius=args.probe_radius,
                    )
                handle.sync()

                if args.debug_tactile and now - last_debug >= args.debug_interval:
                    print(_debug_tactile_contacts(model, data, values, taxels))
                    last_debug = now

                if heatmap is not None and now - last_heatmap >= args.heatmap_interval:
                    _update_heatmaps(sensor, values, heatmap)
                    last_heatmap = now

                sleep_time = render_dt - (time.perf_counter() - loop_start)
                if sleep_time > 0.0:
                    time.sleep(sleep_time)
    finally:
        if heatmap is not None:
            heatmap["cv2"].destroyWindow(heatmap["window"])


def main() -> None:
    run_demo(_parse_args())


if __name__ == "__main__":
    main()
