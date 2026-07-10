# -*- coding: utf-8 -*-
"""Interactive probe demo for dex-hand tactile taxel matrices.

The demo builds the configured robot, injects a small free probe sphere before
model compilation, and shows tactile readings both in the MuJoCo viewer and as
2D patch heatmaps. Use the MuJoCo viewer's built-in Ctrl + left mouse drag to
grab and move the free probe sphere.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import time
from typing import Any, Optional

import mujoco
from mujoco import viewer
import numpy as np

from source.demos.common import (
    add_robot_config_args,
    load_demo_robot_config,
    require_hand,
)
from source.robots.builder import build_robot_spec
from source.robots.config import (
    descriptors_from_robot_config,
    optional_tuple,
)
from source.robots.scene import add_preview_scene
from source.sensors.tactile.dex_hand import (
    SUPPORTED_TACTILE_BACKENDS,
    DexHandTactileSensorBase,
    create_dex_hand_tactile_sensor,
    site_name,
)


PROBE_BODY_NAME = "tactile_probe"
PROBE_JOINT_NAME = "tactile_probe_freejoint"
PROBE_GEOM_NAME = "tactile_probe_geom"


@dataclass(frozen=True)
class TaxelSite:
    site_id: int
    flat_index: int
    patch_name: str


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test dex-hand tactile matrices with a movable probe."
    )
    parser.add_argument(
        "--backend",
        choices=SUPPORTED_TACTILE_BACKENDS,
        default=None,
        help="Tactile backend. Defaults to tactile_backend in the robot config.",
    )
    parser.add_argument(
        "--patch",
        type=str,
        default="",
        help="Only display/test one patch, e.g. skin_0_0_p, skin_4_2_p, skin_palm_p.",
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
    parser.add_argument("--force-max", type=float, default=1.0)
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


def _add_probe_to_spec(
    spec: mujoco.MjSpec,
    *,
    radius: float,
    initial_pos: np.ndarray,
    gravity_comp: bool,
) -> None:
    if radius <= 0.0:
        raise ValueError("probe-radius must be positive.")

    probe = spec.worldbody.add_body()
    probe.name = PROBE_BODY_NAME
    probe.pos = np.asarray(initial_pos, dtype=np.float64).tolist()
    if gravity_comp and hasattr(probe, "gravcomp"):
        probe.gravcomp = 1.0

    joint = probe.add_joint()
    joint.name = PROBE_JOINT_NAME
    joint.type = mujoco.mjtJoint.mjJNT_FREE

    geom = probe.add_geom()
    geom.name = PROBE_GEOM_NAME
    geom.type = mujoco.mjtGeom.mjGEOM_SPHERE
    geom.size = [float(radius), 0.0, 0.0]
    geom.rgba = [1.0, 0.12, 0.08, 0.85]
    geom.mass = 0.01
    geom.condim = 3
    geom.contype = 1
    geom.conaffinity = 3
    geom.friction = [0.8, 0.01, 0.001]


def _build_model_with_probe(
    args: argparse.Namespace,
) -> tuple[mujoco.MjModel, mujoco.MjData, DexHandTactileSensorBase, dict[str, Any]]:
    config = load_demo_robot_config(args)
    require_hand(config, "dex_hand", demo_name="tactile_probe_demo")
    config["enable_tactile_sensors"] = True
    if args.backend is not None:
        config["tactile_backend"] = args.backend

    tactile_sensor = create_dex_hand_tactile_sensor(
        str(config.get("tactile_backend", "simple_box")),
        **dict(config.get("tactile_options") or {}),
    )
    arm_descriptor, hand_descriptor, base_descriptor = descriptors_from_robot_config(config)

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
    _add_probe_to_spec(
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
    sensor: DexHandTactileSensorBase,
    *,
    patch_filter: str,
) -> list[TaxelSite]:
    taxels: list[TaxelSite] = []
    for patch in sensor.patches:
        if patch_filter and patch.name != patch_filter:
            continue
        flat_index = patch.start
        for row in range(patch.rows):
            for col in range(patch.cols):
                full_name = sensor.name_prefix + site_name(patch.name, row, col)
                site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, full_name)
                if site_id < 0:
                    raise ValueError(f"Missing tactile site {full_name!r}.")
                taxels.append(TaxelSite(site_id, flat_index, patch.name))
                flat_index += 1
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

    positions = np.asarray([data.site_xpos[item.site_id] for item in taxels], dtype=np.float64)
    center = positions.mean(axis=0)
    return center + np.asarray([0.0, 0.0, max(5.0 * radius, 0.035)], dtype=np.float64)


def _set_probe_pose(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    pos: np.ndarray,
    quat: np.ndarray,
) -> None:
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, PROBE_JOINT_NAME)
    if joint_id < 0:
        raise RuntimeError(f"Probe joint {PROBE_JOINT_NAME!r} was not compiled.")
    qpos_adr = int(model.jnt_qposadr[joint_id])
    qvel_adr = int(model.jnt_dofadr[joint_id])
    data.qpos[qpos_adr:qpos_adr + 3] = np.asarray(pos, dtype=np.float64)
    data.qpos[qpos_adr + 3:qpos_adr + 7] = np.asarray(quat, dtype=np.float64)
    data.qvel[qvel_adr:qvel_adr + 6] = 0.0


def _probe_joint_addresses(model: mujoco.MjModel) -> tuple[int, int]:
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, PROBE_JOINT_NAME)
    if joint_id < 0:
        raise RuntimeError(f"Probe joint {PROBE_JOINT_NAME!r} was not compiled.")
    return int(model.jnt_qposadr[joint_id]), int(model.jnt_dofadr[joint_id])


def _geom_name(model: mujoco.MjModel, geom_id: int) -> str:
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(geom_id))
    return "" if name is None else name


def _debug_tactile_contacts(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    values: np.ndarray,
) -> str:
    probe_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, PROBE_GEOM_NAME)
    probe_contacts = 0
    skin_contacts = 0
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
        if "skin_" in other_name:
            skin_contacts += 1
        min_dist = min(min_dist, float(contact.dist))

    min_text = "none" if not np.isfinite(min_dist) else f"{min_dist:.6g}"
    sample_names = ", ".join(name for name in other_names[:4] if name)
    return (
        f"max_tactile={float(np.max(values)):.6g} "
        f"contacts={data.ncon} probe_contacts={probe_contacts} "
        f"skin_contacts={skin_contacts} "
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
    sensor: DexHandTactileSensorBase,
    *,
    patch_filter: str,
    force_max: float,
    cell_size: int,
):
    import cv2

    window = "Dex-hand tactile heatmap"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    return {
        "cv2": cv2,
        "window": window,
        "patch_filter": patch_filter,
        "force_max": force_max,
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
    min_width: int = 0,
) -> np.ndarray:
    normalized = np.clip(values / max(force_max, 1e-9), 0.0, 1.0)
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
) -> np.ndarray:
    if patch_filter:
        return _heatmap_tile(
            cv2_module,
            patches[patch_filter],
            title=patch_filter,
            force_max=force_max,
            cell_size=cell_size,
        )

    gap = 10
    title_height = 34
    segment_tile_width = 8 * cell_size
    rows: list[np.ndarray] = []
    for segment_id in range(3):
        row_tiles = []
        for finger_id in range(5):
            patch_name = f"skin_{finger_id}_{segment_id}_p"
            values = patches.get(patch_name)
            if values is None:
                values = np.zeros((1, 1), dtype=np.float32)
            row_tiles.append(
                _heatmap_tile(
                    cv2_module,
                    values,
                    title=f"F{finger_id} S{segment_id}",
                    force_max=force_max,
                    cell_size=cell_size,
                    min_width=segment_tile_width,
                )
            )
        row_height = max(tile.shape[0] for tile in row_tiles)
        padded = [
            _pad_to_shape(cv2_module, tile, height=row_height, width=tile.shape[1])
            for tile in row_tiles
        ]
        separator = np.full((row_height, gap, 3), 24, dtype=np.uint8)
        row = padded[0]
        for tile in padded[1:]:
            row = np.hstack([row, separator, tile])
        rows.append(row)

    body_width = max(row.shape[1] for row in rows)
    rows = [_pad_to_shape(cv2_module, row, height=row.shape[0], width=body_width) for row in rows]
    v_separator = np.full((gap, body_width, 3), 24, dtype=np.uint8)
    finger_panel = rows[0]
    for row in rows[1:]:
        finger_panel = np.vstack([finger_panel, v_separator, row])

    palm = patches.get("skin_palm_p")
    if palm is not None:
        palm_tile = _heatmap_tile(
            cv2_module,
            palm,
            title="Palm",
            force_max=force_max,
            cell_size=max(16, int(cell_size * 0.7)),
        )
        palm_tile = _pad_to_shape(
            cv2_module,
            palm_tile,
            height=finger_panel.shape[0],
            width=palm_tile.shape[1],
        )
        panel = np.hstack([finger_panel, np.full((finger_panel.shape[0], gap, 3), 24, dtype=np.uint8), palm_tile])
    else:
        panel = finger_panel

    title = np.full((title_height, panel.shape[1], 3), 18, dtype=np.uint8)
    cv2_module.putText(
        title,
        "Dex-hand tactile heatmap: rows=skin_*_0/1/2_p, columns=fingers 0..4, right=palm",
        (8, 23),
        cv2_module.FONT_HERSHEY_SIMPLEX,
        0.55,
        (235, 235, 235),
        1,
        cv2_module.LINE_AA,
    )
    return np.vstack([title, panel])


def _update_heatmaps(sensor, model, data, heatmap) -> None:
    cv2_module = heatmap["cv2"]
    patches = sensor.read_patches(model, data)
    panel = _compose_heatmap_panel(
        cv2_module,
        patches,
        patch_filter=heatmap["patch_filter"],
        force_max=heatmap["force_max"],
        cell_size=heatmap["cell_size"],
    )
    cv2_module.imshow(heatmap["window"], panel)
    cv2_module.waitKey(1)


def run_demo(args: argparse.Namespace) -> None:
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
    _set_probe_pose(
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
                values = sensor.read(model, data)
                now = time.perf_counter()

                handle.user_scn.ngeom = 0
                if args.show_scene_heat:
                    _draw_heat_taxels(
                        handle,
                        model,
                        data,
                        taxels,
                        values,
                        force_max=args.force_max,
                        radius=args.probe_radius,
                    )
                handle.sync()

                if args.debug_tactile and now - last_debug >= args.debug_interval:
                    print(_debug_tactile_contacts(model, data, values))
                    last_debug = now

                if heatmap is not None and now - last_heatmap >= args.heatmap_interval:
                    _update_heatmaps(sensor, model, data, heatmap)
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
