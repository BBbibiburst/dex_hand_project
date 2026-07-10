# -*- coding: utf-8 -*-
"""Preview generated dex-hand tactile taxel sites in the MuJoCo viewer.

The preview reads each compiled site's actual geometry, size, position, and
orientation. Therefore box-shaped tactile sites are displayed as real boxes
instead of being approximated with sphere markers.
"""

from __future__ import annotations

import argparse

import mujoco
from mujoco import viewer
import numpy as np

from source.demos.common import (
    add_robot_config_args,
    load_demo_robot_config,
    require_hand,
)
from source.robots.builder import build_robot_model_from_config
from source.robots.registry import get_hand
from source.sensors.tactile.dex_hand import (
    SUPPORTED_TACTILE_BACKENDS,
    DexHandTactileSensorBase,
    create_dex_hand_tactile_sensor,
    site_name,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preview dex-hand tactile taxel sites."
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
        help=(
            "Only draw one skin patch, for example "
            "skin_0_0_p, skin_4_2_p, or skin_palm_p."
        ),
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help=(
            "Visualization-only size multiplier. "
            "Does not change the actual touch sensor."
        ),
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.65,
        help="Preview opacity in the range [0, 1].",
    )
    parser.add_argument(
        "--wireframe",
        action="store_true",
        help="Draw box edges instead of solid translucent boxes.",
    )
    parser.add_argument(
        "--normal-length",
        type=float,
        default=0.0,
        help=(
            "Draw each taxel's local +Z normal with this length in meters. "
            "Set to 0 to disable."
        ),
    )
    parser.add_argument(
        "--no-prefix",
        action="store_true",
        help="Do not prepend the configured hand prefix to tactile site names.",
    )

    add_robot_config_args(
        parser,
        include_device_overrides=False,
        include_tactile_toggle=False,
    )
    return parser.parse_args()


def _patch_color(
    mesh_name: str,
    alpha: float,
) -> tuple[float, float, float, float]:
    """Return a distinct color for each skin patch kind."""

    if mesh_name == "skin_palm_p":
        return (1.0, 0.2, 0.1, alpha)

    if mesh_name.endswith("_0_p"):
        return (0.0, 0.9, 1.0, alpha)

    if mesh_name.endswith("_1_p"):
        return (0.1, 1.0, 0.25, alpha)

    return (1.0, 0.85, 0.05, alpha)


def _collect_sites(
    model: mujoco.MjModel,
    sensor: DexHandTactileSensorBase,
    *,
    prefix: str,
    patch_filter: str,
) -> list[tuple[int, str]]:
    """Resolve all generated tactile site IDs."""

    sites: list[tuple[int, str]] = []

    for mesh_name, rows, cols, _kind in sensor.patch_layout:
        if patch_filter and mesh_name != patch_filter:
            continue

        for row in range(rows):
            for col in range(cols):
                name = prefix + site_name(mesh_name, row, col)

                site_id = mujoco.mj_name2id(
                    model,
                    mujoco.mjtObj.mjOBJ_SITE,
                    name,
                )

                if site_id < 0:
                    raise ValueError(f"Missing tactile site {name!r}.")

                sites.append((site_id, mesh_name))

    return sites


def _append_geom(
    handle: viewer.Handle,
    *,
    geom_type: mujoco.mjtGeom,
    size: np.ndarray,
    pos: np.ndarray,
    mat: np.ndarray,
    rgba: tuple[float, float, float, float],
) -> None:
    """Append one visualization geom to the viewer user scene."""

    scene = handle.user_scn

    if scene.ngeom >= scene.maxgeom:
        return

    mujoco.mjv_initGeom(
        scene.geoms[scene.ngeom],
        geom_type,
        np.asarray(size, dtype=np.float64),
        np.asarray(pos, dtype=np.float64),
        np.asarray(mat, dtype=np.float64).reshape(9),
        np.asarray(rgba, dtype=np.float32),
    )

    scene.ngeom += 1


def _draw_box_edges(
    handle: viewer.Handle,
    *,
    pos: np.ndarray,
    mat: np.ndarray,
    half_size: np.ndarray,
    rgba: tuple[float, float, float, float],
    line_radius: float = 0.00015,
) -> None:
    """Draw the twelve edges of an oriented box."""

    half_size = np.asarray(half_size, dtype=np.float64)
    mat = np.asarray(mat, dtype=np.float64).reshape(3, 3)
    pos = np.asarray(pos, dtype=np.float64)

    local_corners = np.asarray(
        [
            [-half_size[0], -half_size[1], -half_size[2]],
            [+half_size[0], -half_size[1], -half_size[2]],
            [+half_size[0], +half_size[1], -half_size[2]],
            [-half_size[0], +half_size[1], -half_size[2]],
            [-half_size[0], -half_size[1], +half_size[2]],
            [+half_size[0], -half_size[1], +half_size[2]],
            [+half_size[0], +half_size[1], +half_size[2]],
            [-half_size[0], +half_size[1], +half_size[2]],
        ],
        dtype=np.float64,
    )

    # data.site_xmat maps local site coordinates into world coordinates.
    world_corners = pos + local_corners @ mat.T

    edges = (
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 4),
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
    )

    for start_index, end_index in edges:
        scene = handle.user_scn

        if scene.ngeom >= scene.maxgeom:
            return

        geom = scene.geoms[scene.ngeom]

        mujoco.mjv_initGeom(
            geom,
            mujoco.mjtGeom.mjGEOM_CAPSULE,
            np.zeros(3, dtype=np.float64),
            np.zeros(3, dtype=np.float64),
            np.eye(3, dtype=np.float64).reshape(9),
            np.asarray(rgba, dtype=np.float32),
        )

        mujoco.mjv_makeConnector(
            geom,
            mujoco.mjtGeom.mjGEOM_CAPSULE,
            float(line_radius),
            *world_corners[start_index],
            *world_corners[end_index],
        )

        scene.ngeom += 1


def _draw_normal(
    handle: viewer.Handle,
    *,
    pos: np.ndarray,
    mat: np.ndarray,
    length: float,
) -> None:
    """Draw the site's local positive Z axis."""

    if length <= 0.0:
        return

    scene = handle.user_scn

    if scene.ngeom >= scene.maxgeom:
        return

    pos = np.asarray(pos, dtype=np.float64)
    mat = np.asarray(mat, dtype=np.float64).reshape(3, 3)

    # Third column is the local +Z axis represented in world coordinates.
    normal = mat[:, 2]
    end = pos + float(length) * normal

    geom = scene.geoms[scene.ngeom]

    mujoco.mjv_initGeom(
        geom,
        mujoco.mjtGeom.mjGEOM_CAPSULE,
        np.zeros(3, dtype=np.float64),
        np.zeros(3, dtype=np.float64),
        np.eye(3, dtype=np.float64).reshape(9),
        np.asarray((1.0, 0.0, 1.0, 1.0), dtype=np.float32),
    )

    mujoco.mjv_makeConnector(
        geom,
        mujoco.mjtGeom.mjGEOM_CAPSULE,
        0.00025,
        *pos,
        *end,
    )

    scene.ngeom += 1


def _draw_site(
    handle: viewer.Handle,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    site_id: int,
    rgba: tuple[float, float, float, float],
    scale: float,
    wireframe: bool,
    normal_length: float,
) -> None:
    """Draw one site using its compiled type, size, pose, and orientation."""

    site_type = mujoco.mjtGeom(int(model.site_type[site_id]))
    site_size = np.asarray(model.site_size[site_id], dtype=np.float64) * scale
    site_pos = np.asarray(data.site_xpos[site_id], dtype=np.float64)
    site_mat = np.asarray(data.site_xmat[site_id], dtype=np.float64).reshape(3, 3)

    if site_type == mujoco.mjtGeom.mjGEOM_BOX and wireframe:
        _draw_box_edges(
            handle,
            pos=site_pos,
            mat=site_mat,
            half_size=site_size,
            rgba=rgba,
        )
    else:
        _append_geom(
            handle,
            geom_type=site_type,
            size=site_size,
            pos=site_pos,
            mat=site_mat,
            rgba=rgba,
        )

    _draw_normal(
        handle,
        pos=site_pos,
        mat=site_mat,
        length=normal_length,
    )


def main() -> None:
    args = _parse_args()

    if args.scale <= 0.0:
        raise ValueError(f"--scale must be positive, got {args.scale}.")

    if not 0.0 <= args.alpha <= 1.0:
        raise ValueError(f"--alpha must be in [0, 1], got {args.alpha}.")

    if args.normal_length < 0.0:
        raise ValueError(
            f"--normal-length must be non-negative, got {args.normal_length}."
        )

    config = load_demo_robot_config(args)
    require_hand(config, "dex_hand", demo_name="tactile_preview")

    hand_name = str(config.get("hand_name", "dex_hand"))
    backend = args.backend or str(config.get("tactile_backend", "simple_box"))
    tactile_options = dict(config.get("tactile_options") or {})
    tactile_sensor = create_dex_hand_tactile_sensor(backend, **tactile_options)
    hand_descriptor = get_hand(hand_name)

    hand_prefix = (
        ""
        if args.no_prefix
        else str(
            config.get("hand_prefix")
            or hand_descriptor.default_prefix
        )
    )

    model, data = build_robot_model_from_config(
        args.robot_config,
        tactile_sensor=tactile_sensor,
        arm_name=config.get("arm_name"),
        hand_name=hand_name,
        base_name=config.get("base_name"),
        enable_tactile_sensors=True,
        add_preview_scene=True,
    )

    mujoco.mj_forward(model, data)

    sites = _collect_sites(
        model,
        tactile_sensor,
        prefix=hand_prefix,
        patch_filter=args.patch,
    )

    box_count = sum(
        int(model.site_type[site_id])
        == int(mujoco.mjtGeom.mjGEOM_BOX)
        for site_id, _mesh_name in sites
    )

    print(f"Backend: {tactile_sensor.backend_name}")
    print(f"Drawing {len(sites)} tactile sites.")
    print(f"Box sites: {box_count}")
    print(
        "Colors: proximal=cyan, middle=green, "
        "fingertip=yellow, palm=red"
    )

    with viewer.launch_passive(model, data) as handle:
        while handle.is_running():
            mujoco.mj_step(model, data)

            # Remove preview geoms from the previous frame.
            handle.user_scn.ngeom = 0

            for site_id, mesh_name in sites:
                _draw_site(
                    handle,
                    model,
                    data,
                    site_id=site_id,
                    rgba=_patch_color(mesh_name, args.alpha),
                    scale=args.scale,
                    wireframe=args.wireframe,
                    normal_length=args.normal_length,
                )

            handle.sync()


if __name__ == "__main__":
    main()
