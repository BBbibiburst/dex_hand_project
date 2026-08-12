"""Render all validated grasp point clouds and trajectories as a contact sheet."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.lines import Line2D
import numpy as np

from source.evaluation.grasp_schema import TRAJECTORY_STABLE
import trimesh
import mujoco

from source.grasping.dex_hand_surface import load_posed_dex_hand_surface
from source.grasping.grasp_config_search import grasp_config_name, resolve_object
from source.grasping.mesh_pointcloud import sample_surface_pointcloud
from source.grasping.standalone_validator import (
    build_standalone_model,
    set_hand_fraction_targets,
    set_object_pose_for_hand_pose,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REPORT = PROJECT_ROOT / "configs/grasps/dex_hand/full_pipeline_benchmark.json"
DEFAULT_CONFIG_DIR = PROJECT_ROOT / "configs/grasps/dex_hand/dexevolve"
DEFAULT_OUTPUT = PROJECT_ROOT / "docs/grasp_trajectory_catalog.png"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--config-dir", type=Path, default=DEFAULT_CONFIG_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dataset", choices=("all", "ycb", "egad"), default="all")
    parser.add_argument("--columns", type=int, default=6)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--object-points", type=int, default=700)
    parser.add_argument("--hand-points-per-geom", type=int, default=45)
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--style", choices=("mujoco", "mesh"), default="mujoco")
    return parser.parse_args()


def _scene_line(scene, first: np.ndarray, second: np.ndarray, color, width: float = 0.004) -> None:
    if scene.ngeom >= scene.maxgeom:
        return
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(
        geom,
        mujoco.mjtGeom.mjGEOM_LINE,
        np.asarray([width, 0.0, 0.0]),
        np.zeros(3),
        np.eye(3).reshape(9),
        np.asarray(color, dtype=np.float32),
    )
    mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_LINE, width, first, second)
    scene.ngeom += 1


def _scene_sphere(scene, position: np.ndarray, color, radius: float = 0.006) -> None:
    if scene.ngeom >= scene.maxgeom:
        return
    mujoco.mjv_initGeom(
        scene.geoms[scene.ngeom],
        mujoco.mjtGeom.mjGEOM_SPHERE,
        np.asarray([radius, radius, radius]),
        position,
        np.eye(3).reshape(9),
        np.asarray(color, dtype=np.float32),
    )
    scene.ngeom += 1


def _scene_frame(scene, position: np.ndarray, rotation: np.ndarray, length: float = 0.045) -> None:
    for axis, color in zip(
        rotation.T,
        ((1.0, 0.1, 0.1, 1.0), (0.1, 1.0, 0.2, 1.0), (0.15, 0.4, 1.0, 1.0)),
        strict=True,
    ):
        _scene_line(scene, position, position + length * axis, color, width=0.004)


def _render_actual_model(object_id: str, payload: dict, *, width: int = 540, height: int = 420):
    """Render the actual MuJoCo hand/object models with trajectory overlays."""
    mesh_path = resolve_object(object_id)
    final_rotation = np.asarray(payload["hand_rotation_matrix"], dtype=np.float64)
    final_translation = np.asarray(payload["hand_translation"], dtype=np.float64)
    model, data = build_standalone_model(
        object_mesh=mesh_path,
        mesh_center=np.asarray(payload["mesh_center"], dtype=np.float64),
        mesh_scale=float(payload["mesh_scale"]),
        hand_translation=final_translation,
        hand_rotation_matrix=final_rotation,
        object_table_height=payload.get("object_table_height"),
        end_effector_name=payload.get("end_effector_name", "dex_hand"),
    )
    object_geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "validation_object_collision")
    table_geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "validation_table_visual")
    if object_geom >= 0:
        model.geom_rgba[object_geom] = (0.72, 0.30, 0.78, 1.0)
    if table_geom >= 0:
        model.geom_rgba[table_geom] = (0.94, 0.86, 0.72, 1.0)
    model.vis.headlight.ambient[:] = (0.58, 0.58, 0.58)
    model.vis.headlight.diffuse[:] = (0.82, 0.82, 0.82)
    model.vis.headlight.specular[:] = (0.18, 0.18, 0.18)
    fractions = np.asarray(payload["hand_actuator_fractions"], dtype=np.float64)
    for _ in range(500):
        set_hand_fraction_targets(model, data, fractions)
        set_object_pose_for_hand_pose(model, data, final_translation, final_rotation)
        mujoco.mj_step(model, data)
    set_object_pose_for_hand_pose(model, data, final_translation, final_rotation)

    object_rotation = final_rotation.T
    object_position = -(object_rotation @ final_translation)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = 0.5 * object_position
    camera.distance = 0.42
    camera.azimuth = -48
    camera.elevation = -20
    with mujoco.Renderer(model, height=height, width=width) as renderer:
        renderer.update_scene(data, camera=camera)
        base_pixels = renderer.render().copy()
        renderer.enable_segmentation_rendering()
        segmentation = renderer.render().copy()
        renderer.disable_segmentation_rendering()
        pixels = base_pixels.copy()
        pixels[segmentation[..., 0] < 0] = 255

        renderer.update_scene(data, camera=camera)
        scene = renderer.scene
        _scene_frame(scene, np.zeros(3), np.eye(3), length=0.05)
        _scene_frame(scene, object_position, object_rotation, length=0.04)
        approach = np.asarray(payload.get("approach_hand_translations", []), dtype=np.float64)
        closing = np.asarray(payload.get("grasp_hand_translations", []), dtype=np.float64)
        for path, color in (
            (approach, (0.1, 0.9, 0.2, 1.0)),
            (closing, (1.0, 0.15, 0.1, 1.0)),
        ):
            world_path = object_position + (object_rotation @ path.T).T if len(path) else path
            for first, second in zip(world_path[:-1], world_path[1:]):
                _scene_line(scene, first, second, color, width=0.004)
            if len(world_path):
                _scene_sphere(scene, world_path[0], color, radius=0.006)
        contacts = np.asarray(payload.get("contact_points", []), dtype=np.float64).reshape(-1, 3)
        normals = np.asarray(payload.get("contact_normals", []), dtype=np.float64).reshape(-1, 3)
        for index, point in enumerate(contacts):
            world_point = object_position + object_rotation @ point
            color = plt.get_cmap("tab10")(index % 10)
            _scene_sphere(scene, world_point, color, radius=0.006)
            if normals.shape == contacts.shape:
                world_normal = object_rotation @ normals[index]
                _scene_line(
                    scene, world_point, world_point + 0.025 * world_normal, color, width=0.003
                )
        overlay_pixels = renderer.render().copy()
        changed = np.any(np.abs(overlay_pixels.astype(int) - base_pixels.astype(int)) > 3, axis=2)
        pixels[changed] = overlay_pixels[changed]
        return pixels


def _config_path(row: dict, config_dir: Path) -> Path:
    reported = Path(row.get("config", ""))
    if reported.is_file():
        return reported
    fallback = config_dir / f"{grasp_config_name(row['object_id'])}.json"
    if fallback.is_file():
        return fallback
    raise FileNotFoundError(f"No grasp config for {row['object_id']}: {reported} or {fallback}")


def _set_equal_view(axis, arrays: list[np.ndarray]) -> None:
    visible = np.concatenate([array for array in arrays if len(array)], axis=0)
    low, high = visible.min(axis=0), visible.max(axis=0)
    center = 0.5 * (low + high)
    radius = max(0.04, 0.58 * float(np.ptp(visible, axis=0).max()))
    axis.set_xlim(center[0] - radius, center[0] + radius)
    axis.set_ylim(center[1] - radius, center[1] + radius)
    axis.set_zlim(center[2] - radius, center[2] + radius)
    axis.set_box_aspect((1, 1, 1))
    axis.view_init(elev=24, azim=-58)
    axis.set_proj_type("ortho")
    axis.set_xlabel("X (m)", fontsize=5, labelpad=-3)
    axis.set_ylabel("Y (m)", fontsize=5, labelpad=-3)
    axis.set_zlabel("Z (m)", fontsize=5, labelpad=-3)
    axis.tick_params(axis="both", which="major", labelsize=4, pad=-2)
    axis.grid(True, alpha=0.35, linewidth=0.4)


def _draw_path(axis, points: np.ndarray, color: str, *, linewidth: float) -> None:
    if not len(points):
        return
    axis.plot(*points.T, color=color, linewidth=linewidth, alpha=0.95)
    if len(points) > 1:
        index = max(0, len(points) // 2 - 1)
        delta = points[index + 1] - points[index]
        axis.quiver(
            *points[index],
            *delta,
            length=1.0,
            normalize=False,
            color=color,
            linewidth=1.7,
            arrow_length_ratio=0.35,
        )


def _render_tile(axis, object_id: str, payload: dict, index: int, args) -> None:
    mesh_path = resolve_object(object_id)
    cloud = sample_surface_pointcloud(
        mesh_path,
        count=args.object_points,
        seed=index,
    )
    object_points = cloud.points * float(payload["mesh_scale"])
    fractions = np.asarray(payload["hand_actuator_fractions"], dtype=np.float64)
    surface = load_posed_dex_hand_surface(
        actuator_fractions=fractions,
        max_points_per_geom=args.hand_points_per_geom,
        seed=index,
    )
    rotation = np.asarray(payload["hand_rotation_matrix"], dtype=np.float64)
    translation = np.asarray(payload["hand_translation"], dtype=np.float64)
    hand_points = surface.points @ rotation.T + translation
    approach = np.asarray(payload.get("approach_hand_translations", []), dtype=np.float64)
    closing = np.asarray(payload.get("grasp_hand_translations", []), dtype=np.float64)
    contacts = np.asarray(payload.get("contact_points", []), dtype=np.float64).reshape(-1, 3)
    contact_normals = np.asarray(payload.get("contact_normals", []), dtype=np.float64).reshape(
        -1, 3
    )

    loaded = trimesh.load_mesh(mesh_path, process=True)
    object_mesh = loaded.to_geometry() if isinstance(loaded, trimesh.Scene) else loaded
    vertices = (np.asarray(object_mesh.vertices, dtype=np.float64) - cloud.center) * float(
        payload["mesh_scale"]
    )
    object_face_ids = np.linspace(
        0, len(object_mesh.faces) - 1, min(1_500, len(object_mesh.faces)), dtype=np.int64
    )
    axis.add_collection3d(
        Poly3DCollection(
            vertices[np.asarray(object_mesh.faces)[object_face_ids]],
            facecolor="#6f91ad",
            edgecolor="none",
            alpha=0.25,
        )
    )
    hand_triangles = []
    for mesh in surface.meshes:
        face_ids = np.linspace(0, len(mesh.faces) - 1, min(100, len(mesh.faces)), dtype=np.int64)
        posed_vertices = np.asarray(mesh.vertices) @ rotation.T + translation
        hand_triangles.append(posed_vertices[np.asarray(mesh.faces)[face_ids]])
    axis.add_collection3d(
        Poly3DCollection(
            np.concatenate(hand_triangles),
            facecolor="#f4a261",
            edgecolor="#d97736",
            linewidth=0.08,
            alpha=0.46,
        )
    )
    axis.scatter(*object_points.T, s=1.4, color="#8ba6c1", alpha=0.20, depthshade=False)
    if len(approach):
        _draw_path(axis, approach, "#16a34a", linewidth=2.4)
        axis.scatter(*approach[0], color="#16a34a", marker="o", s=15, depthshade=False)
    if len(closing):
        _draw_path(axis, closing, "#dc2626", linewidth=2.4)
    if len(contacts):
        colors = plt.get_cmap("tab10")(np.arange(len(contacts)))
        axis.scatter(
            *contacts.T,
            color=colors,
            edgecolors="black",
            linewidths=0.45,
            s=24,
            depthshade=False,
        )
        if contact_normals.shape == contacts.shape:
            axis.quiver(
                *contacts.T,
                *contact_normals.T,
                length=0.014,
                normalize=True,
                color=colors,
                linewidth=0.9,
                arrow_length_ratio=0.25,
            )
    fingertip_centers = surface.fingertip_centers @ rotation.T + translation
    axis.scatter(
        *fingertip_centers.T,
        marker="x",
        s=22,
        color="#d97706",
        linewidths=1.1,
        depthshade=False,
    )
    _set_equal_view(
        axis,
        [object_points, hand_points, approach, closing, contacts, fingertip_centers],
    )
    dataset, name = object_id.split(":", 1)
    archive_size = len(payload.get("trajectory_stable_candidates", []))
    suffix = f" · {archive_size} candidates" if archive_size else ""
    axis.set_title(
        f"{index + 1:03d}  {dataset.upper()}\n{name.replace('_', ' ')}{suffix}",
        fontsize=6.8,
        color="#1f2937",
        pad=0,
        backgroundcolor="#edf6fb" if dataset == "ycb" else "#f3eefb",
    )


def render(args: argparse.Namespace) -> Path:
    if args.columns <= 0 or args.object_points < 32 or args.hand_points_per_geom <= 0:
        raise ValueError("columns and point counts must be positive")
    report = json.loads(args.report.resolve().read_text(encoding="utf-8"))
    rows = [row for row in report.get("objects", []) if row.get("status") == TRAJECTORY_STABLE]
    if args.dataset != "all":
        rows = [row for row in rows if row["object_id"].startswith(f"{args.dataset}:")]
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("--limit must be positive")
        rows = rows[: args.limit]
    if not rows:
        raise ValueError(f"No trajectory-stable objects found in {args.report}")

    columns = min(args.columns, len(rows))
    row_count = math.ceil(len(rows) / columns)
    figure = plt.figure(figsize=(columns * 3.0, row_count * 2.8 + 1.1), facecolor="white")
    figure.suptitle(
        f"Dex Hand grasp catalog — {len(rows)} objects\n"
        + (
            "Actual MuJoCo models, contacts, coordinate frames and trajectories"
            if args.style == "mujoco"
            else "Meshes, point clouds, contacts, approach and closing trajectories"
        ),
        fontsize=16,
        fontweight="bold",
        color="#172033",
        y=0.997,
    )
    legend = [
        Line2D(
            [],
            [],
            marker="s",
            linestyle="",
            color="#6f91ad",
            label="object model" if args.style == "mujoco" else "object mesh/cloud",
        ),
        Line2D(
            [],
            [],
            marker="s",
            linestyle="",
            color="#f4a261",
            label="Dex Hand model" if args.style == "mujoco" else "end-effector mesh",
        ),
        Line2D([], [], marker="o", linestyle="", color="#1f77b4", label="contacts"),
        Line2D([], [], marker="x", linestyle="", color="#d97706", label="contact centers"),
        Line2D([], [], color="#16a34a", linewidth=2, label="approach"),
        Line2D([], [], color="#dc2626", linewidth=2, label="closing"),
    ]
    figure.legend(
        handles=legend, loc="upper center", ncol=6, frameon=False, bbox_to_anchor=(0.5, 0.958)
    )
    failures: list[str] = []
    config_dir = args.config_dir.resolve()
    for index, row in enumerate(rows):
        axis = figure.add_subplot(
            row_count,
            columns,
            index + 1,
            **({"projection": "3d"} if args.style == "mesh" else {}),
        )
        try:
            path = _config_path(row, config_dir)
            payload = json.loads(path.read_text(encoding="utf-8"))
            if args.style == "mesh":
                _render_tile(axis, row["object_id"], payload, index, args)
            else:
                axis.imshow(_render_actual_model(row["object_id"], payload))
                axis.set_axis_off()
                dataset, name = row["object_id"].split(":", 1)
                axis.set_title(
                    f"{index + 1:03d}  {dataset.upper()} · {name.replace('_', ' ')}",
                    fontsize=7,
                    color="#1f2937",
                    pad=2,
                    backgroundcolor="#edf6fb" if dataset == "ycb" else "#f3eefb",
                )
        except Exception as exc:
            failures.append(f"{row['object_id']}: {exc}")
            axis.set_axis_off()
            if hasattr(axis, "text2D"):
                axis.text2D(0.5, 0.5, "render failed", ha="center", va="center", color="#b91c1c")
            else:
                axis.text(0.5, 0.5, "render failed", ha="center", va="center", color="#b91c1c")
            axis.set_title(f"{index + 1:03d}  {row['object_id']}", fontsize=7)
    figure.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.925, wspace=0.01, hspace=0.08)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=args.dpi, facecolor="white")
    plt.close(figure)
    print(f"Rendered {len(rows) - len(failures)}/{len(rows)} grasp tiles: {args.output}")
    if failures:
        print("\n".join(failures))
    return args.output


def main() -> None:
    render(parse_args())


if __name__ == "__main__":
    main()
