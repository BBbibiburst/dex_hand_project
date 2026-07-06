# -*- coding: utf-8 -*-
"""Generate tactile taxel sites for dex-hand skin meshes."""

from __future__ import annotations

from dataclasses import dataclass
import struct
import tempfile
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np


DEX_HAND_TACTILE_SITE_PREFIX = "taxel_"
DEX_HAND_TACTILE_SENSOR_PREFIX = "touch_"
DEX_HAND_TACTILE_GROUP = "4"
DEFAULT_TAXEL_RADIUS = 0.0018


@dataclass(frozen=True)
class TactilePatchSpec:
    mesh_name: str
    rows: int
    cols: int

    @property
    def taxel_count(self) -> int:
        return self.rows * self.cols


@dataclass(frozen=True)
class FingerSegmentSurfaceFit:
    center: np.ndarray
    axis: np.ndarray
    section_x: np.ndarray
    section_y: np.ndarray
    axial_low: float
    axial_high: float
    outer_center: np.ndarray
    outer_radius_x: float
    outer_radius_y: float
    inner_center: np.ndarray
    inner_radius_x: float
    inner_radius_y: float
    arc_start: float
    arc_end: float


def dex_hand_tactile_patches() -> tuple[TactilePatchSpec, ...]:
    patches: list[TactilePatchSpec] = []
    for finger_id in range(5):
        patches.append(TactilePatchSpec(f"skin_{finger_id}_0_p", 7, 8))
        patches.append(TactilePatchSpec(f"skin_{finger_id}_1_p", 4, 8))
        patches.append(TactilePatchSpec(f"skin_{finger_id}_2_p", 4, 8))
    patches.append(TactilePatchSpec("skin_palm_p", 7, 16))
    return tuple(patches)


DEX_HAND_TACTILE_PATCHES = dex_hand_tactile_patches()
DEX_HAND_TACTILE_COUNT = sum(patch.taxel_count for patch in DEX_HAND_TACTILE_PATCHES)


def tactile_site_name(mesh_name: str, row: int, col: int) -> str:
    return f"{DEX_HAND_TACTILE_SITE_PREFIX}{mesh_name}_r{row:02d}_c{col:02d}"


def tactile_sensor_name(mesh_name: str, row: int, col: int) -> str:
    return f"{DEX_HAND_TACTILE_SENSOR_PREFIX}{mesh_name}_r{row:02d}_c{col:02d}"


def tactile_sensor_names(prefix: str = "") -> tuple[str, ...]:
    names: list[str] = []
    for patch in DEX_HAND_TACTILE_PATCHES:
        for row in range(patch.rows):
            for col in range(patch.cols):
                names.append(prefix + tactile_sensor_name(patch.mesh_name, row, col))
    return tuple(names)


def write_augmented_dex_hand_xml(
    hand_xml_path: Path,
    *,
    taxel_radius: float = DEFAULT_TAXEL_RADIUS,
) -> Path:
    """Create a temporary dex-hand XML with tactile sites and touch sensors."""
    tree = ET.parse(hand_xml_path)
    root = tree.getroot()
    mesh_dir = hand_xml_path.parent

    mesh_files = _mesh_file_map(root)
    body_by_geom = _body_by_named_geom(root)
    sensor_root = root.find("sensor")
    if sensor_root is None:
        sensor_root = ET.SubElement(root, "sensor")

    for patch in DEX_HAND_TACTILE_PATCHES:
        geom = body_by_geom.get(patch.mesh_name)
        if geom is None:
            raise ValueError(f"Skin geom {patch.mesh_name!r} was not found in {hand_xml_path}.")

        body, geom_elem = geom
        mesh_file = mesh_files.get(patch.mesh_name)
        if mesh_file is None:
            raise ValueError(f"Skin mesh asset {patch.mesh_name!r} was not found in {hand_xml_path}.")

        mesh_points = _surface_grid_points(
            mesh_dir / mesh_file,
            patch.rows,
            patch.cols,
            mesh_name=patch.mesh_name,
        )
        body_points = _transform_points(
            mesh_points,
            _parse_vec(geom_elem.get("pos"), 3, default=0.0),
            _parse_quat(geom_elem.get("quat")),
        )

        for row in range(patch.rows):
            for col in range(patch.cols):
                idx = row * patch.cols + col
                site_name = tactile_site_name(patch.mesh_name, row, col)
                sensor_name = tactile_sensor_name(patch.mesh_name, row, col)
                if body.find(f"./site[@name='{site_name}']") is None:
                    site = ET.SubElement(body, "site")
                    site.set("name", site_name)
                    site.set("type", "sphere")
                    site.set("pos", _format_vec(body_points[idx]))
                    site.set("size", f"{taxel_radius:.8g}")
                    site.set("rgba", "0 0.8 1 0.35")
                    site.set("group", DEX_HAND_TACTILE_GROUP)
                if sensor_root.find(f"./touch[@name='{sensor_name}']") is None:
                    touch = ET.SubElement(sensor_root, "touch")
                    touch.set("name", sensor_name)
                    touch.set("site", site_name)

    tmp = tempfile.NamedTemporaryFile(
        mode="wb",
        suffix="_tactile.xml",
        prefix="dex_hand_",
        dir=hand_xml_path.parent,
        delete=False,
    )
    with tmp:
        tree.write(tmp, encoding="utf-8", xml_declaration=True)
    return Path(tmp.name)


def _mesh_file_map(root: ET.Element) -> dict[str, str]:
    result: dict[str, str] = {}
    asset = root.find("asset")
    if asset is None:
        return result
    for mesh in asset.findall("mesh"):
        name = mesh.get("name")
        file_name = mesh.get("file")
        if name and file_name:
            result[name] = file_name
    return result


def _body_by_named_geom(root: ET.Element) -> dict[str, tuple[ET.Element, ET.Element]]:
    result: dict[str, tuple[ET.Element, ET.Element]] = {}
    worldbody = root.find("worldbody")
    if worldbody is None:
        return result

    def visit(body: ET.Element) -> None:
        for geom in body.findall("geom"):
            name = geom.get("name")
            if name:
                result[name] = (body, geom)
        for child in body.findall("body"):
            visit(child)

    for body in worldbody.findall("body"):
        visit(body)
    return result


def _read_stl_vertices(path: Path) -> np.ndarray:
    data = path.read_bytes()
    if _looks_like_binary_stl(data):
        tri_count = struct.unpack_from("<I", data, 80)[0]
        vertices = np.empty((tri_count * 3, 3), dtype=np.float64)
        offset = 84
        for tri_idx in range(tri_count):
            offset += 12
            for vertex_idx in range(3):
                vertices[tri_idx * 3 + vertex_idx] = struct.unpack_from("<3f", data, offset)
                offset += 12
            offset += 2
        return vertices
    return _read_ascii_stl_vertices(data.decode("utf-8", errors="ignore"))


def _looks_like_binary_stl(data: bytes) -> bool:
    if len(data) < 84:
        return False
    tri_count = struct.unpack_from("<I", data, 80)[0]
    return 84 + tri_count * 50 == len(data)


def _read_ascii_stl_vertices(text: str) -> np.ndarray:
    vertices: list[list[float]] = []
    for raw_line in text.splitlines():
        parts = raw_line.strip().split()
        if len(parts) == 4 and parts[0].lower() == "vertex":
            vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
    if not vertices:
        raise ValueError("STL file contains no vertices.")
    return np.asarray(vertices, dtype=np.float64)


def _surface_grid_points(
    path: Path,
    rows: int,
    cols: int,
    *,
    mesh_name: str = "",
) -> np.ndarray:
    vertices = _read_stl_vertices(path)
    if mesh_name.endswith("_2_p"):
        return _fingertip_skin_grid_points(vertices, rows, cols)
    if mesh_name == "skin_palm_p":
        return _palm_skin_grid_points(vertices, rows, cols)
    return _finger_segment_skin_grid_points(vertices, rows, cols)


def _finger_segment_skin_grid_points(
    vertices: np.ndarray,
    rows: int,
    cols: int,
) -> np.ndarray:
    """Generate taxels for skin_*_0_p and skin_*_1_p segment pads.

    These pads are closer to a thick partial elliptic cylinder than a flat
    patch: one direction follows the finger segment axis, the other follows the
    exposed outer ellipse arc.
    """
    fit = _fit_finger_segment_surfaces(vertices)
    theta_values = _ellipse_arc_mid_angles(
        fit.outer_radius_x,
        fit.outer_radius_y,
        cols,
        fit.arc_start,
        fit.arc_end,
    )
    axial_edges = np.linspace(fit.axial_low, fit.axial_high, rows + 1, dtype=np.float64)
    axial_values = 0.5 * (axial_edges[:-1] + axial_edges[1:])

    points: list[np.ndarray] = []
    for z_value in axial_values:
        for theta in theta_values:
            x_value = fit.outer_center[0] + fit.outer_radius_x * np.cos(theta)
            y_value = fit.outer_center[1] + fit.outer_radius_y * np.sin(theta)
            points.append(
                fit.center
                + z_value * fit.axis
                + x_value * fit.section_x
                + y_value * fit.section_y
            )
    return np.asarray(points, dtype=np.float64)


def _fit_finger_segment_surfaces(vertices: np.ndarray) -> FingerSegmentSurfaceFit:
    """Separate and fit inner/outer partial elliptic cylinders for segment skins."""
    centered = vertices - vertices.mean(axis=0)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    axis = vh[0]
    section_x = vh[1]
    section_y = vh[2]

    triangles = _vertices_as_triangles(vertices)
    face_centers = triangles.mean(axis=1)
    face_normals = _triangle_normals(triangles)
    face_centered = face_centers - vertices.mean(axis=0)
    face_section = np.column_stack([face_centered @ section_x, face_centered @ section_y])

    coarse_center = np.median(face_section, axis=0)
    face_rel = face_section - coarse_center
    coarse_radius_x = max(np.percentile(np.abs(face_rel[:, 0]), 95.0), 1e-9)
    coarse_radius_y = max(np.percentile(np.abs(face_rel[:, 1]), 95.0), 1e-9)
    r_norm = np.sqrt(
        (face_rel[:, 0] / coarse_radius_x) ** 2
        + (face_rel[:, 1] / coarse_radius_y) ** 2
    )

    radial_len = np.linalg.norm(face_rel, axis=1) + 1e-12
    radial = face_rel / radial_len[:, None]
    normal_section = np.column_stack([face_normals @ section_x, face_normals @ section_y])
    normal_radial = np.sum(normal_section * radial, axis=1)

    if np.std(r_norm) < 0.02:
        outer_faces = normal_radial > 0.0
    else:
        pos_vote = (r_norm > np.median(r_norm)).astype(np.float64)
        normal_vote = (normal_radial > 0.0).astype(np.float64)
        outer_faces = (0.6 * pos_vote + 0.4 * normal_vote) > 0.5

    if outer_faces.sum() < 0.05 * len(outer_faces) or (~outer_faces).sum() < 0.05 * len(outer_faces):
        outer_faces = r_norm >= np.percentile(r_norm, 60.0)

    outer_vertices = triangles[outer_faces].reshape(-1, 3)
    inner_vertices = triangles[~outer_faces].reshape(-1, 3)
    if len(inner_vertices) == 0:
        inner_vertices = vertices
    if len(outer_vertices) == 0:
        outer_vertices = vertices

    outer_center, outer_radius_x, outer_radius_y = _fit_axis_aligned_section_ellipse(
        outer_vertices,
        vertices.mean(axis=0),
        section_x,
        section_y,
    )
    inner_center, inner_radius_x, inner_radius_y = _fit_axis_aligned_section_ellipse(
        inner_vertices,
        vertices.mean(axis=0),
        section_x,
        section_y,
    )

    if outer_radius_x + outer_radius_y < inner_radius_x + inner_radius_y:
        outer_center, inner_center = inner_center, outer_center
        outer_radius_x, inner_radius_x = inner_radius_x, outer_radius_x
        outer_radius_y, inner_radius_y = inner_radius_y, outer_radius_y
        outer_vertices, inner_vertices = inner_vertices, outer_vertices

    outer_centered = outer_vertices - vertices.mean(axis=0)
    outer_section = np.column_stack([outer_centered @ section_x, outer_centered @ section_y])
    outer_rel = outer_section - outer_center
    angles = np.mod(
        np.arctan2(
            outer_rel[:, 1] / max(outer_radius_y, 1e-9),
            outer_rel[:, 0] / max(outer_radius_x, 1e-9),
        ),
        2.0 * np.pi,
    )
    arc_start, arc_end = _occupied_angle_arc(angles)

    axial = centered @ axis
    axial_low, axial_high = np.percentile(axial, [7.5, 92.5])
    return FingerSegmentSurfaceFit(
        center=vertices.mean(axis=0),
        axis=axis,
        section_x=section_x,
        section_y=section_y,
        axial_low=float(axial_low),
        axial_high=float(axial_high),
        outer_center=outer_center,
        outer_radius_x=float(outer_radius_x),
        outer_radius_y=float(outer_radius_y),
        inner_center=inner_center,
        inner_radius_x=float(inner_radius_x),
        inner_radius_y=float(inner_radius_y),
        arc_start=float(arc_start),
        arc_end=float(arc_end),
    )


def _fit_axis_aligned_section_ellipse(
    points: np.ndarray,
    center: np.ndarray,
    section_x: np.ndarray,
    section_y: np.ndarray,
) -> tuple[np.ndarray, float, float]:
    centered = points - center
    section = np.column_stack([centered @ section_x, centered @ section_y])
    ellipse_center = np.median(section, axis=0)
    rel = section - ellipse_center
    radius_x = max(np.percentile(np.abs(rel[:, 0]), 95.0), 1e-9)
    radius_y = max(np.percentile(np.abs(rel[:, 1]), 95.0), 1e-9)
    return ellipse_center, radius_x, radius_y


def _vertices_as_triangles(vertices: np.ndarray) -> np.ndarray:
    usable = (len(vertices) // 3) * 3
    if usable == 0:
        raise ValueError("STL vertices cannot form triangles.")
    return vertices[:usable].reshape(-1, 3, 3)


def _triangle_normals(triangles: np.ndarray) -> np.ndarray:
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    lengths = np.linalg.norm(normals, axis=1)
    valid = lengths > 1e-12
    normals[valid] /= lengths[valid, None]
    normals[~valid] = 0.0
    return normals


def _fingertip_skin_grid_points(
    vertices: np.ndarray,
    rows: int,
    cols: int,
) -> np.ndarray:
    """Generate taxels for rounded skin_*_2_p fingertip pads.

    Fingertips are closer to a half ellipsoid blended into a finger pulp than a
    cylinder.  We still use an axial/arc parameterization, but sample from the
    actual outer STL vertices so each axial row can follow the changing radius.
    """
    centered = vertices - vertices.mean(axis=0)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    axis = vh[0]
    section_x = vh[1]
    section_y = vh[2]

    axial = centered @ axis
    section = np.column_stack([centered @ section_x, centered @ section_y])
    center_2d = np.median(section, axis=0)
    rel = section - center_2d

    radius_x = max(np.percentile(np.abs(rel[:, 0]), 95.0), 1e-9)
    radius_y = max(np.percentile(np.abs(rel[:, 1]), 95.0), 1e-9)
    norm_radius = np.sqrt((rel[:, 0] / radius_x) ** 2 + (rel[:, 1] / radius_y) ** 2)
    outer_mask = norm_radius >= np.percentile(norm_radius, 58.0)
    if int(outer_mask.sum()) < rows * cols:
        outer_mask = norm_radius >= np.percentile(norm_radius, 45.0)

    outer_vertices = vertices[outer_mask]
    outer_axial = axial[outer_mask]
    outer_rel = rel[outer_mask]
    outer_angles = np.mod(
        np.arctan2(outer_rel[:, 1] / radius_y, outer_rel[:, 0] / radius_x),
        2.0 * np.pi,
    )

    arc_start, arc_end = _occupied_angle_arc(outer_angles)
    theta_values = _ellipse_arc_mid_angles(radius_x, radius_y, cols, arc_start, arc_end)
    axial_values = _linspace_midpoints(axial, rows)

    axial_step = max((np.percentile(axial, 92.5) - np.percentile(axial, 7.5)) / rows, 1e-9)
    theta_step = max((arc_end - arc_start) / cols, 1e-9)

    points: list[np.ndarray] = []
    for z_value in axial_values:
        for theta in theta_values:
            score = (
                ((outer_axial - z_value) / axial_step) ** 2
                + (_angle_distance(outer_angles, theta) / theta_step) ** 2
                - 0.08 * norm_radius[outer_mask]
            )
            points.append(outer_vertices[int(np.argmin(score))])
    return np.asarray(points, dtype=np.float64)


def _palm_skin_grid_points(
    vertices: np.ndarray,
    rows: int,
    cols: int,
) -> np.ndarray:
    """Generate taxels for the skin_palm_p palm pad."""
    return _axis_surface_grid_points(
        vertices,
        rows,
        cols,
        u_axis=2,
        v_axis=1,
        surface_axis=0,
        surface_side=1.0,
    )


def _axis_surface_grid_points(
    vertices: np.ndarray,
    rows: int,
    cols: int,
    *,
    u_axis: int,
    v_axis: int,
    surface_axis: int,
    surface_side: float,
) -> np.ndarray:
    u_values = _linspace_percentile(vertices[:, u_axis], cols)
    v_values = _linspace_percentile(vertices[:, v_axis], rows)
    surface_values = surface_side * vertices[:, surface_axis]
    threshold = np.percentile(surface_values, 65.0)
    surface_vertices = vertices[surface_values >= threshold]
    if surface_vertices.shape[0] < rows * cols:
        surface_vertices = vertices

    points: list[np.ndarray] = []
    uv = surface_vertices[:, [u_axis, v_axis]]
    for v in v_values:
        for u in u_values:
            distances = np.sum((uv - np.asarray([u, v])) ** 2, axis=1)
            points.append(surface_vertices[int(np.argmin(distances))])
    return np.asarray(points, dtype=np.float64)


def _occupied_angle_arc(angles: np.ndarray, *, bins: int = 96) -> tuple[float, float]:
    hist, _ = np.histogram(angles, bins=bins, range=(0.0, 2.0 * np.pi))
    occupied = hist > 0
    doubled_empty = np.concatenate([~occupied, ~occupied])

    best_start = 0
    best_len = 0
    cur_start = 0
    cur_len = 0
    for idx, is_empty in enumerate(doubled_empty):
        if is_empty:
            if cur_len == 0:
                cur_start = idx
            cur_len += 1
            if cur_len > best_len:
                best_start = cur_start
                best_len = cur_len
        else:
            cur_len = 0

    if best_len == 0 or best_len >= bins:
        return 0.0, 2.0 * np.pi

    bin_width = 2.0 * np.pi / bins
    start = ((best_start + best_len) % bins) * bin_width
    arc_bins = bins - min(best_len, bins)
    end = start + arc_bins * bin_width
    return start, end


def _ellipse_arc_mid_angles(
    radius_x: float,
    radius_y: float,
    count: int,
    start: float,
    end: float,
) -> np.ndarray:
    if count == 1:
        return np.asarray([(start + end) * 0.5], dtype=np.float64)

    samples = np.linspace(start, end, 512, dtype=np.float64)
    speed = np.sqrt(
        (radius_x * np.sin(samples)) ** 2 + (radius_y * np.cos(samples)) ** 2
    )
    cumulative = np.zeros_like(samples)
    cumulative[1:] = np.cumsum(0.5 * (speed[1:] + speed[:-1]) * np.diff(samples))
    targets = (np.arange(count, dtype=np.float64) + 0.5) * cumulative[-1] / count
    return np.interp(targets, cumulative, samples)


def _angle_distance(values: np.ndarray, target: float) -> np.ndarray:
    return np.abs((values - target + np.pi) % (2.0 * np.pi) - np.pi)


def _linspace_midpoints(values: np.ndarray, count: int) -> np.ndarray:
    low, high = np.percentile(values, [7.5, 92.5])
    if count == 1:
        return np.asarray([(low + high) * 0.5], dtype=np.float64)
    edges = np.linspace(low, high, count + 1, dtype=np.float64)
    return 0.5 * (edges[:-1] + edges[1:])


def _linspace_percentile(values: np.ndarray, count: int) -> np.ndarray:
    low, high = np.percentile(values, [7.5, 92.5])
    if count == 1:
        return np.asarray([(low + high) * 0.5], dtype=np.float64)
    return np.linspace(low, high, count, dtype=np.float64)


def _parse_vec(text: str | None, size: int, *, default: float) -> np.ndarray:
    if text is None:
        return np.full(size, default, dtype=np.float64)
    values = [float(value) for value in text.split()]
    if len(values) != size:
        raise ValueError(f"Expected {size} values, got {text!r}.")
    return np.asarray(values, dtype=np.float64)


def _parse_quat(text: str | None) -> np.ndarray:
    if text is None:
        return np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    quat = _parse_vec(text, 4, default=0.0)
    norm = np.linalg.norm(quat)
    if norm <= 1e-12:
        raise ValueError(f"Invalid zero quaternion {text!r}.")
    return quat / norm


def _quat_to_mat(quat: np.ndarray) -> np.ndarray:
    w, x, y, z = quat
    return np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _transform_points(points: np.ndarray, pos: np.ndarray, quat: np.ndarray) -> np.ndarray:
    return points @ _quat_to_mat(quat).T + pos


def _format_vec(values: np.ndarray) -> str:
    return " ".join(f"{float(value):.8g}" for value in values)
