"""Render the local YCB and EGAD assets as one labelled contact sheet."""

from __future__ import annotations

import argparse
import html
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = PROJECT_ROOT / "assets" / "maniskill" / "manifest.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "docs"

CATEGORY_COLORS = {
    "Food": "#d99b45",
    "Containers & tableware": "#4c9dc2",
    "Tools & hardware": "#9a78c2",
    "Sports": "#50a66f",
    "Toys & components": "#d36f82",
    "Household": "#778899",
    "EGAD complexity A": "#e84c3d",
    "EGAD complexity B": "#ef8b2c",
    "EGAD complexity C": "#e2bd28",
    "EGAD complexity D": "#48a868",
    "EGAD complexity E": "#27aeb5",
    "EGAD complexity F": "#3978cf",
    "EGAD complexity G": "#8658bf",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--columns", type=int, default=5)
    parser.add_argument("--tile-width", type=int, default=720)
    parser.add_argument("--tile-height", type=int, default=350)
    return parser.parse_args()


def font(size: int, *, bold: bool = False) -> ImageFont.ImageFont:
    candidates = [
        Path("C:/Windows/Fonts") / ("arialbd.ttf" if bold else "arial.ttf"),
        Path("/usr/share/fonts/truetype/dejavu")
        / ("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"),
    ]
    for candidate in candidates:
        if candidate.is_file():
            return ImageFont.truetype(str(candidate), size)
    return ImageFont.load_default()


def ycb_category(object_id: str) -> str:
    name = object_id.lower()
    if any(word in name for word in (
        "can", "cracker", "sugar", "soup", "mustard", "tuna", "pudding",
        "gelatin", "meat", "banana", "strawberry", "apple", "lemon",
        "peach", "pear", "orange", "plum",
    )):
        return "Food"
    if any(word in name for word in (
        "pitcher", "bottle", "bowl", "mug", "plate", "fork", "spoon",
        "knife", "spatula", "skillet", "cups",
    )):
        return "Containers & tableware"
    if any(word in name for word in (
        "drill", "scissors", "padlock", "marker", "wrench", "screwdriver",
        "hammer", "clamp",
    )):
        return "Tools & hardware"
    if any(word in name for word in (
        "soccer", "softball", "baseball", "tennis", "racquetball", "golf",
    )):
        return "Sports"
    if any(word in name for word in (
        "dice", "marbles", "blocks", "peg", "airplane", "lego", "rubiks",
    )):
        return "Toys & components"
    return "Household"


def category(record: dict) -> str:
    if record["dataset"] == "egad":
        return f"EGAD complexity {record['object_id'][0]}"
    return ycb_category(record["object_id"])


def read_obj(path: Path) -> tuple[np.ndarray, np.ndarray]:
    vertices: list[list[float]] = []
    faces: list[list[int]] = []
    with path.open("r", encoding="utf-8", errors="ignore") as stream:
        for line in stream:
            if line.startswith("v "):
                values = line.split()
                vertices.append([float(values[1]), float(values[2]), float(values[3])])
            elif line.startswith("f "):
                indices = [int(value.split("/")[0]) for value in line.split()[1:]]
                indices = [index - 1 if index > 0 else len(vertices) + index for index in indices]
                for offset in range(1, len(indices) - 1):
                    faces.append([indices[0], indices[offset], indices[offset + 1]])
    if not vertices or not faces:
        raise ValueError(f"No triangular geometry found in {path}")
    return np.asarray(vertices, dtype=np.float32), np.asarray(faces, dtype=np.int32)


def find_obj(record: dict, manifest_path: Path) -> Path:
    source = Path(record["source_path"])
    source = source if source.is_absolute() else PROJECT_ROOT / source
    candidates = [
        source / relative
        for relative in record.get("model_files", [])
        if Path(relative).suffix.lower() == ".obj"
    ]
    candidates.extend(sorted(source.glob("*.obj")))
    if not candidates:
        destination = Path(record["project_path"])
        destination = destination if destination.is_absolute() else PROJECT_ROOT / destination
        candidates.extend(sorted(destination.glob("*.obj")))
    if not candidates:
        raise FileNotFoundError(f"No OBJ mesh for {record['dataset']}:{record['object_id']}")
    return next((path for path in candidates if path.name == "textured.obj"), candidates[0])


def projected_mesh(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    vertices, faces = read_obj(path)
    vertices -= (vertices.min(axis=0) + vertices.max(axis=0)) / 2
    scale = np.linalg.norm(vertices.max(axis=0) - vertices.min(axis=0))
    vertices /= max(float(scale), 1e-8)
    azimuth, elevation = math.radians(-38), math.radians(24)
    rz = np.array([
        [math.cos(azimuth), -math.sin(azimuth), 0],
        [math.sin(azimuth), math.cos(azimuth), 0],
        [0, 0, 1],
    ])
    rx = np.array([
        [1, 0, 0],
        [0, math.cos(elevation), -math.sin(elevation)],
        [0, math.sin(elevation), math.cos(elevation)],
    ])
    rotated = vertices @ (rz @ rx).T
    if len(faces) > 7000:
        faces = faces[np.linspace(0, len(faces) - 1, 7000, dtype=int)]
    triangles = rotated[faces]
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    lengths = np.linalg.norm(normals, axis=1)
    visible = lengths > 1e-10
    triangles, normals, lengths = triangles[visible], normals[visible], lengths[visible]
    normals /= lengths[:, None]
    light = np.array([-0.35, -0.5, 0.79])
    shades = np.clip(0.28 + 0.72 * np.abs(normals @ light), 0, 1)
    order = np.argsort(triangles[:, :, 2].mean(axis=1))
    return triangles[order, :, :2], shades[order], triangles[order, :, 2].mean(axis=1)


def render_shape(path: Path, width: int, height: int, tint: str) -> Image.Image:
    image = Image.new("RGB", (width, height), "#f6f7f9")
    draw = ImageDraw.Draw(image)
    triangles, shades, _ = projected_mesh(path)
    xy = triangles.reshape(-1, 2)
    low, high = xy.min(axis=0), xy.max(axis=0)
    extent = np.maximum(high - low, 1e-8)
    scale = min((width - 26) / extent[0], (height - 20) / extent[1])
    center = (low + high) / 2
    base = tuple(int(tint[index : index + 2], 16) for index in (1, 3, 5))
    for triangle, shade in zip(triangles, shades):
        points = (triangle - center) * scale
        points[:, 0] += width / 2
        points[:, 1] = height / 2 - points[:, 1]
        color = tuple(int(channel * (0.48 + 0.52 * float(shade))) for channel in base)
        draw.polygon([tuple(point) for point in points], fill=color)
    return image


def render_mujoco_scene(path: Path, width: int, height: int, tint: str) -> Image.Image:
    """Render one mesh on a lit MuJoCo tabletop using an offscreen camera."""
    import mujoco

    vertices, _ = read_obj(path)
    low, high = vertices.min(axis=0), vertices.max(axis=0)
    extent = np.maximum(high - low, 1e-8)
    scale = 0.15 / float(extent.max())
    center = (low + high) / 2
    mesh_position = -center * scale
    mesh_position[2] += float(extent[2] * scale / 2 + 0.006)
    rgb = tuple(int(tint[index : index + 2], 16) / 255 for index in (1, 3, 5))
    mesh_path = html.escape(path.resolve().as_posix(), quote=True)
    texture_path = path.parent / "texture_map.png"
    if not texture_path.is_file():
        texture_path = path.parent / "material_0.png"
    if texture_path.is_file():
        texture_asset = (
            f'<texture name="object_tex" type="2d" '
            f'file="{html.escape(texture_path.resolve().as_posix(), quote=True)}"/>\n'
            '    <material name="object_mat" texture="object_tex" '
            'specular=".2" shininess=".25"/>'
        )
        object_material = 'material="object_mat"'
    else:
        texture_asset = ""
        object_material = f'rgba="{rgb[0]} {rgb[1]} {rgb[2]} 1"'
    xml = f"""
<mujoco model="object_catalog">
  <option gravity="0 0 -9.81"/>
  <visual>
    <quality shadowsize="2048"/>
    <headlight ambient="0.25 0.25 0.25" diffuse="0.7 0.7 0.7" specular="0.25 0.25 0.25"/>
  </visual>
  <asset>
    <mesh name="object" file="{mesh_path}" scale="{scale} {scale} {scale}"/>
    <texture name="sky" type="skybox" builtin="gradient" rgb1=".72 .82 .92"
             rgb2=".96 .97 .98" width="512" height="3072"/>
    <texture name="floor_tex" type="2d" builtin="checker" rgb1=".72 .76 .79"
             rgb2=".86 .88 .90" width="256" height="256"/>
    <material name="floor_mat" texture="floor_tex" texrepeat="3 3" reflectance=".08"/>
    {texture_asset}
  </asset>
  <worldbody>
    <light pos="-1 -1 2" dir=".45 .45 -1" directional="true" castshadow="true"/>
    <light pos="1 -.4 1.2" dir="-.5 .2 -1" directional="true" diffuse=".35 .35 .35"/>
    <geom name="table" type="box" size=".32 .28 .025" pos="0 0 -.025"
          material="floor_mat" friction=".8 .02 .002"/>
    <body name="object" pos="{mesh_position[0]} {mesh_position[1]} {mesh_position[2]}">
      <geom type="mesh" mesh="object" {object_material}
            contype="1" conaffinity="1"/>
    </body>
  </worldbody>
</mujoco>
"""
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = (0, 0, 0.065)
    camera.distance = 0.48
    camera.azimuth = 135
    camera.elevation = -27
    with mujoco.Renderer(model, height=height, width=width) as renderer:
        renderer.update_scene(data, camera=camera)
        pixels = renderer.render()
    return Image.fromarray(pixels)


def display_name(record: dict) -> str:
    object_id = record["object_id"]
    if record["dataset"] == "ycb":
        return object_id.split("_", 1)[-1].replace("_", " ")
    return f"procedural shape {object_id}"


def render_dataset(
    records: list[dict],
    dataset: str,
    output: Path,
    *,
    columns: int,
    tile_width: int,
    tile_height: int,
    manifest_path: Path,
) -> list[str]:
    rows = math.ceil(len(records) / columns)
    header_height = 170
    sheet = Image.new(
        "RGB",
        (columns * tile_width, header_height + rows * tile_height),
        "white",
    )
    draw = ImageDraw.Draw(sheet)
    dataset_name = "YCB real-scanned objects" if dataset == "ycb" else "EGAD evaluation objects"
    draw.text((36, 24), f"{dataset_name} — {len(records)} objects", fill="#18222d", font=font(42, bold=True))
    draw.text(
        (38, 82),
        "Each tile: normalized shape view (left)  |  MuJoCo tabletop render (right)",
        fill="#405060",
        font=font(25),
    )
    draw.text(
        (38, 126),
        "Ordered by semantic category." if dataset == "ycb"
        else "Ordered by complexity A–G, then difficulty 0–6.",
        fill="#607080",
        font=font(20),
    )

    title_font, label_font, small_font = font(18, bold=True), font(16), font(14)
    failures: list[str] = []
    for index, record in enumerate(records):
        row, column = divmod(index, columns)
        x, y = column * tile_width, header_height + row * tile_height
        color = CATEGORY_COLORS[record["_category"]]
        draw.rectangle((x + 5, y + 5, x + tile_width - 5, y + tile_height - 5), fill="#ffffff", outline=color, width=5)
        try:
            mesh_path = find_obj(record, manifest_path)
            preview_width = (tile_width - 30) // 2
            preview = render_shape(mesh_path, preview_width, 220, color)
            scene = render_mujoco_scene(mesh_path, preview_width, 220, color)
            sheet.paste(preview, (x + 10, y + 48))
            sheet.paste(scene, (x + 18 + preview_width, y + 48))
            draw.line((x + 14 + preview_width, y + 52, x + 14 + preview_width, y + 264), fill="#d6dbe0", width=2)
        except (OSError, ValueError, RuntimeError) as exc:
            failures.append(f"{record['dataset']}:{record['object_id']}: {exc}")
            draw.text((x + 18, y + 130), "preview unavailable", fill="#aa3344", font=label_font)
        draw.text((x + 15, y + 14), f"{index + 1:03d}  {record['dataset'].upper()} · {record['object_id']}", fill="#202830", font=title_font)
        draw.text((x + 15, y + 276), display_name(record)[:42], fill="#263442", font=label_font)
        draw.text((x + 15, y + 308), record["_category"], fill=color, font=small_font)

    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output, optimize=True)
    print(f"Rendered {len(records) - len(failures)}/{len(records)} {dataset.upper()} previews: {output}")
    return failures


def main() -> int:
    args = parse_args()
    manifest_path = args.manifest.resolve()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = payload["objects"]
    for record in records:
        record["_category"] = category(record)
    ycb_order = [
        "Food", "Containers & tableware", "Tools & hardware",
        "Sports", "Toys & components", "Household",
    ]
    order = {name: index for index, name in enumerate(ycb_order)}
    order.update({f"EGAD complexity {letter}": 20 + index for index, letter in enumerate("ABCDEFG")})
    records.sort(key=lambda item: (order[item["_category"]], item["object_id"]))

    output_dir = args.output_dir.resolve()
    failures: list[str] = []
    for dataset in ("ycb", "egad"):
        subset = [record for record in records if record["dataset"] == dataset]
        failures.extend(
            render_dataset(
                subset,
                dataset,
                output_dir / f"object_catalog_{dataset}.png",
                columns=args.columns,
                tile_width=args.tile_width,
                tile_height=args.tile_height,
                manifest_path=manifest_path,
            )
        )
    if failures:
        print("\n".join(failures))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
