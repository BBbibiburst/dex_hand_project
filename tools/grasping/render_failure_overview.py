"""Render a classified contact sheet for failed Top100 grasp objects."""

from __future__ import annotations

import argparse
import json
import math
import textwrap
from collections import OrderedDict
from pathlib import Path

from PIL import Image, ImageDraw

from tools.render_object_catalog import find_obj, font, render_mujoco_scene


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FORMAL_SUCCESS = {"LATTICE_SUCCESS", "RL_SUCCESS"}
CATEGORIES = OrderedDict(
    (
        (
            "initial_pose_unstable",
            ("A  Initial pose moves or tips", "#c44747", "Re-place or replace object"),
        ),
        (
            "no_strict_grasp",
            ("B  No strict thumb-opposed grasp", "#7b5ba7", "Generation/object mismatch"),
        ),
        (
            "trajectory_reproduction",
            ("C  Grasp exists, execution loses it", "#d47a32", "Improve wrist path/templates"),
        ),
        (
            "marginal_grasp",
            ("D  Marginal grasp; PPO not converged", "#c19a28", "Near-threshold contact/lift"),
        ),
        (
            "weak_direct",
            ("E  Weak direct candidate; no RL progress", "#477fa8", "Replace or improve generation"),
        ),
    )
)


def _classify(row: dict, stability: dict | None) -> str:
    if stability is not None and not bool(stability.get("stable", False)):
        return "initial_pose_unstable"
    if row["status"] == "NO_GRASP_GENERATED":
        return "no_strict_grasp"
    if bool(row["grasp_success"]):
        return "trajectory_reproduction"
    if row["status"] == "RL_PROMISING":
        return "marginal_grasp"
    return "weak_direct"


def _short_name(object_id: str, width: int = 31) -> list[str]:
    name = object_id.split(":", 1)[-1].replace("_", " ")
    return textwrap.wrap(name, width=width, max_lines=2, placeholder="…")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary", type=Path, default=PROJECT_ROOT / "outputs/dex_hand_top100_v3/summary.json"
    )
    parser.add_argument(
        "--stability",
        type=Path,
        default=PROJECT_ROOT / "outputs/underactuated_top100_initial_stability.json",
    )
    parser.add_argument(
        "--manifest", type=Path, default=PROJECT_ROOT / "assets/maniskill/manifest.json"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "outputs/dex_hand_top100_v3/failure_overview.png",
    )
    args = parser.parse_args()

    summary = json.loads(args.summary.read_text(encoding="utf-8"))["results"]
    failures = [row for row in summary if row["status"] not in FORMAL_SUCCESS]
    stability_payload = json.loads(args.stability.read_text(encoding="utf-8"))
    stability = {row["object_id"]: row for row in stability_payload["results"]}
    records = json.loads(args.manifest.read_text(encoding="utf-8"))["objects"]
    records_by_id = {
        f"{record['dataset']}:{record['object_id']}": record for record in records
    }
    grouped = {key: [] for key in CATEGORIES}
    for row in failures:
        grouped[_classify(row, stability.get(row["object_id"]))].append(row)

    columns, tile_w, tile_h = 5, 300, 230
    header_h, section_header_h = 275, 58
    total_h = header_h + sum(
        section_header_h + math.ceil(len(rows) / columns) * tile_h
        for rows in grouped.values()
        if rows
    )
    sheet = Image.new("RGB", (columns * tile_w, total_h), "#f3f5f7")
    draw = ImageDraw.Draw(sheet)
    draw.text((35, 24), "DexHand Top100 — failed-object diagnosis", fill="#17222d", font=font(38, bold=True))
    draw.text(
        (37, 76),
        f"Formal failures: {len(failures)}/100  |  Lattice/RL formal success: {100-len(failures)}/100",
        fill="#4b5966",
        font=font(21),
    )
    box_w = 280
    for index, (key, (title, color, hint)) in enumerate(CATEGORIES.items()):
        x = 35 + index * (box_w + 12)
        draw.rounded_rectangle((x, 122, x + box_w, 244), radius=12, fill="white", outline=color, width=4)
        draw.text((x + 14, 137), str(len(grouped[key])), fill=color, font=font(32, bold=True))
        for line_index, line in enumerate(
            textwrap.wrap(title[3:], width=25, max_lines=2, placeholder="…")
        ):
            draw.text(
                (x + 65, 137 + line_index * 18),
                line,
                fill="#28343e",
                font=font(13, bold=True),
            )
        draw.text((x + 14, 199), hint, fill="#66727d", font=font(14))

    y = header_h
    for key, (title, color, hint) in CATEGORIES.items():
        rows = grouped[key]
        if not rows:
            continue
        draw.rectangle((0, y, sheet.width, y + section_header_h), fill=color)
        draw.text((25, y + 13), f"{title}  ·  {len(rows)} objects", fill="white", font=font(23, bold=True))
        y += section_header_h
        for index, row in enumerate(rows):
            grid_y, grid_x = divmod(index, columns)
            x0, y0 = grid_x * tile_w, y + grid_y * tile_h
            draw.rectangle((x0 + 5, y0 + 5, x0 + tile_w - 5, y0 + tile_h - 5), fill="white", outline="#d7dde2", width=2)
            record = records_by_id[row["object_id"]]
            try:
                # Preserve YCB/GSO scan textures and show the object in a real
                # MuJoCo tabletop camera instead of an abstract solid mesh.
                preview = render_mujoco_scene(find_obj(record), tile_w - 20, 130, color)
                sheet.paste(preview, (x0 + 10, y0 + 8))
            except (OSError, RuntimeError, ValueError):
                draw.text((x0 + 15, y0 + 55), "preview unavailable", fill="#9b3441", font=font(14))
            lines = _short_name(row["object_id"])
            for line_index, line in enumerate(lines):
                draw.text((x0 + 12, y0 + 143 + line_index * 18), line, fill="#202a33", font=font(14, bold=True))
            metric_y = y0 + 184
            draw.text(
                (x0 + 12, metric_y),
                f"G {float(row['grasp_best_lift_mm']):.1f}  L {float(row['lattice_best_lift_mm']):.1f} mm  RL {int(row['rl_updates'])}",
                fill="#56636e",
                font=font(13),
            )
            if key == "initial_pose_unstable":
                item = stability[row["object_id"]]
                draw.text(
                    (x0 + 12, metric_y + 20),
                    f"move {item['horizontal_displacement_m']*1000:.1f} mm  rotate {item['orientation_change_deg']:.1f}°",
                    fill=color,
                    font=font(13, bold=True),
                )
        y += math.ceil(len(rows) / columns) * tile_h

    args.output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(args.output, optimize=True)
    report = {
        "formal_failure_count": len(failures),
        "categories": {
            key: {
                "title": title,
                "count": len(grouped[key]),
                "object_ids": [row["object_id"] for row in grouped[key]],
            }
            for key, (title, _, _) in CATEGORIES.items()
        },
    }
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"[done] {args.output}")
    for key, (title, _, _) in CATEGORIES.items():
        print(f"  {len(grouped[key]):2d}  {title}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
