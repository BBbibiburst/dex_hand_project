"""Audit one fixed camera ROI against independent verified Lift roots."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw

from apps.audit_vision_resolution import _masks, _set_frame
from apps.trace_diffusion_rollout import _make_env
from source.grasping.contracts import DemonstrationEpisode


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-a", type=Path, required=True)
    parser.add_argument("--roi-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    previews = args.output / "previews"
    previews.mkdir(exist_ok=True)
    manifests = json.loads(args.source_a.read_text())["accepted_manifests"]
    roi = json.loads(args.roi_summary.read_text())["fixed_roi"]["normalized_xyxy"]
    x0, y0, x1, y1 = (
        int(round(roi[0] * args.width)),
        int(round(roi[1] * args.height)),
        int(round(roi[2] * args.width)),
        int(round(roi[3] * args.height)),
    )
    rows = []
    for root_index, manifest in enumerate(manifests):
        manifest = Path(manifest)
        payload = json.loads(manifest.read_text())
        episode = DemonstrationEpisode.load(manifest)
        stages = np.asarray(episode.arrays["stage"])
        codes = episode.metadata["stage_codes"]
        env = _make_env(payload, int(payload["frames"]) + 20)
        env.model.vis.global_.offwidth = args.width
        env.model.vis.global_.offheight = args.height
        env.reset(seed=int(episode.metadata.get("source_seed", 0)))
        binding = env.task._require_bindings().objects["object"]
        object_ids = {
            index
            for index in range(env.model.ngeom)
            if int(env.model.geom_bodyid[index]) == binding.body_id
        }
        robot_ids = set(env.task._require_bindings().robot_geom_ids) - object_ids
        rgb = mujoco.Renderer(env.model, height=args.height, width=args.width)
        seg = mujoco.Renderer(env.model, height=args.height, width=args.width)
        seg.enable_segmentation_rendering()
        try:
            for stage_name in ("close", "hold"):
                indices = np.flatnonzero(stages == int(codes[stage_name]))
                for position, frame_index in (("start", indices[0]), ("mid", indices[len(indices) // 2])):
                    _set_frame(env, episode, int(frame_index))
                    rgb.update_scene(env.data, camera="agentview")
                    image = rgb.render().copy()
                    seg.update_scene(env.data, camera="agentview")
                    segmentation = seg.render().copy()
                    obj, _, adjacency = _masks(segmentation, object_ids, robot_ids)
                    roi_obj = obj[y0:y1, x0:x1]
                    roi_adj = adjacency[y0:y1, x0:x1]
                    row = {
                        "root": root_index,
                        "manifest": str(manifest),
                        "stage": stage_name,
                        "position": position,
                        "frame": int(frame_index),
                        "object_pixels": int(obj.sum()),
                        "object_coverage": float(roi_obj.sum() / max(obj.sum(), 1)),
                        "adjacency_pixels": int(adjacency.sum()),
                        "adjacency_coverage": float(
                            roi_adj.sum() / max(adjacency.sum(), 1)
                        ),
                    }
                    rows.append(row)
                    if position == "mid":
                        preview = Image.fromarray(image)
                        draw = ImageDraw.Draw(preview)
                        draw.rectangle((x0, y0, x1, y1), outline=(255, 30, 30), width=5)
                        preview.resize((960, 540)).save(
                            previews / f"root_{root_index:02d}_{stage_name}.jpg", quality=90
                        )
        finally:
            rgb.close()
            seg.close()
            env.close()
    summary = {
        "roots": len(manifests),
        "evaluated_frames": len(rows),
        "resolution_wh": [args.width, args.height],
        "fixed_roi_normalized_xyxy": roi,
        "fixed_roi_pixel_xyxy": [x0, y0, x1, y1],
        "object_coverage_mean": float(np.mean([row["object_coverage"] for row in rows])),
        "object_coverage_min": float(np.min([row["object_coverage"] for row in rows])),
        "adjacency_coverage_mean": float(
            np.mean([row["adjacency_coverage"] for row in rows])
        ),
        "adjacency_coverage_min": float(
            np.min([row["adjacency_coverage"] for row in rows])
        ),
        "full_object_frames": int(sum(row["object_coverage"] >= 0.999 for row in rows)),
        "rows": rows,
        "note": "Segmentation is used only for offline audit, never for online ROI placement.",
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({key: value for key, value in summary.items() if key != "rows"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
