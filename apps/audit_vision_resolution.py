"""Audit fixed-camera RGB detail, cost, downsampling, and deployable fixed ROI."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

import mujoco
import numpy as np
from PIL import Image
from scipy import ndimage

from apps.trace_diffusion_rollout import _make_env
from source.grasping.contracts import DemonstrationEpisode
from source.imitation.vision_input import crop_normalized_rgb, resize_rgb_uint8


RESOLUTIONS = ((160, 120), (320, 240), (640, 480), (1920, 1080))


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timing-repeats", type=int, default=20)
    return parser


def _set_frame(env, episode, index):
    env.data.qpos[:] = episode.arrays["qpos"][index]
    env.data.qvel[:] = episode.arrays["qvel"][index]
    env.data.ctrl[:] = episode.arrays["ctrl"][index]
    mujoco.mj_forward(env.model, env.data)


def _rss_mib():
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith("VmRSS:"):
            return int(line.split()[1]) / 1024.0
    return None


def _bbox(mask):
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)]


def _expand(box, width, height, scale=3.0):
    x0, y0, x1, y1 = box
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    bw, bh = (x1 - x0) * scale, (y1 - y0) * scale
    return [
        max(0, int(cx - bw / 2)),
        max(0, int(cy - bh / 2)),
        min(width, int(cx + bw / 2)),
        min(height, int(cy + bh / 2)),
    ]


def _masks(segmentation, object_ids, robot_ids):
    geom = segmentation[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM)
    ids = segmentation[..., 0]
    obj = geom & np.isin(ids, tuple(object_ids))
    robot = geom & np.isin(ids, tuple(robot_ids))
    adjacency = ndimage.binary_dilation(obj, iterations=2) & robot
    return obj, robot, adjacency


def _camera_payload(env, camera_id):
    extent = float(env.model.stat.extent)
    return {
        "name": "agentview",
        "id": int(camera_id),
        "fovy_degrees": float(env.model.cam_fovy[camera_id]),
        "world_position_xyz": env.data.cam_xpos[camera_id].tolist(),
        "world_rotation_matrix": env.data.cam_xmat[camera_id].reshape(3, 3).tolist(),
        "model_extent_m": extent,
        "znear_multiplier": float(env.model.vis.map.znear),
        "zfar_multiplier": float(env.model.vis.map.zfar),
        "near_plane_m": float(env.model.vis.map.znear * extent),
        "far_plane_m": float(env.model.vis.map.zfar * extent),
    }


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    frames_dir = args.output / "frames"
    frames_dir.mkdir(exist_ok=True)
    crops_dir = args.output / "roi"
    crops_dir.mkdir(exist_ok=True)
    payload = json.loads(args.manifest.read_text())
    episode = DemonstrationEpisode.load(args.manifest)
    codes = episode.metadata["stage_codes"]
    stages = np.asarray(episode.arrays["stage"])
    requested = ("settle", "approach", "close", "hold", "lift", "verify")
    representatives = {}
    for name in requested:
        indices = np.flatnonzero(stages == int(codes[name]))
        representatives[name] = int(indices[len(indices) // 2])
    env = _make_env(payload, int(payload["frames"]) + 20)
    env.reset(seed=int(episode.metadata.get("source_seed", 0)))
    camera_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_CAMERA, "agentview")
    binding = env.task._require_bindings().objects["object"]
    # Task bindings contain collision geoms. Segmentation sees the separate
    # visual geom as well, so collect every compiled geom on the object body.
    object_ids = {
        index
        for index in range(env.model.ngeom)
        if int(env.model.geom_bodyid[index]) == binding.body_id
    }
    robot_ids = set(env.task._require_bindings().robot_geom_ids) - object_ids
    # The compiled default is 640x480. This only enlarges the offscreen color/
    # depth buffers; it does not alter physics, cameras, or task thresholds.
    env.model.vis.global_.offwidth = max(width for width, _ in RESOLUTIONS)
    env.model.vis.global_.offheight = max(height for _, height in RESOLUTIONS)
    _set_frame(env, episode, representatives["close"])
    camera = _camera_payload(env, camera_id)
    rows = []
    images = {}
    segmentations = {}
    rss_start = _rss_mib()
    try:
        for width, height in RESOLUTIONS:
            env.model.vis.global_.offwidth = width
            env.model.vis.global_.offheight = height
            rgb = mujoco.Renderer(env.model, height=height, width=width)
            seg = mujoco.Renderer(env.model, height=height, width=width)
            seg.enable_segmentation_rendering()
            try:
                for stage, index in representatives.items():
                    _set_frame(env, episode, index)
                    rgb.update_scene(env.data, camera="agentview")
                    image = rgb.render().copy()
                    seg.update_scene(env.data, camera="agentview")
                    segmentation = seg.render().copy()
                    images[(stage, width, height)] = image
                    segmentations[(stage, width, height)] = segmentation
                    Image.fromarray(image).save(frames_dir / f"{stage}_{width}x{height}.png")
                _set_frame(env, episode, representatives["close"])
                timings = []
                for _ in range(args.timing_repeats + 2):
                    started = time.perf_counter()
                    rgb.update_scene(env.data, camera="agentview")
                    rgb.render()
                    timings.append(time.perf_counter() - started)
                image = images[("close", width, height)]
                obj, robot, adjacency = _masks(
                    segmentations[("close", width, height)], object_ids, robot_ids
                )
                rows.append(
                    {
                        "resolution_wh": [width, height],
                        "shape": list(image.shape),
                        "dtype": str(image.dtype),
                        "range": [int(image.min()), int(image.max())],
                        "frame_bytes": int(image.nbytes),
                        "object_pixels": int(obj.sum()),
                        "object_fraction": float(obj.mean()),
                        "robot_pixels": int(robot.sum()),
                        "object_robot_adjacency_pixels": int(adjacency.sum()),
                        "segmentation_available": bool(obj.any()),
                        "object_bbox_xyxy": _bbox(obj),
                        "render_mean_ms": 1000 * float(np.mean(timings[2:])),
                        "render_p50_ms": 1000 * float(np.percentile(timings[2:], 50)),
                        "render_p95_ms": 1000 * float(np.percentile(timings[2:], 95)),
                    }
                )
            finally:
                rgb.close()
                seg.close()
    finally:
        env.close()
    rss_end = _rss_mib()

    # Render segmentation in an isolated default-size context. Some headless
    # EGL drivers return empty ID buffers when a segmentation renderer shares
    # a model whose offscreen buffer was enlarged to 1920x1080.
    for segmentation_width, segmentation_height in ((640, 480), (1920, 1080)):
        seg_env = _make_env(payload, int(payload["frames"]) + 20)
        seg_env.model.vis.global_.offwidth = segmentation_width
        seg_env.model.vis.global_.offheight = segmentation_height
        seg_env.reset(seed=int(episode.metadata.get("source_seed", 0)))
        seg_renderer = mujoco.Renderer(
            seg_env.model, height=segmentation_height, width=segmentation_width
        )
        seg_renderer.enable_segmentation_rendering()
        try:
            for stage, index in representatives.items():
                _set_frame(seg_env, episode, index)
                seg_renderer.update_scene(seg_env.data, camera="agentview")
                segmentations[(stage, segmentation_width, segmentation_height)] = (
                    seg_renderer.render().copy()
                )
        finally:
            seg_renderer.close()
            seg_env.close()

    for row in rows:
        width, height = row["resolution_wh"]
        source_width, source_height = ((1920, 1080) if width == 1920 else (640, 480))
        source_obj, source_robot, source_adj = _masks(
            segmentations[("close", source_width, source_height)], object_ids, robot_ids
        )
        source_box = _bbox(source_obj)
        area_scale = (width * height) / (source_width * source_height)
        row["object_pixels"] = int(round(source_obj.sum() * area_scale))
        row["object_fraction"] = float(source_obj.mean())
        row["robot_pixels"] = int(round(source_robot.sum() * area_scale))
        row["object_robot_adjacency_pixels"] = int(round(source_adj.sum() * area_scale))
        row["segmentation_available"] = True
        row["segmentation_metric_source"] = (
            f"{source_width}x{source_height} ID buffer projected by pixel area"
        )
        row["object_bbox_xyxy"] = [
            int(round(source_box[0] * width / source_width)),
            int(round(source_box[1] * height / source_height)),
            int(round(source_box[2] * width / source_width)),
            int(round(source_box[3] * height / source_height)),
        ]

    full_w, full_h = 1920, 1080
    # On this headless EGL stack, 1920x1080 RGB is valid but the second
    # simultaneous segmentation context can return an empty ID buffer. Use
    # the verified 640x480 segmentation projection as the fixed calibration
    # source and scale its normalized box; online RGB never reads this mask.
    calibration_w, calibration_h = 1920, 1080
    calibration_masks = []
    for stage in ("close", "hold"):
        obj, _, _ = _masks(
            segmentations[(stage, calibration_w, calibration_h)], object_ids, robot_ids
        )
        calibration_masks.append(obj)
    union = np.logical_or.reduce(calibration_masks)
    roi_calibration_px = _expand(_bbox(union), calibration_w, calibration_h, scale=3.0)
    roi_norm = [
        roi_calibration_px[0] / calibration_w,
        roi_calibration_px[1] / calibration_h,
        roi_calibration_px[2] / calibration_w,
        roi_calibration_px[3] / calibration_h,
    ]
    roi_px = [
        int(round(roi_norm[0] * full_w)),
        int(round(roi_norm[1] * full_h)),
        int(round(roi_norm[2] * full_w)),
        int(round(roi_norm[3] * full_h)),
    ]
    roi_rows = []
    for stage in requested:
        image = images[(stage, full_w, full_h)]
        obj, _, adjacency = _masks(
            segmentations[(stage, calibration_w, calibration_h)], object_ids, robot_ids
        )
        x0, y0, x1, y1 = roi_calibration_px
        roi_obj = obj[y0:y1, x0:x1]
        roi_adj = adjacency[y0:y1, x0:x1]
        crop = crop_normalized_rgb(image, roi_norm, output_width=160, output_height=120)
        Image.fromarray(crop).save(crops_dir / f"{stage}_fixed_roi_160x120.png")
        roi_rows.append(
            {
                "stage": stage,
                "object_coverage": float(roi_obj.sum() / max(obj.sum(), 1)),
                "adjacency_coverage": float(roi_adj.sum() / max(adjacency.sum(), 1)),
                "object_pixels_at_calibration_resolution": int(obj.sum()),
                "adjacency_pixels_at_calibration_resolution": int(adjacency.sum()),
            }
        )

    comparisons = []
    for stage in requested:
        direct = images[(stage, 160, 120)]
        down = resize_rgb_uint8(images[(stage, 1920, 1080)], width=160, height=120)
        delta = direct.astype(np.int16) - down.astype(np.int16)
        Image.fromarray(down).save(frames_dir / f"{stage}_1920_to_160.png")
        comparisons.append(
            {
                "stage": stage,
                "mae_uint8": float(np.mean(np.abs(delta))),
                "rmse_uint8": float(np.sqrt(np.mean(delta.astype(np.float64) ** 2))),
                "p95_abs_uint8": float(np.percentile(np.abs(delta), 95)),
                "maximum_abs_uint8": int(np.max(np.abs(delta))),
                "different_pixel_fraction": float(np.any(delta != 0, axis=2).mean()),
            }
        )

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(requested), 4, figsize=(16, 3 * len(requested)))
    for row, stage in enumerate(requested):
        for column, (width, height) in enumerate(RESOLUTIONS):
            axes[row, column].imshow(images[(stage, width, height)])
            axes[row, column].set_title(f"{stage}: {width}x{height}")
            axes[row, column].axis("off")
    fig.tight_layout()
    fig.savefig(args.output / "resolution_stage_comparison.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(len(requested), 3, figsize=(12, 3 * len(requested)))
    for row, stage in enumerate(requested):
        direct = images[(stage, 160, 120)]
        down = resize_rgb_uint8(images[(stage, 1920, 1080)], width=160, height=120)
        crop = np.asarray(Image.open(crops_dir / f"{stage}_fixed_roi_160x120.png"))
        for axis, image, title in zip(
            axes[row],
            (direct, down, crop),
            ("direct 160x120", "1920x1080 -> 160x120", "fixed ROI -> 160x120"),
            strict=True,
        ):
            axis.imshow(image)
            axis.set_title(f"{stage}: {title}")
            axis.axis("off")
    fig.tight_layout()
    fig.savefig(args.output / "direct_downsample_roi_comparison.png", dpi=180)
    plt.close(fig)

    result = {
        "manifest": str(args.manifest),
        "mujoco_version": mujoco.__version__,
        "render_backend_environment": os.environ.get("MUJOCO_GL", "default"),
        "camera": camera,
        "representative_source_indices": representatives,
        "preprocessing_audit": {
            "source": "MuJoCo RGB uint8 HWC",
            "current_policy": "uint8 HWC -> float32 CHW / 255; no resize/crop/color conversion",
            "high_to_baseline": "Pillow bilinear RGB 1920x1080 -> 160x120",
        },
        "resolutions": rows,
        "process_rss_start_mib": rss_start,
        "process_rss_end_mib": rss_end,
        "process_rss_change_mib": (
            None if rss_start is None or rss_end is None else rss_end - rss_start
        ),
        "gpu_renderer_memory_note": (
            "OpenGL/EGL allocations are not exposed by torch.cuda; frame-buffer byte counts "
            "and process RSS are reported instead."
        ),
        "direct_160_vs_1920_downsample": comparisons,
        "fixed_roi": {
            "definition": (
                "one normalized agentview calibration box derived offline from expert "
                "close+hold object masks; no online simulator-state query"
            ),
            "pixel_xyxy_at_1920x1080": roi_px,
            "calibration_resolution_wh": [calibration_w, calibration_h],
            "calibration_pixel_xyxy": roi_calibration_px,
            "normalized_xyxy": roi_norm,
            "stage_coverage": roi_rows,
        },
    }
    (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
