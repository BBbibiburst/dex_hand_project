"""Bounded native-HD replay benchmark for the P29 dataset audit.

This tool never advances or changes a production policy.  It restores recorded
MuJoCo states from one historical episode, renders the native camera buffer,
encodes AV1, and writes a small auditable multimodal clip.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import cv2
import mujoco
import numpy as np
from PIL import Image

from source.envs.manipulation import (
    make_lift_env,
    make_nut_assembly_env,
    make_pick_place_env,
    make_push_env,
    make_stack_env,
)
from source.imitation.observation import real_observable_state


FACTORIES = {
    "lift": make_lift_env,
    "pick_place": make_pick_place_env,
    "stack": make_stack_env,
    "push": make_push_env,
    "nut_assembly": make_nut_assembly_env,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _task_name(payload: dict) -> str:
    return str(payload.get("metadata", {}).get("task") or "lift")


def _task_config(payload: dict, task: str) -> dict:
    saved = payload.get("metadata", {}).get("task_config")
    if saved:
        return dict(saved)
    if task in {"lift", "pick_place"}:
        return {"object_id": str(payload["object_id"]), "terminate_on_success": False}
    raise ValueError(f"Manifest has no task_config for {task!r}.")


def _clip_indices(archive: dict[str, np.ndarray], maximum_frames: int) -> np.ndarray:
    count = len(archive["qpos"])
    take = min(int(maximum_frames), count)
    success = np.flatnonzero(np.asarray(archive.get("task_success", ()), dtype=bool))
    center = int(success[0]) if len(success) else int(round(0.75 * (count - 1)))
    start = min(max(center - 3 * take // 4, 0), count - take)
    return np.arange(start, start + take, dtype=np.int64)


def _ffmpeg_executable() -> str:
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()


def _encode_command(executable: str, output: Path, width: int, height: int, fps: int):
    return [
        executable,
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s:v",
        f"{width}x{height}",
        "-r",
        str(fps),
        "-i",
        "pipe:0",
        "-an",
        "-c:v",
        "libaom-av1",
        "-crf",
        "35",
        "-b:v",
        "0",
        "-cpu-used",
        "8",
        "-row-mt",
        "1",
        "-pix_fmt",
        "yuv420p",
        "-y",
        str(output),
    ]


def _video_contract(path: Path) -> dict:
    capture = cv2.VideoCapture(str(path))
    try:
        opened = bool(capture.isOpened())
        frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) if opened else 0
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)) if opened else 0
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) if opened else 0
        fps = float(capture.get(cv2.CAP_PROP_FPS)) if opened else 0.0
        opencv_decoded = 0
        while opened:
            ok, _ = capture.read()
            if not ok:
                break
            opencv_decoded += 1
    finally:
        capture.release()
    decode = subprocess.run(
        [
            _ffmpeg_executable(),
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(path),
            "-map",
            "0:v:0",
            "-f",
            "null",
            "-",
            "-progress",
            "pipe:1",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=180,
    )
    progress = {}
    for line in decode.stdout.splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            progress[key] = value
    software_decoded = int(progress.get("frame", 0))
    return {
        "opened": opened,
        "declared_frames": frames,
        "opencv_decoded_frames": opencv_decoded,
        "software_decode_return_code": decode.returncode,
        "software_decoded_frames": software_decoded,
        "software_decode_complete": bool(decode.returncode == 0 and software_decoded == frames),
        "width": width,
        "height": height,
        "fps": fps,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--height", type=int, required=True)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--max-frames", type=int, default=80)
    parser.add_argument("--max-output-bytes", type=int, default=1_000_000_000)
    parser.add_argument("--camera", default="agentview")
    return parser


def run(args: argparse.Namespace) -> dict:
    started = time.perf_counter()
    manifest = args.manifest.resolve()
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    array_path = manifest.parent / str(payload.get("arrays", "episode.npz"))
    with np.load(array_path, allow_pickle=False) as source:
        archive = {name: np.asarray(source[name]) for name in source.files}
    indices = _clip_indices(archive, args.max_frames)
    task = _task_name(payload)
    control_dt = float(payload.get("metadata", {}).get("control_dt", 1.0 / args.fps))
    if abs(control_dt - 1.0 / args.fps) > 1e-6:
        raise ValueError(f"control_dt={control_dt} is incompatible with fps={args.fps}.")

    args.output.mkdir(parents=True, exist_ok=False)
    video_path = args.output / "agentview_av1.mp4"
    ffmpeg_log = args.output / "ffmpeg.stderr.log"
    env = FACTORIES[task](
        task_config=_task_config(payload, task),
        control_mode="position",
        enable_tactile_sensors=True,
        render_mode=None,
        control_dt=control_dt,
        episode_length=len(archive["qpos"]) + 20,
    )
    renderer = None
    process = None
    render_seconds = []
    pipe_seconds = []
    states = []
    images_for_preview: list[tuple[int, np.ndarray]] = []
    try:
        env.reset(seed=int(payload.get("seed", 0)))
        if env.data.qpos.shape != archive["qpos"].shape[1:]:
            raise ValueError(
                f"qpos contract differs: env={env.data.qpos.shape}, episode={archive['qpos'].shape}."
            )
        env.model.vis.global_.offwidth = int(args.width)
        env.model.vis.global_.offheight = int(args.height)
        renderer = mujoco.Renderer(env.model, height=args.height, width=args.width)
        command = _encode_command(
            _ffmpeg_executable(), video_path, args.width, args.height, args.fps
        )
        with ffmpeg_log.open("wb") as error_stream:
            process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=error_stream)
            assert process.stdin is not None
            for local_index, source_index in enumerate(indices):
                env.data.qpos[:] = archive["qpos"][source_index]
                env.data.qvel[:] = archive["qvel"][source_index]
                env.data.ctrl[:] = archive["ctrl"][source_index]
                mujoco.mj_forward(env.model, env.data)
                states.append(real_observable_state(env))
                render_started = time.perf_counter()
                renderer.update_scene(env.data, camera=args.camera)
                frame = renderer.render().copy()
                render_seconds.append(time.perf_counter() - render_started)
                if local_index in {0, len(indices) // 2, len(indices) - 1}:
                    images_for_preview.append((local_index, frame.copy()))
                pipe_started = time.perf_counter()
                process.stdin.write(np.ascontiguousarray(frame).tobytes())
                pipe_seconds.append(time.perf_counter() - pipe_started)
            process.stdin.close()
            return_code = process.wait(timeout=180)
        if return_code:
            raise RuntimeError(f"ffmpeg failed with return code {return_code}.")
    finally:
        if process is not None and process.poll() is None:
            process.kill()
        if renderer is not None:
            renderer.close()
        env.close()

    actions = np.asarray(archive["action"])[indices].astype(np.float32)
    state = np.asarray(states, dtype=np.float32)
    tactile = (
        np.asarray(archive["tactile"])[indices].astype(np.float32)
        if "tactile" in archive and archive["tactile"].shape[1:] == (712,)
        else None
    )
    arrays = {
        "source_frame_index": indices,
        "timestamp": np.arange(len(indices), dtype=np.float64) * control_dt,
        "sim_time": indices.astype(np.float64) * control_dt,
        "state": state,
        "action": actions,
        "object_position": np.asarray(archive["object_position"])[indices].astype(np.float32),
        "object_quaternion_wxyz": np.asarray(archive["object_quaternion_wxyz"])[indices].astype(
            np.float32
        ),
        "stage": np.asarray(archive["stage"])[indices],
        "success": np.asarray(archive["task_success"])[indices],
    }
    if tactile is not None:
        arrays["tactile"] = tactile
    np.savez_compressed(args.output / "modalities.npz", **arrays)
    for frame_index, image in images_for_preview:
        Image.fromarray(image).save(args.output / f"preview_{frame_index:04d}.png")

    video = _video_contract(video_path)
    output_bytes = sum(path.stat().st_size for path in args.output.rglob("*") if path.is_file())
    if output_bytes > args.max_output_bytes:
        raise RuntimeError(f"Output {output_bytes} exceeds limit {args.max_output_bytes}.")
    elapsed = time.perf_counter() - started
    result = {
        "schema_version": "p29-smoke-v1",
        "task": task,
        "object_id": payload.get("object_id"),
        "success_label": bool(payload.get("success")),
        "failure_reason": payload.get("failure_reason"),
        "source_manifest": str(manifest),
        "source_manifest_sha256": _sha256(manifest),
        "source_episode": str(array_path.resolve()),
        "source_episode_sha256": _sha256(array_path),
        "source_frames": int(len(archive["qpos"])),
        "clip_source_index_range": [int(indices[0]), int(indices[-1])],
        "frames": int(len(indices)),
        "physical_seconds": float(len(indices) * control_dt),
        "resolution_wh": [args.width, args.height],
        "fps": args.fps,
        "codec": {
            "name": "libaom-av1",
            "container": "mp4",
            "crf": 35,
            "cpu_used": 8,
            "pixel_format": "yuv420p",
        },
        "modalities": {
            "state": {"shape": list(state.shape), "dtype": str(state.dtype)},
            "action": {"shape": list(actions.shape), "dtype": str(actions.dtype)},
            "tactile": (
                {"shape": list(tactile.shape), "dtype": str(tactile.dtype)}
                if tactile is not None
                else {"status": "MISSING_IN_SOURCE; not synthesized"}
            ),
            "object_state": {"shape": [len(indices), 7], "dtype": "float32"},
        },
        "timing": {
            "wall_seconds": elapsed,
            "render_mean_ms": 1000 * float(np.mean(render_seconds)),
            "render_p50_ms": 1000 * float(np.percentile(render_seconds, 50)),
            "render_p95_ms": 1000 * float(np.percentile(render_seconds, 95)),
            "encoder_pipe_block_mean_ms": 1000 * float(np.mean(pipe_seconds)),
            "effective_physics_to_wall_ratio": float(len(indices) * control_dt / elapsed),
        },
        "storage": {
            "video_bytes": video_path.stat().st_size,
            "total_output_bytes": output_bytes,
            "video_bytes_per_physical_second": video_path.stat().st_size
            / (len(indices) * control_dt),
        },
        "integrity": {
            "video": video,
            "timestamp_strictly_increasing": bool(np.all(np.diff(arrays["timestamp"]) > 0)),
            "sim_time_strictly_increasing": bool(np.all(np.diff(arrays["sim_time"]) > 0)),
            "finite": bool(all(np.isfinite(value).all() for value in arrays.values())),
            "video_sha256": _sha256(video_path),
            "modalities_sha256": _sha256(args.output / "modalities.npz"),
        },
        "host": {"pid": os.getpid(), "mujoco_gl": os.environ.get("MUJOCO_GL")},
    }
    (args.output / "manifest.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return result


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    result = run(args)
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
