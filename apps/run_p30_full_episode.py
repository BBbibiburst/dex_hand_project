"""Profile one complete historical episode at native 1080p with bounded encoding."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import queue
import subprocess
import threading
import time

import mujoco
import numpy as np
import psutil

from apps.run_p29_hd_replay_smoke import (
    FACTORIES,
    _ffmpeg_executable,
    _sha256,
    _task_config,
    _task_name,
    _video_contract,
)
from source.imitation.observation import real_observable_state


ENCODERS = {
    "libaom-av1": [
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
    ],
    "libx264": ["-c:v", "libx264", "-crf", "23", "-preset", "medium"],
    "libx265": ["-c:v", "libx265", "-crf", "28", "-preset", "medium"],
}


def _encode_command(output: Path, width: int, height: int, fps: int, encoder: str):
    return [
        _ffmpeg_executable(),
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
        *ENCODERS[encoder],
        "-pix_fmt",
        "yuv420p",
        "-y",
        str(output),
    ]


class ResourceMonitor:
    def __init__(self) -> None:
        self.stop_event = threading.Event()
        self.rows: list[dict] = []
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        parent = psutil.Process()
        psutil.cpu_percent(interval=None)
        while not self.stop_event.wait(0.5):
            processes = [parent]
            try:
                processes.extend(parent.children(recursive=True))
            except psutil.Error:
                pass
            rss = 0
            for process in processes:
                try:
                    rss += process.memory_info().rss
                except psutil.Error:
                    pass
            self.rows.append(
                {
                    "wall_time": time.time(),
                    "system_cpu_percent": psutil.cpu_percent(interval=None),
                    "system_memory_percent": psutil.virtual_memory().percent,
                    "process_tree_rss_bytes": rss,
                }
            )

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> dict:
        self.stop_event.set()
        self.thread.join(timeout=2)
        return {
            "samples": len(self.rows),
            "system_cpu_percent_mean": float(
                np.mean([row["system_cpu_percent"] for row in self.rows])
            )
            if self.rows
            else None,
            "system_cpu_percent_max": max(
                (row["system_cpu_percent"] for row in self.rows), default=None
            ),
            "system_memory_percent_max": max(
                (row["system_memory_percent"] for row in self.rows), default=None
            ),
            "process_tree_rss_bytes_max": max(
                (row["process_tree_rss_bytes"] for row in self.rows), default=None
            ),
        }


class EncoderPipe:
    def __init__(self, process, mode: str, queue_size: int):
        self.process = process
        self.mode = mode
        self.queue_wait_seconds = 0.0
        self.pipe_write_seconds = 0.0
        self.error: Exception | None = None
        self.frames = 0
        self.queue = queue.Queue(maxsize=queue_size) if mode == "async" else None
        self.thread = None
        if self.queue is not None:
            self.thread = threading.Thread(target=self._consume, daemon=True)
            self.thread.start()

    def _write(self, frame: np.ndarray) -> None:
        started = time.perf_counter()
        assert self.process.stdin is not None
        self.process.stdin.write(np.ascontiguousarray(frame).tobytes())
        self.pipe_write_seconds += time.perf_counter() - started
        self.frames += 1

    def _consume(self) -> None:
        assert self.queue is not None
        try:
            while True:
                item = self.queue.get()
                if item is None:
                    break
                self._write(item)
        except Exception as exc:  # propagated to the producer at close
            self.error = exc

    def submit(self, frame: np.ndarray) -> None:
        if self.error:
            raise self.error
        if self.queue is None:
            self._write(frame)
            return
        started = time.perf_counter()
        self.queue.put(frame, block=True)
        self.queue_wait_seconds += time.perf_counter() - started

    def close(self) -> None:
        if self.queue is not None:
            self.queue.put(None)
            assert self.thread is not None
            self.thread.join()
        if self.error:
            raise self.error
        assert self.process.stdin is not None
        self.process.stdin.close()


def _safe_budget(path: Path) -> int:
    stat = os.statvfs(path)
    total = stat.f_blocks * stat.f_frsize
    available = stat.f_bavail * stat.f_frsize
    return int(available - 0.20 * total)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--encoder", choices=sorted(ENCODERS), default="libaom-av1")
    parser.add_argument("--mode", choices=("serial", "async"), default="serial")
    parser.add_argument("--queue-size", type=int, default=8)
    parser.add_argument("--raw-rgb-output", type=Path)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--max-output-bytes", type=int, default=15_000_000_000)
    parser.add_argument("--minimum-safe-budget-bytes", type=int, default=1_000_000_000_000)
    parser.add_argument("--camera", default="agentview")
    return parser


def run(args: argparse.Namespace) -> dict:
    total_started = time.perf_counter()
    if args.output.exists():
        raise FileExistsError(f"Refusing to reuse output: {args.output}")
    parent = args.output.parent
    parent.mkdir(parents=True, exist_ok=True)
    safe_budget_before = _safe_budget(parent)
    if safe_budget_before < args.minimum_safe_budget_bytes:
        raise RuntimeError(f"Safe disk budget is only {safe_budget_before} bytes.")
    args.output.mkdir()
    phases: dict[str, float] = {}

    started = time.perf_counter()
    manifest = args.manifest.resolve()
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    array_path = manifest.parent / str(payload.get("arrays", "episode.npz"))
    with np.load(array_path, allow_pickle=False) as source:
        archive = {name: np.asarray(source[name]) for name in source.files}
    indices = np.arange(len(archive["qpos"]), dtype=np.int64)
    if args.max_frames is not None:
        indices = indices[: args.max_frames]
    phases["source_load_seconds"] = time.perf_counter() - started

    task = _task_name(payload)
    control_dt = float(payload.get("metadata", {}).get("control_dt", 1.0 / args.fps))
    if abs(control_dt - 1.0 / args.fps) > 1e-6:
        raise ValueError(f"control_dt={control_dt} is incompatible with fps={args.fps}.")
    started = time.perf_counter()
    env = FACTORIES[task](
        task_config=_task_config(payload, task),
        control_mode="position",
        enable_tactile_sensors=True,
        render_mode=None,
        control_dt=control_dt,
        episode_length=len(archive["qpos"]) + 20,
    )
    phases["environment_create_seconds"] = time.perf_counter() - started
    renderer = process = monitor = raw_stream = None
    video_path = args.output / "agentview.mp4"
    ffmpeg_log = args.output / "ffmpeg.stderr.log"
    state_rows = []
    state_restore_seconds = render_seconds = raw_write_seconds = 0.0
    try:
        started = time.perf_counter()
        env.reset(seed=int(payload.get("seed", 0)))
        phases["environment_reset_seconds"] = time.perf_counter() - started
        if env.data.qpos.shape != archive["qpos"].shape[1:]:
            raise ValueError(
                f"qpos contract differs: env={env.data.qpos.shape}, episode={archive['qpos'].shape}."
            )
        env.model.vis.global_.offwidth = args.width
        env.model.vis.global_.offheight = args.height
        renderer = mujoco.Renderer(env.model, height=args.height, width=args.width)
        command = _encode_command(video_path, args.width, args.height, args.fps, args.encoder)
        started = time.perf_counter()
        error_stream = ffmpeg_log.open("wb")
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=error_stream)
        phases["encoder_process_start_seconds"] = time.perf_counter() - started
        pipe = EncoderPipe(process, args.mode, args.queue_size)
        if args.raw_rgb_output is not None:
            args.raw_rgb_output.parent.mkdir(parents=True, exist_ok=True)
            raw_stream = args.raw_rgb_output.open("xb")
        monitor = ResourceMonitor()
        monitor.start()
        for source_index in indices:
            started = time.perf_counter()
            env.data.qpos[:] = archive["qpos"][source_index]
            env.data.qvel[:] = archive["qvel"][source_index]
            env.data.ctrl[:] = archive["ctrl"][source_index]
            mujoco.mj_forward(env.model, env.data)
            state_rows.append(real_observable_state(env))
            state_restore_seconds += time.perf_counter() - started

            started = time.perf_counter()
            renderer.update_scene(env.data, camera=args.camera)
            frame = renderer.render().copy()
            render_seconds += time.perf_counter() - started
            if raw_stream is not None:
                started = time.perf_counter()
                raw_stream.write(np.ascontiguousarray(frame).tobytes())
                raw_write_seconds += time.perf_counter() - started
            pipe.submit(frame)
        started = time.perf_counter()
        pipe.close()
        return_code = process.wait(timeout=600)
        error_stream.close()
        phases["encoder_finalize_seconds"] = time.perf_counter() - started
        if return_code:
            raise RuntimeError(f"Encoder failed with return code {return_code}.")
        phases["state_restore_seconds"] = state_restore_seconds
        phases["render_seconds"] = render_seconds
        phases["encoder_pipe_write_seconds"] = pipe.pipe_write_seconds
        phases["encoder_queue_wait_seconds"] = pipe.queue_wait_seconds
        phases["raw_rgb_write_seconds"] = raw_write_seconds
    finally:
        if monitor is not None:
            resources = monitor.stop()
        else:
            resources = {}
        if raw_stream is not None:
            raw_stream.flush()
            os.fsync(raw_stream.fileno())
            raw_stream.close()
        if process is not None and process.poll() is None:
            process.kill()
        if renderer is not None:
            renderer.close()
        env.close()

    state = np.asarray(state_rows, dtype=np.float32)
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
        "action": np.asarray(archive["action"])[indices].astype(np.float32),
        "object_position": np.asarray(archive["object_position"])[indices].astype(np.float32),
        "object_quaternion_wxyz": np.asarray(archive["object_quaternion_wxyz"])[indices].astype(
            np.float32
        ),
        "stage": np.asarray(archive["stage"])[indices],
        "success": np.asarray(archive["task_success"])[indices],
    }
    if tactile is not None:
        arrays["tactile"] = tactile
    started = time.perf_counter()
    np.savez_compressed(args.output / "modalities.npz", **arrays)
    with video_path.open("rb+") as stream:
        os.fsync(stream.fileno())
    phases["modalities_save_and_video_fsync_seconds"] = time.perf_counter() - started
    started = time.perf_counter()
    video = _video_contract(video_path)
    phases["software_decode_validation_seconds"] = time.perf_counter() - started
    started = time.perf_counter()
    video_sha = _sha256(video_path)
    modalities_sha = _sha256(args.output / "modalities.npz")
    phases["content_hash_seconds"] = time.perf_counter() - started
    total_seconds = time.perf_counter() - total_started
    output_bytes = sum(path.stat().st_size for path in args.output.rglob("*") if path.is_file())
    if args.raw_rgb_output is not None and args.raw_rgb_output.exists():
        output_bytes += args.raw_rgb_output.stat().st_size
    if output_bytes > args.max_output_bytes:
        raise RuntimeError(f"Output {output_bytes} exceeds maximum {args.max_output_bytes}.")
    safe_budget_after = _safe_budget(parent)
    result = {
        "schema_version": "p30-full-episode-v1",
        "task": task,
        "object_id": payload.get("object_id"),
        "success_label": bool(payload.get("success")),
        "failure_reason": payload.get("failure_reason"),
        "source_manifest": str(manifest),
        "source_manifest_sha256": _sha256(manifest),
        "source_episode": str(array_path.resolve()),
        "source_episode_sha256": _sha256(array_path),
        "frames": len(indices),
        "physical_seconds": len(indices) * control_dt,
        "resolution_wh": [args.width, args.height],
        "fps": args.fps,
        "encoder": args.encoder,
        "encoder_arguments": ENCODERS[args.encoder],
        "pipeline_mode": args.mode,
        "queue_size": args.queue_size if args.mode == "async" else 0,
        "phases": phases,
        "resources": resources,
        "wall_seconds": total_seconds,
        "effective_physics_seconds_per_wall_second": len(indices) * control_dt / total_seconds,
        "storage": {
            "video_bytes": video_path.stat().st_size,
            "modalities_bytes": (args.output / "modalities.npz").stat().st_size,
            "total_output_bytes_including_raw_if_requested": output_bytes,
            "video_bytes_per_physical_second": video_path.stat().st_size
            / (len(indices) * control_dt),
            "safe_budget_before_bytes": safe_budget_before,
            "safe_budget_after_bytes": safe_budget_after,
        },
        "modalities": {
            "state_shape": list(state.shape),
            "action_shape": list(arrays["action"].shape),
            "tactile": list(tactile.shape) if tactile is not None else "MISSING_NOT_SYNTHESIZED",
        },
        "integrity": {
            "finite": bool(all(np.isfinite(value).all() for value in arrays.values())),
            "timestamp_strictly_increasing": bool(np.all(np.diff(arrays["timestamp"]) > 0)),
            "sim_time_strictly_increasing": bool(np.all(np.diff(arrays["sim_time"]) > 0)),
            "video": video,
            "video_sha256": video_sha,
            "modalities_sha256": modalities_sha,
        },
        "command": _encode_command(video_path, args.width, args.height, args.fps, args.encoder),
    }
    (args.output / "manifest.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return result


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    print(json.dumps(run(args), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
