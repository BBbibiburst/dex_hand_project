"""Bounded online x264 stream with failure propagation and full decode checks."""

from __future__ import annotations

from pathlib import Path
import queue
import re
import shutil
import subprocess
import threading
import time

import imageio_ffmpeg
import numpy as np
from source.data.production_profile import PhaseProfile
from source.data.production_capture import capture_contract


def verify_video(path: Path, frames: int, *, capture=None) -> dict:
    spec = capture_contract(capture)
    started = time.monotonic()
    decode = subprocess.run([
        imageio_ffmpeg.get_ffmpeg_exe(), "-v", "info", "-i", str(path), "-map", "0:v:0",
        "-vf", "showinfo", "-pix_fmt", "rgb24", "-f", "hash", "-hash", "sha256",
        "-", "-progress", "pipe:1"],
        capture_output=True, text=True, timeout=60, check=True)
    progress = dict(line.split("=", 1) for line in decode.stdout.splitlines() if "=" in line)
    decoded = int(progress.get("frame", -1))
    pts = np.asarray([float(value) for value in re.findall(
        r"pts_time:([0-9.e+\-]+)", decode.stderr)])
    shapes = re.findall(r"\bs:(\d+)x(\d+)", decode.stderr)
    if decoded != frames or len(pts) != frames or not np.allclose(
            pts, np.arange(frames) / spec['fps'], atol=1e-4, rtol=0):
        raise ValueError("Decoded count/PTS differs from the RGB sampling contract")
    if len(shapes) != frames or any(shape != (str(spec['width']), str(spec['height'])) for shape in shapes):
        raise ValueError("Decoded frame dimensions differ from RGB capture contract")
    digest = progress.get("SHA256")
    if digest is None or len(digest) != 64:
        raise ValueError("Decoded video content digest missing")
    return {"decoded_frames": decoded, "pts_verified": True, "dimensions_verified": True,
            "decoded_rgb_sha256": digest, 'wall_seconds': time.monotonic() - started}


class OnlineVideo:
    def __init__(self, path: Path, *, timeout=180., max_bytes=1_000_000_000, capture=None):
        self.capture = capture_contract(capture)
        self.control_frames = 0
        self.control_indices = []
        self.profile = PhaseProfile()
        self.path = path
        self.deadline = time.monotonic() + timeout
        self.max_bytes = max_bytes
        self.frames = 0
        self.error = None
        self.closed = False
        self.queue = queue.Queue(maxsize=8)
        self.executable = imageio_ffmpeg.get_ffmpeg_exe()
        self.log = path.with_suffix(".ffmpeg.log").open("xb")
        self.process = self.profile.call('encoder_start', subprocess.Popen, [
            self.executable, "-nostdin", "-hide_banner", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", "rgb24", "-s:v", f"{self.capture['width']}x{self.capture['height']}",
            "-r", str(self.capture['fps']), "-i", "pipe:0", "-an", "-c:v", "libx264",
            "-crf", str(self.capture['crf']), "-preset", self.capture['preset'], "-threads", "2",
            "-pix_fmt", "yuv420p", "-n", str(path)],
            stdin=subprocess.PIPE, stderr=self.log)
        self.thread = threading.Thread(target=self._consume, daemon=True)
        self.thread.start()

    def _consume(self):
        try:
            while True:
                item = self.queue.get()
                if item is None:
                    break
                self.profile.call('encoder_pipe_write', self.process.stdin.write, item.tobytes())
            self.process.stdin.close()
        except Exception as error:
            self.error = error

    def _check(self):
        if self.error:
            raise RuntimeError("Online encoder failed") from self.error
        if time.monotonic() >= self.deadline:
            raise TimeoutError("Online video wall-clock budget exceeded")
        if self.path.exists() and self.path.stat().st_size > self.max_bytes:
            raise RuntimeError("Online video space budget exceeded")
        disk = shutil.disk_usage(self.path.parent)
        if disk.free < .2 * disk.total:
            raise RuntimeError("Disk reserve below 20 percent")

    def _put(self, value):
        while True:
            self._check()
            try:
                self.queue.put(value, timeout=.1)
                return
            except queue.Full:
                if not self.thread.is_alive():
                    raise RuntimeError("Encoder consumer stopped")

    def submit(self, frame):
        if self.closed:
            raise RuntimeError("Encoder already closed")
        if frame.shape != (self.capture['height'], self.capture['width'], 3) or frame.dtype != np.uint8:
            raise ValueError("RGB uint8 shape differs from declared capture dimensions")
        self.profile.call('frame_submit_inclusive', self._put, frame.copy())
        self.frames += 1

    @property
    def should_capture(self):
        return self.control_frames % (20 // self.capture['fps']) == 0

    def record_control(self, frame):
        if self.should_capture:
            if frame is None:
                raise ValueError('Missing RGB frame at declared sampling index')
            self.submit(frame)
            self.control_indices.append(self.control_frames)
        elif frame is not None:
            raise ValueError('Unexpected RGB frame outside declared sampling index')
        self.control_frames += 1

    def abort(self):
        if self.process.poll() is None:
            self.process.kill()
        self.process.wait(timeout=10)
        self.thread.join(timeout=5)
        self.log.close()
        self.closed = True

    def finish(self):
        try:
            started = time.monotonic()
            self._put(None)
            self.thread.join(timeout=max(.1, self.deadline - time.monotonic()))
            self._check()
            if self.thread.is_alive():
                raise TimeoutError("Encoder thread did not finish")
            result = self.process.wait(timeout=max(.1, self.deadline - time.monotonic()))
            if result != 0:
                raise RuntimeError(f"Encoder exited with {result}")
            self.profile.add('encoder_close_wait', time.monotonic() - started)
            verification = self.profile.call('video_decode_verify', verify_video, self.path, self.frames, capture=self.capture)
            self.closed = True
            self.log.close()
            return {"frames": self.frames, **verification,
                    "width": self.capture['width'], "height": self.capture['height'], "fps": self.capture['fps'],
                    "control_frames": self.control_frames or self.frames,
                    "codec": "libx264", "crf": self.capture['crf'], "queue_capacity": 8,
                    "pts_verified": True,
                    "bytes": self.path.stat().st_size, 'wall_profile': self.profile.snapshot()}
        except BaseException:
            self.abort()
            raise
