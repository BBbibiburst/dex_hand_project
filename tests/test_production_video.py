import time

import numpy as np
import pytest

from source.data.production_video import OnlineVideo


def test_actual_encoder_full_decode_and_pts(tmp_path):
    video = OnlineVideo(tmp_path / "episode.mp4", timeout=20)
    try:
        for value in (0, 90, 180):
            video.submit(np.full((1080, 1920, 3), value, np.uint8))
        result = video.finish()
        assert result["frames"] == result["decoded_frames"] == 3
        assert result["pts_verified"]
        assert result['wall_profile']['phases']['encoder_pipe_write']['calls'] == 3
        assert result['wall_profile']['phases']['video_decode_verify']['seconds'] >= 0
    finally:
        if not video.closed:
            video.abort()


def test_dead_encoder_propagates_and_does_not_wait_forever(tmp_path):
    video = OnlineVideo(tmp_path / "failed.mp4", timeout=5)
    video.process.kill()
    video.process.wait(timeout=3)
    started = time.monotonic()
    try:
        with pytest.raises((RuntimeError, TimeoutError)):
            for _ in range(15):
                video.submit(np.zeros((1080, 1920, 3), np.uint8))
            video.finish()
    finally:
        video.abort()
    assert time.monotonic() - started < 6
