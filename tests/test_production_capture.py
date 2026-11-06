import json
from pathlib import Path

import numpy as np
import pytest

from apps.five_task_production import load_config
from source.data.production_capture import capture_from_config, recorded_rgb_contract
from source.data.production_video import OnlineVideo, verify_video


def speed():
    return capture_from_config(load_config(Path('configs/five_task_speed.json')))


def test_speed_profile_is_explicit_and_old_config_stays_hd():
    config = load_config(Path('configs/five_task_speed.json'))
    assert config['workers'] == 'auto'
    assert speed()['fps'] == 5
    assert capture_from_config(load_config(Path('configs/five_task_collection.json'))) is None


def test_sparse_rgb_preserves_each_control_sample_and_explicit_indices(tmp_path):
    capture = speed()
    video = OnlineVideo(tmp_path / 'speed.mp4', capture=capture, timeout=20)
    try:
        for step in range(10):
            frame = np.full((360, 640, 3), step*20, dtype=np.uint8) if video.should_capture else None
            video.record_control(frame)
        result = video.finish()
        assert result['frames'] == 3
        assert result['control_frames'] == 10
        assert video.control_indices == [0, 4, 8]
        assert verify_video(video.path, 3, capture=capture)['pts_verified']
        with pytest.raises(ValueError):
            verify_video(video.path, 3)  # A 5Hz low-res video cannot pass as legacy HD20.
    finally:
        if not video.closed:
            video.abort()
    arrays = {'timestamp': np.arange(10)*.05}
    contract = {'rgb_capture': capture, 'rgb_control_indices': [0, 4, 8]}
    assert recorded_rgb_contract(arrays, contract)[0] == 3
    contract['rgb_control_indices'] = [0, 3, 8]
    with pytest.raises(ValueError, match='indices'):
        recorded_rgb_contract(arrays, contract)


def test_speed_config_supports_server_dataset_path_without_windows_volume(tmp_path, monkeypatch):
    from source.data.production_preflight import check_dataset_volume

    monkeypatch.setenv('DEX_DATASET_ROOT', str(tmp_path/'server_data'))
    config = load_config(Path('configs/five_task_speed.json'))
    assert config['dataset_root'] == str(tmp_path/'server_data')
    result = check_dataset_volume(config)
    assert 'Windows F' not in ' '.join(result['errors'])
    assert not (tmp_path/'server_data').exists()  # Preflight must not create the dataset.


def test_invalid_fps_cannot_silently_change_control_timing(tmp_path):
    config = json.loads(Path('configs/five_task_speed.json').read_text())
    config['rgb_fps'] = 3
    path = tmp_path/'bad.json'
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match='divide'):
        load_config(path)
