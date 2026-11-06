from pathlib import Path

from source.data.production_preflight import check_volume


def test_formal_target_cannot_escape_f_volume():
    result = check_volume(Path('/tmp/not_a_production_volume'))
    assert not result['ready']


def test_root_fallback_rejected_even_when_directory_exists(monkeypatch):
    import source.data.production_preflight as module

    monkeypatch.setattr(module, 'mount_info', lambda _: {
        'mountpoint': '/', 'filesystem': 'ext4', 'source': '/dev/root', 'options': 'rw'})
    result = check_volume(Path('/mnt/f/dataset'))
    assert not result['ready']
    assert any('root filesystem' in error or 'does not exist' in error for error in result['errors'])
