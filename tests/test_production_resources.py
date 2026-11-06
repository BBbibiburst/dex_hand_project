from types import SimpleNamespace

import psutil
import pytest

from source.data.production_resources import memory_status, canary_storage_status


def test_retained_scan_retries_disappearing_directory_but_not_permission_error(tmp_path, monkeypatch):
    import os
    from source.data.production_resources import retained_file_bytes

    child = tmp_path / 'moving'
    child.mkdir()
    (child / 'data').write_bytes(b'actual bytes')
    original = os.scandir
    seen = []

    def rename_race(path):
        if str(path) == str(child):
            seen.append(path)
            if len(seen) == 1:
                raise FileNotFoundError('A directory was atomically renamed')
        return original(path)

    monkeypatch.setattr(os, 'scandir', rename_race)
    assert retained_file_bytes(tmp_path) == len(b'actual bytes')
    assert len(seen) == 2
    monkeypatch.setattr(os, 'scandir', lambda _: (_ for _ in ()).throw(PermissionError('denied')))
    with pytest.raises(PermissionError):
        retained_file_bytes(tmp_path)


def test_retained_scan_is_bounded_and_missing_root_never_looks_empty(tmp_path, monkeypatch):
    import os
    from source.data.production_resources import retained_file_bytes

    with pytest.raises(FileNotFoundError):
        retained_file_bytes(tmp_path / 'missing')
    child = tmp_path / 'moving'
    child.mkdir()
    original = os.scandir
    count = []

    def always_racing(path):
        if str(path) == str(child):
            count.append(1)
            raise FileNotFoundError('moving')
        return original(path)

    monkeypatch.setattr(os, 'scandir', always_racing)
    with pytest.raises(RuntimeError, match='remained unstable'):
        retained_file_bytes(tmp_path)
    assert len(count) == 3


def test_memory_reserve_uses_available_memory_and_is_configurable(monkeypatch):
    monkeypatch.setattr(psutil, 'virtual_memory', lambda: SimpleNamespace(
        available=512 * 1024 ** 2, total=8 * 1024 ** 3, percent=93.75))
    assert not memory_status({})['ready']
    assert memory_status({'minimum_available_memory_bytes': 256 * 1024 ** 2})['ready']
    with pytest.raises(ValueError, match='at least 256'):
        memory_status({'minimum_available_memory_bytes': 1})


def test_canary_rejects_changed_test_filesystem(tmp_path):
    result = canary_storage_status(tmp_path, tmp_path.stat().st_dev + 1)
    assert not result['ready']
    assert result['development_volume_only']
    assert any('identity changed' in e for e in result['errors'])
