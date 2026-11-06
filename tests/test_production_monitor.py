import json

from source.data.production_monitor import write_worker_status


def test_worker_snapshots_are_separate_and_replace_atomically(tmp_path):
    write_worker_status(tmp_path, 0, {'state': 'ACTIVE', 'job_id': 'one'})
    write_worker_status(tmp_path, 1, {'state': 'ACTIVE', 'job_id': 'two'})
    write_worker_status(tmp_path, 0, {'state': 'ATTEMPT_ENDED', 'job_id': 'one'})
    first = json.loads((tmp_path / 'runtime/worker_0.json').read_text())
    second = json.loads((tmp_path / 'runtime/worker_1.json').read_text())
    assert first['state'] == 'ATTEMPT_ENDED'
    assert second['job_id'] == 'two'
    assert first['updated_unix_seconds'] > 0
    assert not list((tmp_path / 'runtime').glob('*.tmp'))
