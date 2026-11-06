import json
import subprocess
import sys
import sqlite3

import pytest

from test_production_artifacts import synthetic_case as format_fixture
from source.data.five_task_dataset import canonical_hash
from source.data.production_artifacts import package_episode
from source.data.production_index import open_inventory, active_database, pointer_path
from source.data.production_scheduler import enqueue_next
from source.data.production_templates import candidate


@pytest.fixture
def recoverable(tmp_path):
    root = tmp_path / 'dataset'
    template = {'template_id': 'fixture', 'task': 'lift', 'asset_hash': 'a' * 64,
                'base_seed': 1, 'perturbation_fields': ['dx_m']}
    case = format_fixture.__wrapped__(tmp_path)
    accepted = candidate(template, 42, 0)
    job_id = 'fixture_' + canonical_hash(accepted)[:20]
    package_episode(case, root, job_id, 'a' * 64, job_payload=accepted)
    work = root / 'work' / job_id
    work.mkdir(parents=True)
    (work / 'job_001.json').write_text(json.dumps(accepted))
    pending = candidate(template, 42, 1)
    pending_id = 'fixture_' + canonical_hash(pending)[:20]
    work = root / 'work' / pending_id
    work.mkdir(parents=True)
    (work / 'job_001.json').write_text(json.dumps(pending))
    database = tmp_path / 'control' / 'queue.sqlite'
    database.parent.mkdir()
    database.write_bytes(b'original corrupt SQLite evidence')
    return root, database, template, pending_id


def test_recovery_preserves_corrupt_database_and_exhausts_unknown_retries(recoverable):
    root, database, template, pending_id = recoverable
    queue = open_inventory(root, database)
    assert database.read_bytes() == b'original corrupt SQLite evidence'
    assert queue.path != database
    assert active_database(root, database) == queue.path
    with queue.connect() as db:
        row = db.execute('SELECT state,attempts,max_attempts FROM jobs WHERE job_id=?', (pending_id,)).fetchone()
        assert tuple(row) == ('QUARANTINED', 3, 3)
        assert db.execute("SELECT count(*) FROM events WHERE state='RUNNING'").fetchone()[0] == 0
    config = {'task_hours': dict.fromkeys(('lift','pick_place','stack','push','nut_assembly'), 200.),
              'minimum_independent_assets': 100, 'count_failed_episodes': False, 'maximum_retries': 3}
    decision = enqueue_next(queue, [template], config, 42)
    assert decision['state'] == 'ENQUEUED'
    assert queue.claim()['payload']['candidate_index'] == 2
    assert open_inventory(root, database).path == queue.path


def test_recovery_refuses_live_worker_or_disabled_policy(recoverable, monkeypatch):
    import source.data.production_index as module

    root, database, _, _ = recoverable
    with pytest.raises(RuntimeError, match='disabled'):
        open_inventory(root, database, auto_rebuild=False)
    monkeypatch.setattr(module, 'live_dataset_processes', lambda _: [12345])
    with pytest.raises(RuntimeError, match='alive'):
        open_inventory(root, database)
    assert not pointer_path(database).exists()


def test_recovery_does_not_activate_when_payload_history_is_missing(recoverable):
    root, database, _, pending = recoverable
    (root / 'work' / pending / 'job_001.json').rename(root / 'work' / pending / 'preserved_unknown.json')
    with pytest.raises(ValueError, match='no recoverable payload'):
        open_inventory(root, database)
    assert not pointer_path(database).exists()


def test_active_pointer_cannot_select_unrelated_database(tmp_path):
    root = tmp_path / 'dataset'
    database = tmp_path / 'queue.sqlite'
    other = tmp_path / 'other.sqlite'
    other.touch()
    pointer_path(database).write_text(json.dumps({'schema': 'production-active-index-v1',
        'dataset_root': str(root.resolve()), 'original_database': str(database.resolve()),
        'active_database': str(other.resolve())}))
    with pytest.raises(ValueError, match='identity'):
        active_database(root, database)


def test_live_worker_detection_uses_actual_process_and_dataset_path(tmp_path):
    from source.data.production_index import live_dataset_processes

    root = tmp_path / 'dataset'
    process = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(10)',
                                'apps/production_worker.py', '--output', str(root / 'work/fixture')])
    try:
        assert process.pid in live_dataset_processes(root)
        assert process.pid not in live_dataset_processes(tmp_path / 'unrelated')
    finally:
        process.terminate()
        process.wait(timeout=5)


def test_recovery_before_first_export_preserves_exhausted_candidate(tmp_path):
    root = tmp_path / 'dataset'
    payload = candidate({'template_id': 'fixture', 'task': 'lift', 'asset_hash': 'a' * 64,
                         'base_seed': 1, 'perturbation_fields': ['dx_m']}, 42, 0)
    job_id = 'fixture_' + canonical_hash(payload)[:20]
    work = root / 'work' / job_id
    work.mkdir(parents=True)
    (work / 'job_001.json').write_text(json.dumps(payload))
    database = tmp_path / 'broken.sqlite'
    database.write_bytes(b'not a database')
    queue = open_inventory(root, database)
    assert queue.status()['episodes'] == 0
    assert queue.status()['jobs'] == {'QUARANTINED': 1}
    assert queue.claim() is None
    stats, _ = queue.scheduling_stats()
    assert stats['fixture']['last_candidate_index'] == 0


def test_unrelated_valid_sqlite_is_never_rebuilt(tmp_path):
    database = tmp_path / 'unrelated.sqlite'
    with sqlite3.connect(database) as db:
        db.execute('CREATE TABLE unrelated(value TEXT)')
    before = database.read_bytes()
    with pytest.raises(ValueError, match='not this production queue'):
        open_inventory(tmp_path / 'dataset', database)
    assert database.read_bytes() == before
    assert not pointer_path(database).exists()
