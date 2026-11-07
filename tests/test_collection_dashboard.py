from collections import deque
import json
import sqlite3

import pytest

from apps.watch_collection import snapshot, projection, render


def config(tmp_path):
    return {'dataset_root': str(tmp_path / 'data'), 'queue_database': str(tmp_path / 'queue.sqlite'),
            'task_hours': {'lift': 200., 'push': 200.}, 'minimum_independent_assets': 100,
            'count_failed_episodes': True, 'coverage_requires_success': False}


def database(config):
    with sqlite3.connect(config['queue_database']) as db:
        db.executescript('CREATE TABLE jobs(task,state,lease_until,error);'
                        'CREATE TABLE episodes(task,asset_hash,success,seconds);'
                        'CREATE TABLE events(id INTEGER PRIMARY KEY,job_id,state,details,timestamp);')
        db.executemany('INSERT INTO episodes VALUES(?,?,?,?)',
                       [('lift', 'asset1', 1, 100.), ('lift', 'asset2', 0, 20.), ('push', 'asset1', 1, 60.)])
        db.execute("INSERT INTO jobs VALUES('push','RUNNING',2000,NULL)")
        db.execute('INSERT INTO events VALUES(1,?,?,?,?)',
                   ('session1', 'RUNTIME_SESSION_STARTED', json.dumps({'workers': 24, 'owner': {'pid': 1}}), 1000))


def test_snapshot_uses_credited_unique_data_without_writing(tmp_path, monkeypatch):
    cfg = config(tmp_path)
    database(cfg)
    from pathlib import Path
    path = Path(cfg['queue_database'])
    before = path.read_bytes()
    monkeypatch.setattr('apps.watch_collection.process_alive', lambda owner: True)
    data = snapshot(cfg, now=1060)
    assert data['credited_seconds'] == 180
    assert data['independent_assets'] == 2
    assert data['tasks']['lift']['successes'] == 1
    assert data['tasks']['push']['running'] == 1
    assert data['coordinator_running']
    assert path.read_bytes() == before
    estimate = projection(data, deque())
    assert estimate['rate'] == 3  # Aggregate observed throughput; not multiplied by 24.
    assert estimate['total_eta_seconds'] == pytest.approx((400*3600-180)/3)
    assert '3.000倍实时' in render(data, estimate)
    cfg['count_failed_episodes'] = False
    data = snapshot(cfg, now=1060)
    assert data['credited_seconds'] == 160
    assert data['independent_assets'] == 1


def test_recent_rate_and_zero_progress_do_not_make_up_eta(tmp_path, monkeypatch):
    cfg = config(tmp_path)
    database(cfg)
    monkeypatch.setattr('apps.watch_collection.process_alive', lambda owner: True)
    history = deque()
    data = snapshot(cfg, now=1060)
    projection(data, history)
    data['timestamp'] += 60
    estimate = projection(data, history)
    assert estimate['rate'] == 0
    assert estimate['total_eta_seconds'] is None
    data['coordinator_running'] = False
    assert projection(data, deque())['total_eta_seconds'] is None


def test_empty_database_is_not_created(tmp_path):
    cfg = config(tmp_path)
    assert not snapshot(cfg)['available']
    assert not (tmp_path / 'queue.sqlite').exists()
