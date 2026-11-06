import json
from pathlib import Path

import pytest

from test_production_artifacts import synthetic_case as format_fixture
from source.data.production_artifacts import package_episode, register_artifact
from source.data.production_periodic_audit import audit_after_attempt
from source.data.production_queue import ProductionQueue


@pytest.fixture
def registered(tmp_path):
    case = format_fixture.__wrapped__(tmp_path)
    root = tmp_path / 'dataset'
    queue = ProductionQueue(tmp_path / 'queue.sqlite')
    queue.enqueue('job', 'lift', 'a' * 64, {})
    job = queue.claim()
    package_episode(case, root, 'job', 'a' * 64)
    register_artifact(queue, job, root, allow_development=True)
    return root, queue


def test_periodic_file_audit_survives_restart_and_detects_sql_mismatch(registered):
    root, queue = registered
    assert audit_after_attempt(queue, root, interval=2)['state'] == 'NOT_DUE'
    queue = ProductionQueue(queue.path)
    passed = audit_after_attempt(queue, root, interval=2)
    assert passed['state'] == 'PASSED'
    assert json.loads(Path(passed['record_path']).read_text())['passed']
    with queue.transaction() as db:
        db.execute('UPDATE episodes SET seconds=999')
    failed = audit_after_attempt(queue, root, interval=2, force=True, count_attempt=False)
    assert failed['state'] == 'FAILED'
    assert 'duration' in failed['error']
    # An unresolved failure bypasses the usual interval after reopening.
    queue = ProductionQueue(queue.path)
    assert audit_after_attempt(queue, root, interval=100)['state'] == 'FAILED'


def test_periodic_audit_rejects_changed_episode_bytes(registered):
    root, queue = registered
    with (root / 'episodes/job/modalities.npz').open('ab') as stream:
        stream.write(b'tamper')
    result = audit_after_attempt(queue, root, interval=1)
    assert result['state'] == 'FAILED'
    assert 'content changed' in result['error']


def test_cycle_has_frozen_ceiling_and_revisits_old_records(tmp_path, monkeypatch):
    import source.data.production_periodic_audit as module

    root = tmp_path / 'dataset'
    root.mkdir()
    queue = ProductionQueue(tmp_path / 'queue.sqlite')

    def add(index):
        digest = f'{index:064x}'
        queue.enqueue(str(index), 'lift', digest, {})
        job = queue.claim()
        queue.validating(job['job_id'], job['lease_token'])
        queue.register(job['job_id'], job['lease_token'], lineage_hash=digest, content_hash=digest,
                       relative_path=f'episodes/{index}', seconds=1., success=True, verification={'passed': True})

    monkeypatch.setattr(module, 'verify_inventory_row', lambda *_: {'frames': 1})
    add(1)
    add(2)
    assert audit_after_attempt(queue, root, interval=1)['job_id'] == '1'
    add(3)
    assert audit_after_attempt(queue, root, interval=1)['job_id'] == '2'
    # New rows cannot keep an old cycle open forever; the next cycle revisits old files.
    assert [audit_after_attempt(queue, root, interval=1)['job_id'] for _ in range(3)] == ['1', '2', '3']


def test_startup_integrity_failure_stops_before_worker_launch(tmp_path, monkeypatch):
    from apps.five_task_production import load_config
    import source.data.production_coordinator as coordinator
    import source.data.production_periodic_audit as module

    config = load_config(Path('configs/five_task_production.json'))
    monkeypatch.setattr(coordinator, 'execution_fingerprint', lambda _: {})
    monkeypatch.setattr(coordinator, 'load_templates', lambda *_: [
        {'template_id': 'fixture', 'task': 'lift', 'asset_hash': 'a' * 64,
         'base_seed': 0, 'perturbation_fields': ['dx_m']}])
    monkeypatch.setattr(module, 'audit_after_attempt', lambda *a, **kw: {
        'state': 'FAILED', 'error': 'artificial stored file mismatch'})
    monkeypatch.setattr(coordinator.subprocess, 'Popen', lambda *a, **kw: pytest.fail('Must not start worker'))
    result = coordinator._coordinate(config, tmp_path / 'dataset', 1, 42, canary_mode=True, wall_budget=10)
    assert result['termination'] == 'INTEGRITY_AUDIT_FAILED'
    assert result['attempted_jobs'] == 0
    assert result['integrity_failures'][0]['error'] == 'artificial stored file mismatch'
