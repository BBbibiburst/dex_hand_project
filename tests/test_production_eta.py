from pathlib import Path

import pytest

from apps.five_task_production import load_config
from source.data.production_eta import begin_session, finish_session
from source.data.production_queue import ProductionQueue, TASKS
from source.data.production_statistics import production_statistics


def measured_queue(tmp_path, monkeypatch):
    queue = ProductionQueue(tmp_path / 'queue.sqlite')
    clock = [100.]
    monkeypatch.setattr('source.data.production_eta.time.monotonic', lambda: clock[0])
    session = begin_session(queue, 2)
    for task_index, task in enumerate(TASKS):
        for i in range(3):
            name = f'{task}_{i}'
            queue.enqueue(name, task, f'{task_index + 1:064x}', {})
            job = queue.claim()
            queue.validating(name, job['lease_token'])
            queue.register(name, job['lease_token'], lineage_hash=f'{task_index * 3 + i + 1:064x}',
                           content_hash=f'{task_index * 3 + i + 1:064x}', relative_path=f'episodes/{name}',
                           seconds=10., success=True, verification={'passed': True})
            queue.record_attempt_metrics(name, 1, 20., 'EXPORTED')
    clock[0] = 300.
    finish_session(queue, session)
    config = {**load_config(Path('configs/five_task_production.json')),
              'queue_database': str(queue.path), 'minimum_independent_assets': 5}
    return queue, config, session


def test_projection_uses_observed_concurrency_not_number_of_workers(tmp_path, monkeypatch):
    _, config, _ = measured_queue(tmp_path, monkeypatch)
    result = production_statistics(config)
    estimate = result['remaining_time_projection']
    assert estimate['calibration']['measured_worker_seconds_per_wall_second'] == 1.5
    expected = sum(task['remaining_worker_seconds_at_observed_mix'] for task in result['tasks'].values()) / 1.5
    assert result['wall_clock_eta_seconds'] == pytest.approx(expected)
    assert estimate['unavailable_reasons'] == []


def test_coverage_does_not_get_invented_eta(tmp_path, monkeypatch):
    _, config, _ = measured_queue(tmp_path, monkeypatch)
    result = production_statistics({**config, 'minimum_independent_assets': 100})
    assert result['wall_clock_eta_seconds'] is None
    assert result['remaining_time_projection']['hours_only_wall_seconds_at_observed_mix'] > 0
    assert 'ASSET_COVERAGE_TIME_UNKNOWN' in result['remaining_time_projection']['unavailable_reasons']


def test_crashed_session_or_changed_parallelism_invalidates_calibration(tmp_path, monkeypatch):
    queue, config, session = measured_queue(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match='immutable'):
        finish_session(queue, session)
    changed = production_statistics({**config, 'workers': 1})
    assert changed['wall_clock_eta_seconds'] is None
    assert 'WORKER_CONFIGURATION_CHANGED' in changed['remaining_time_projection']['unavailable_reasons']
    begin_session(queue, 2)  # Simulated missing terminal record after process loss.
    monkeypatch.setattr('source.data.production_eta.process_identity', lambda _: None)
    result = production_statistics(config)
    assert result['wall_clock_eta_seconds'] is None
    assert 'UNFINISHED_OR_AMBIGUOUS_SESSION' in result['remaining_time_projection']['unavailable_reasons']


def test_live_owner_and_lease_separate_in_progress_from_lost_timing(tmp_path, monkeypatch):
    queue, config, _ = measured_queue(tmp_path, monkeypatch)
    begin_session(queue, 2)
    clock = [400.]
    monkeypatch.setattr('source.data.production_eta.time.monotonic', lambda: clock[0])
    queue.enqueue('running', 'lift', f'{1:064x}', {})
    queue.claim()
    result = production_statistics(config)
    lift = result['tasks']['lift']
    assert lift['unmeasured_attempts'] == 1
    assert lift['in_progress_untimed_attempts'] == 1
    assert lift['unmeasured_completed_attempts'] == 0
    assert result['wall_clock_eta_seconds'] is not None
    assert result['remaining_time_projection']['calibration']['live_sessions'] == 1
    with queue.transaction() as db:
        db.execute("UPDATE jobs SET lease_until=0 WHERE job_id='running'")
    expired = production_statistics(config)
    assert expired['tasks']['lift']['unmeasured_completed_attempts'] == 1
    assert expired['wall_clock_eta_seconds'] is None


def test_missing_success_outcome_keeps_success_only_eta_unknown(tmp_path, monkeypatch):
    queue, config, _ = measured_queue(tmp_path, monkeypatch)
    with queue.transaction() as db:
        db.execute("UPDATE episodes SET success=0 WHERE task='lift'")
    result = production_statistics(config)
    assert result['wall_clock_eta_seconds'] is None
    assert 'UNMEASURED_TASK_OR_REQUIRED_OUTCOME:lift' in result['remaining_time_projection']['unavailable_reasons']
