from pathlib import Path

import pytest

from apps.five_task_production import load_config, inspect
from source.data.production_queue import ProductionQueue
from source.data.production_statistics import production_statistics


def test_asset_statistics_include_duplicate_cost_without_double_credit(tmp_path):
    queue = ProductionQueue(tmp_path / 'queue.sqlite')
    config = {**load_config(Path('configs/five_task_production.json')), 'queue_database': str(queue.path)}
    for name in ('first', 'duplicate'):
        queue.enqueue(name, 'lift', 'a' * 64, {})
        job = queue.claim()
        queue.validating(name, job['lease_token'])
        state = queue.register(name, job['lease_token'], lineage_hash='b' * 64, content_hash='c' * 64,
                               relative_path='episodes/' + name, seconds=10., success=True, verification={'passed': True})
        queue.record_attempt_metrics(name, 1, 20., state)
    result = production_statistics(config)
    asset = result['assets'][0]
    assert asset['worker_wall_seconds'] == 40
    assert asset['success_seconds'] == 10
    assert asset['successful_episodes'] == asset['unique_registered_contents'] == 1
    assert asset['failed_episodes'] == 0
    assert asset['duplicate_worker_seconds'] == 20
    assert asset['timed_eligible_physical_seconds_per_worker_second'] == .25
    assert result['wall_clock_eta_seconds'] is None
    assert result['tasks']['lift']['remaining_worker_seconds_at_observed_mix'] == pytest.approx((200 * 3600 - 10) * 4)
    assert result['tasks']['push']['remaining_worker_seconds_at_observed_mix'] is None


def test_lost_timing_does_not_create_a_fast_eta_and_status_is_read_only(tmp_path):
    queue = ProductionQueue(tmp_path / 'queue.sqlite')
    queue.enqueue('job', 'lift', 'a' * 64, {}, max_attempts=2)
    job = queue.claim()
    queue.fail('job', job['lease_token'], 'historical crash')
    job = queue.claim()
    queue.validating('job', job['lease_token'])
    queue.register('job', job['lease_token'], lineage_hash='b' * 64, content_hash='c' * 64,
                   relative_path='episodes/job', seconds=10., success=True, verification={'passed': True})
    queue.record_attempt_metrics('job', 2, 1., 'EXPORTED')
    config = {**load_config(Path('configs/five_task_production.json')), 'queue_database': str(queue.path)}
    with queue.connect() as db:
        before = list(db.iterdump())
    result = inspect(config)['runtime_statistics']
    with queue.connect() as db:
        assert list(db.iterdump()) == before
    assert result['assets'][0]['unmeasured_attempts'] == 1
    assert result['tasks']['lift']['remaining_worker_seconds_at_observed_mix'] is None


def test_attempt_metric_identity_is_immutable(tmp_path):
    queue = ProductionQueue(tmp_path / 'queue.sqlite')
    queue.enqueue('job', 'lift', 'a' * 64, {})
    with pytest.raises(ValueError, match='actually claimed'):
        queue.record_attempt_metrics('job', 1, 1., 'EXPORTED')
    queue.claim()
    queue.record_attempt_metrics('job', 1, 1., 'PENDING')
    queue.record_attempt_metrics('job', 1, 1., 'PENDING')
    with pytest.raises(ValueError, match='immutable'):
        queue.record_attempt_metrics('job', 1, 2., 'PENDING')
    with pytest.raises(ValueError):
        queue.record_attempt_metrics('job', 1, float('nan'), 'PENDING')
