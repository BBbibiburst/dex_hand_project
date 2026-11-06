import pytest

from source.data.production_queue import ProductionQueue, TASKS
from source.data.production_quotas import validate_outcome_hours


def record(queue, key, task, success, seconds):
    queue.enqueue(key, task, ('a' if success else 'b') * 64, {})
    job = queue.claim([task])
    queue.validating(job['job_id'], job['lease_token'])
    from source.data.five_task_dataset import canonical_hash
    digest = canonical_hash(key)
    queue.register(key, job['lease_token'], lineage_hash=digest, content_hash=digest,
                   relative_path='artificial/' + key, seconds=seconds, success=success,
                   verification={'passed': True, 'artificial_test': True})


def test_excess_success_cannot_replace_missing_failure_quota(tmp_path):
    queue = ProductionQueue(tmp_path / 'queue.sqlite')
    hours = dict.fromkeys(TASKS, 200.)
    targets = {task: {'success': 180., 'failure': 20.} for task in TASKS}
    for task in TASKS:
        record(queue, task + '_success', task, True, 250 * 3600)
    result = queue.quota_status(hours, min_assets=1, count_failures=True, outcome_hours=targets,
                               coverage_success_only=True)
    assert result['raw_eligible_hours'] == 1250
    assert result['eligible_hours'] == 900
    assert not result['complete']
    assert all(row['failure'] == 20 for row in result['remaining_outcome_hours'].values())
    assert set(result['needed_tasks']) == set(TASKS)
    for task in TASKS:
        record(queue, task + '_failure', task, False, 20 * 3600)
    result = queue.quota_status(hours, min_assets=1, count_failures=True, outcome_hours=targets)
    assert result['complete']
    assert result['eligible_hours'] == 1000


def test_failed_only_assets_do_not_satisfy_success_required_coverage(tmp_path):
    queue = ProductionQueue(tmp_path / 'queue.sqlite')
    for task in TASKS:
        record(queue, task, task, False, 200 * 3600)
    result = queue.quota_status(dict.fromkeys(TASKS, 200.), min_assets=1,
                               count_failures=True, coverage_success_only=True)
    assert result['remaining_task_hours'] == dict.fromkeys(TASKS, 0.)
    assert result['remaining_assets'] == 1
    assert not result['complete']


@pytest.mark.parametrize('failure,count_failed', [(20, False), (-1, True), (float('nan'), True)])
def test_invalid_outcome_contract_rejected(failure, count_failed):
    hours = dict.fromkeys(TASKS, 200.)
    targets = {task: {'success': 180., 'failure': failure} for task in TASKS}
    with pytest.raises(ValueError):
        validate_outcome_hours(hours, targets, count_failed)


def test_stalled_watchdog_persists_and_real_progress_resets_it(tmp_path):
    path = tmp_path / 'queue.sqlite'
    queue = ProductionQueue(path)
    assert not queue.observe_progress(0., 1, limit=2)['stalled']
    assert not queue.observe_progress(0., 1, attempt=True, limit=2)['stalled']
    queue = ProductionQueue(path)
    assert queue.observe_progress(0., 1, attempt=True, limit=2)['stalled']
    assert queue.observe_progress(0., 1, limit=2)['stalled']
    progress = queue.observe_progress(0., 0, attempt=True, limit=2)
    assert not progress['stalled']
    assert progress['attempts_without_progress'] == 0
