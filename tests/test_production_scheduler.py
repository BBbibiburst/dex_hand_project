from source.data.production_queue import ProductionQueue, TASKS
from source.data.production_scheduler import enqueue_next
from source.data.production_templates import candidate
from source.data.five_task_dataset import canonical_hash


def template(name, task='lift', asset='a'):
    return {'template_id': name, 'task': task, 'asset_hash': asset * 64,
            'base_seed': 0, 'perturbation_fields': ['dx_m', 'dy_m']}


def config():
    return {'task_hours': dict.fromkeys(TASKS, 200.), 'minimum_independent_assets': 100,
            'count_failed_episodes': False, 'maximum_retries': 1,
            'template_failure_limit': 1, 'template_duplicate_limit': 2}


def test_new_asset_preferred_over_second_active_template_for_same_asset(tmp_path):
    queue = ProductionQueue(tmp_path / 'queue.sqlite')
    rows = [template('a_one'), template('a_two'), template('b_one', asset='b')]
    first = enqueue_next(queue, rows, config(), 42)
    second = enqueue_next(queue, rows, config(), 42)
    assert first['asset_hash'] == 'a' * 64
    assert second['asset_hash'] == 'b' * 64


def test_failed_template_isolated_without_redistributing_its_quota(tmp_path):
    queue = ProductionQueue(tmp_path / 'queue.sqlite')
    bad = template('bad')
    enqueue_next(queue, [bad], config(), 42)
    job = queue.claim()
    queue.fail(job['job_id'], job['lease_token'], 'unreachable', retryable=False)
    blocked = enqueue_next(queue, [bad], config(), 42)
    assert blocked['state'] == 'WAITING_OR_BLOCKED'
    assert blocked['remaining_task_hours']['lift'] == 200
    assert blocked['blocked_templates'][0]['reason'] == 'consecutive_failures'
    healthy = template('good', task='push', asset='b')
    assert enqueue_next(queue, [bad, healthy], config(), 42)['task'] == 'push'


def test_duplicate_limit_and_candidate_indices_persist_across_restart(tmp_path):
    queue = ProductionQueue(tmp_path / 'queue.sqlite')
    row = template('one')
    for index in range(3):
        enqueue_next(queue, [row], config(), 42)
        job = queue.claim()
        assert job['payload']['candidate_index'] == index
        queue.validating(job['job_id'], job['lease_token'])
        queue.register(job['job_id'], job['lease_token'], lineage_hash='c' * 64,
                       content_hash='d' * 64, relative_path=f'episodes/{index}', seconds=1.,
                       success=True, verification={'passed': True})
        queue = ProductionQueue(queue.path)
    health, _ = queue.scheduling_stats()
    assert health['one']['scheduled'] == 3
    assert health['one']['successful'] == 1
    assert health['one']['duplicates'] == 2
    assert health['one']['active'] == 0
    assert enqueue_next(queue, [row], config(), 42)['state'] == 'WAITING_OR_BLOCKED'


def seed_measured_history(queue, row, seconds, success=True):
    for index in range(3):
        payload = candidate(row, 42, index)
        job_id = row['template_id'] + '_' + canonical_hash(payload)[:20]
        queue.enqueue(job_id, row['task'], row['asset_hash'], payload)
        job = queue.claim()
        queue.validating(job_id, job['lease_token'])
        queue.register(job_id, job['lease_token'], lineage_hash=canonical_hash(job_id),
            content_hash=canonical_hash(job_id), relative_path='episodes/' + job_id,
            seconds=1., success=success, verification={'passed': True})
        queue.record_attempt_metrics(job_id, 1, seconds, 'EXPORTED')


def test_measured_efficiency_breaks_ties_after_coverage_and_survives_restart(tmp_path):
    queue = ProductionQueue(tmp_path / 'queue.sqlite')
    slow, fast = template('a_slow'), template('z_fast', asset='b')
    seed_measured_history(queue, slow, 10.)
    seed_measured_history(queue, fast, 1.)
    queue = ProductionQueue(queue.path)
    decision = enqueue_next(queue, [slow, fast], config(), 42)
    assert decision['asset_hash'] == fast['asset_hash']
    assert decision['selection_evidence']['measured_physical_per_worker_second'] == 1.
    assert decision['selection_evidence']['timing_complete']


def test_exploration_and_new_asset_coverage_are_not_overridden_by_efficiency(tmp_path):
    queue = ProductionQueue(tmp_path / 'queue.sqlite')
    slow, fast = template('a_slow'), template('z_fast', asset='b')
    seed_measured_history(queue, slow, 10.)
    seed_measured_history(queue, fast, 1.)
    cfg = {**config(), 'efficiency_exploration_interval': 1}
    decision = enqueue_next(queue, [slow, fast], cfg, 42)
    assert decision['asset_hash'] == slow['asset_hash']
    assert decision['selection_evidence']['exploration_slot']
    # A brand-new asset still comes before either already-covered template.
    fresh = template('fresh', asset='c')
    assert enqueue_next(queue, [slow, fast, fresh], config(), 42)['asset_hash'] == fresh['asset_hash']


def test_missing_or_rebuilt_history_disables_efficiency_scores(tmp_path):
    queue = ProductionQueue(tmp_path / 'queue.sqlite')
    fast = template('one')
    seed_measured_history(queue, fast, 1.)
    with queue.transaction() as db:
        db.execute("INSERT INTO queue_meta VALUES('file_rebuild_history','{}')")
    decision = enqueue_next(queue, [fast], config(), 42)
    assert decision['selection_evidence']['measured_physical_per_worker_second'] is None


def test_missing_timing_is_not_treated_as_zero_cost_and_cache_rebuild_is_idempotent(tmp_path):
    queue = ProductionQueue(tmp_path / 'queue.sqlite')
    row = template('one')
    seed_measured_history(queue, row, 2.)
    expected = queue.scheduling_efficiency()
    with queue.transaction() as db:
        db.execute("DELETE FROM queue_meta WHERE key='template_efficiency_v1'")
    queue = ProductionQueue(queue.path)
    assert queue.scheduling_efficiency() == expected
    payload = candidate(row, 42, 3)
    job_id = row['template_id'] + '_' + canonical_hash(payload)[:20]
    queue.enqueue(job_id, row['task'], row['asset_hash'], payload)
    job = queue.claim()
    queue.fail(job_id, job['lease_token'], 'artificial unmeasured crash', retryable=False)
    decision = enqueue_next(queue, [row], {**config(), 'template_failure_limit': 10}, 42)
    assert decision['selection_evidence']['measured_physical_per_worker_second'] is None
    assert not decision['selection_evidence']['timing_complete']


def test_fulfilled_failure_quota_cannot_make_failed_template_look_productive(tmp_path):
    queue = ProductionQueue(tmp_path / 'queue.sqlite')
    good, bad = template('z_success'), template('a_fast_failure', asset='b')
    seed_measured_history(queue, good, 10.)
    seed_measured_history(queue, bad, 1., success=False)
    cfg = {**config(), 'count_failed_episodes': True, 'coverage_requires_success': False,
           'template_failure_limit': 10,
           'task_outcome_hours': {task: {'success': 200., 'failure': 0.} for task in TASKS}}
    assert enqueue_next(queue, [good, bad], cfg, 42)['asset_hash'] == good['asset_hash']
