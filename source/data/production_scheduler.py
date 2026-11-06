"""Coverage-aware candidate planning from durable, incrementally updated stats."""

from source.data.five_task_dataset import canonical_hash
from source.data.production_templates import candidate


def enqueue_next(queue, templates, config, sampling_seed):
    quota = queue.quota_status(config['task_hours'], min_assets=config['minimum_independent_assets'],
                               count_failures=config['count_failed_episodes'], outcome_hours=config.get('task_outcome_hours'),
                               coverage_success_only=config.get('coverage_requires_success', True))
    if quota['complete']:
        return {'state': 'COMPLETE'}
    stats, covered = queue.scheduling_stats(count_failures=config['count_failed_episodes']
                                           and not config.get('coverage_requires_success', True))
    efficiency = queue.scheduling_efficiency()
    failure_limit = config.get('template_failure_limit', 3)
    duplicate_limit = config.get('template_duplicate_limit', 3)
    blocked = []
    choices = []
    task_scheduled = {}
    task_active = {}
    asset_scheduled = {}
    asset_active = {}
    for template in templates:
        row = stats.get(template['template_id'], {})
        task = template['task']
        task_scheduled[task] = task_scheduled.get(task, 0) + row.get('scheduled', 0)
        task_active[task] = task_active.get(task, 0) + row.get('active', 0)
        asset = template['asset_hash']
        asset_scheduled[asset] = asset_scheduled.get(asset, 0) + row.get('scheduled', 0)
        asset_active[asset] = asset_active.get(asset, 0) + row.get('active', 0)
    for template in templates:
        task = template['task']
        if task not in quota['needed_tasks']:
            continue
        row = stats.get(template['template_id'], {})
        reason = None
        if row.get('consecutive_failures', 0) >= failure_limit:
            reason = 'consecutive_failures'
        elif row.get('consecutive_duplicates', 0) >= duplicate_limit:
            reason = 'consecutive_duplicates'
        if reason:
            blocked.append({'template_id': template['template_id'], 'task': task, 'reason': reason})
            continue
        if row.get('active', 0):
            continue
        total = row.get('successful', 0) + row.get('failed', 0) + row.get('duplicates', 0)
        penalty = (row.get('failed', 0) + row.get('duplicates', 0)) / max(1, total)
        fraction = quota['remaining_task_hours'][task] / config['task_hours'][task]
        measured = efficiency.get(template['template_id'], {})
        complete = (measured.get('timed', 0) >= config.get('efficiency_minimum_attempts', 3)
                    and measured.get('timed') == measured.get('claimed') and measured.get('worker_seconds', 0) > 0)
        productive = measured.get('success_seconds', 0) + (measured.get('failure_seconds', 0) if config['count_failed_episodes'] else 0)
        remaining_outcomes = quota.get('remaining_outcome_hours')
        if remaining_outcomes is not None:
            productive = sum(measured.get(outcome + '_seconds', 0) for outcome in ('success', 'failure')
                             if remaining_outcomes[task][outcome] > 0)
        rate = productive / measured['worker_seconds'] if complete else None
        exploration = task_scheduled[task] % config.get('efficiency_exploration_interval', 10) == 0
        efficiency_rank = -rate if rate is not None and template['asset_hash'] in covered and not exploration else 0.
        rank = (-fraction, task_active.get(task, 0), task_scheduled.get(task, 0),
                template['asset_hash'] in covered, asset_active.get(template['asset_hash'], 0),
                efficiency_rank, asset_scheduled.get(template['asset_hash'], 0), row.get('scheduled', 0), penalty,
                template.get('validation_status') != 'CURRENT_SINGLE_CONDITION_SUCCESS', template['template_id'])
        choices.append((rank, template, row, {'schema': 'coverage-efficiency-selection-v1',
                        'measured_physical_per_worker_second': rate, 'timing_complete': complete,
                        'exploration_slot': exploration, 'timed_attempts': measured.get('timed', 0)}))
    if not choices:
        return {'state': 'WAITING_OR_BLOCKED', 'blocked_templates': blocked,
                'remaining_task_hours': quota['remaining_task_hours']}
    _, template, row, evidence = min(choices, key=lambda item: item[0])
    payload = candidate(template, sampling_seed, row.get('last_candidate_index', -1) + 1,
                        config.get('perturbation_contract'))
    from source.data.production_capture import capture_from_config

    capture = capture_from_config(config)
    if capture is not None:
        payload['rgb_capture'] = capture
    job_id = template['template_id'] + '_' + canonical_hash(payload)[:20]
    created = queue.enqueue(job_id, template['task'], template['asset_hash'], payload,
                            max_attempts=config['maximum_retries'])
    if created:
        with queue.transaction() as db:
            queue.event(db, job_id, 'SCHEDULER_SELECTED', evidence)
    return {'state': 'ENQUEUED' if created else 'ALREADY_PLANNED', 'job_id': job_id,
            'task': template['task'], 'asset_hash': template['asset_hash'], 'selection_evidence': evidence}
