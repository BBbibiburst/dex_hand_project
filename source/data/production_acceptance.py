"""Recompute dataset acceptance from verified files, independently of SQL totals."""

import math

from source.data.production_queue import TASKS
from source.data.production_quotas import validate_outcome_hours


def acceptance_from_audit(config, audit):
    targets = validate_outcome_hours(config['task_hours'], config.get('task_outcome_hours'),
                                     config['count_failed_episodes'])
    by_task = {task: {'successful_hours': 0., 'failed_hours': 0., 'episodes': 0} for task in TASKS}
    assets = set()
    contents = set()
    lineages = set()
    duplicates = []
    unique = []
    for row in audit['episodes']:
        if not row.get('passed'):
            continue
        if row['task'] not in by_task or not math.isfinite(row['duration_seconds']) or row['duration_seconds'] <= 0:
            raise ValueError('Verified file audit has invalid task or physical duration')
        if row['content_hash'] in contents or row['lineage_hash'] in lineages:
            duplicates.append(row['job_id'])
            continue
        contents.add(row['content_hash'])
        lineages.add(row['lineage_hash'])
        unique.append(row)
        counts = by_task[row['task']]
        counts['episodes'] += 1
        counts['successful_hours' if row['success'] else 'failed_hours'] += row['duration_seconds'] / 3600
        if row['success'] or not config.get('coverage_requires_success', True):
            assets.add(row['asset_hash'])
    for task, counts in by_task.items():
        if targets is None:
            credit = counts['successful_hours'] + (counts['failed_hours'] if config['count_failed_episodes'] else 0.)
            remaining = max(0., config['task_hours'][task] - credit)
            missing_outcomes = None
        else:
            missing_outcomes = {'success': max(0., targets[task]['success'] - counts['successful_hours']),
                                'failure': max(0., targets[task]['failure'] - counts['failed_hours'])}
            remaining = sum(missing_outcomes.values())
            credit = min(targets[task]['success'], counts['successful_hours']) + min(targets[task]['failure'], counts['failed_hours'])
        counts.update(credited_hours=credit, remaining_hours=remaining, remaining_outcome_hours=missing_outcomes)
    target_met = all(row['remaining_hours'] == 0 for row in by_task.values()) and len(assets) >= config['minimum_independent_assets']
    formal = bool(unique) and all(row.get('production_admission') == 'production_verified' for row in unique)
    integrity = bool(audit['passed']) and not duplicates
    return {'evidence_source': 'independently_verified_episode_files', 'file_integrity_passed': integrity,
            'duplicate_jobs_not_credited': duplicates, 'tasks': by_task, 'verified_unique_episodes': len(unique),
            'unique_seconds': sum(row['duration_seconds'] for row in unique),
            'credited_hours': sum(row['credited_hours'] for row in by_task.values()),
            'independent_assets': len(assets), 'remaining_assets': max(0, config['minimum_independent_assets'] - len(assets)),
            'data_target_met': target_met, 'all_credited_files_formally_admitted': formal,
            'formal_dataset_accepted': target_met and formal and integrity,
            'software_delivery_certified': False,
            'software_delivery_note': 'Data counters do not certify unverified deployment/endurance requirements'}
