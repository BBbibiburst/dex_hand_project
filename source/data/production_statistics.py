"""Read-only task/asset efficiency with explicit gaps in timing evidence."""

from pathlib import Path
from contextlib import contextmanager
import sqlite3

from source.data.production_queue import TASKS, ProductionQueue
from source.data.production_quotas import validate_outcome_hours
from source.data.production_eta import session_calibration, remaining_time_projection, live_untimed_attempts
from source.data.production_parallel import worker_plan


class ReadOnlyQueue(ProductionQueue):
    def __init__(self, path):
        self.path = Path(path)

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path.resolve().as_uri() + '?mode=ro', uri=True)
        db.row_factory = sqlite3.Row
        try:
            yield db
        finally:
            db.close()


def production_statistics(config):
    from source.data.production_index import active_database

    database = active_database(config['dataset_root'], config['queue_database'])
    if not database.exists():
        return {'available': False, 'reason': 'No inventory', 'assets': [], 'wall_clock_eta_seconds': None}
    db = sqlite3.connect(database.resolve().as_uri() + '?mode=ro', uri=True)
    db.row_factory = sqlite3.Row
    try:
        db.execute('BEGIN')
        recovered_history = db.execute("SELECT 1 FROM queue_meta WHERE key='file_rebuild_history'").fetchone() is not None
        rows = { (r['task'], r['asset_hash']): dict(r) for r in db.execute(
            'SELECT task,asset_hash,count(*) AS jobs,sum(attempts) AS claimed_attempts '
            'FROM jobs GROUP BY task,asset_hash') }
        has_metrics = db.execute("SELECT 1 FROM sqlite_master WHERE name='attempt_metrics'").fetchone()
        if has_metrics:
            has_duration = 'candidate_seconds' in {r['name'] for r in db.execute('PRAGMA table_info(attempt_metrics)')}
            duration_sql = ('coalesce(sum(m.candidate_seconds),0) AS known_pre_dedup_candidate_seconds,'
                'sum(CASE WHEN m.candidate_seconds IS NULL THEN 1 ELSE 0 END) AS unknown_candidate_attempts,'
                "coalesce(sum(CASE WHEN m.state='DUPLICATE' THEN m.candidate_seconds ELSE 0 END),0) AS known_duplicate_candidate_seconds,"
                if has_duration else
                '0 AS known_pre_dedup_candidate_seconds,count(*) AS unknown_candidate_attempts,'
                '0 AS known_duplicate_candidate_seconds,')
            metrics = db.execute('SELECT j.task,j.asset_hash,count(*) AS timed_attempts,'
                + duration_sql +
                'sum(m.wall_seconds) AS worker_wall_seconds,'
                "sum(CASE WHEN m.state='DUPLICATE' THEN 1 ELSE 0 END) AS duplicate_attempts,"
                "sum(CASE WHEN m.state='DUPLICATE' THEN m.wall_seconds ELSE 0 END) AS duplicate_worker_seconds,"
                "sum(CASE WHEN m.state NOT IN ('EXPORTED','DUPLICATE') THEN 1 ELSE 0 END) AS execution_error_attempts,"
                "sum(CASE WHEN m.state NOT IN ('EXPORTED','DUPLICATE') THEN m.wall_seconds ELSE 0 END) AS execution_error_worker_seconds,"
                "sum(CASE WHEN m.state='EXPORTED' AND e.success=1 THEN e.seconds ELSE 0 END) AS timed_success_seconds,"
                "sum(CASE WHEN m.state='EXPORTED' AND e.success=0 THEN e.seconds ELSE 0 END) AS timed_failure_seconds "
                'FROM attempt_metrics m JOIN jobs j ON j.job_id=m.job_id '
                'LEFT JOIN episodes e ON e.job_id=m.job_id GROUP BY j.task,j.asset_hash').fetchall()
            for row in metrics:
                rows[(row['task'], row['asset_hash'])].update(dict(row))
        for row in db.execute('SELECT task,asset_hash,count(*) AS registered_episodes,'
                'sum(CASE WHEN success=1 THEN 1 ELSE 0 END) AS successful_episodes,'
                'sum(CASE WHEN success=0 THEN 1 ELSE 0 END) AS failed_episodes,'
                'sum(CASE WHEN success=1 THEN seconds ELSE 0 END) AS success_seconds,'
                'sum(CASE WHEN success=0 THEN seconds ELSE 0 END) AS failure_seconds '
                'FROM episodes GROUP BY task,asset_hash'):
            rows[(row['task'], row['asset_hash'])].update(dict(row))
        in_progress = live_untimed_attempts(db) if has_metrics else {}
        calibration = session_calibration(db, worker_plan(config)['pool_workers'], in_progress=sum(in_progress.values())) if has_metrics else {
            'unavailable_reasons': ['NO_ATTEMPT_TIMING_TABLE']}
        covered_assets = db.execute('SELECT count(DISTINCT asset_hash) FROM episodes' +
            (' WHERE success=1' if config.get('coverage_requires_success', True)
             or not config['count_failed_episodes'] else '')).fetchone()[0]
    finally:
        db.close()
    assets = []
    for row in rows.values():
        for key in ('timed_attempts', 'worker_wall_seconds', 'duplicate_attempts', 'duplicate_worker_seconds',
                    'execution_error_attempts', 'execution_error_worker_seconds',
                    'timed_success_seconds', 'timed_failure_seconds', 'registered_episodes',
                    'successful_episodes', 'failed_episodes',
                    'known_pre_dedup_candidate_seconds', 'unknown_candidate_attempts', 'known_duplicate_candidate_seconds',
                    'success_seconds', 'failure_seconds'):
            row.setdefault(key, 0)
        row['unmeasured_attempts'] = max(0, row['claimed_attempts'] - row['timed_attempts'])
        row['in_progress_untimed_attempts'] = in_progress.get((row['task'], row['asset_hash']), 0)
        row['unmeasured_completed_attempts'] = max(0, row['unmeasured_attempts'] - row['in_progress_untimed_attempts'])
        row['unknown_candidate_attempts'] += row['unmeasured_attempts']
        row['pre_dedup_candidate_seconds'] = (None if row['unknown_candidate_attempts'] or recovered_history
                                             else row['known_pre_dedup_candidate_seconds'])
        row['post_dedup_registered_seconds'] = row['success_seconds'] + row['failure_seconds']
        row['unique_registered_contents'] = row['registered_episodes']
        row['has_registered_successful_episode'] = row['successful_episodes'] > 0
        row['mean_timed_attempt_worker_seconds'] = row['worker_wall_seconds'] / row['timed_attempts'] if row['timed_attempts'] else None
        credited = row['timed_success_seconds'] + (row['timed_failure_seconds'] if config['count_failed_episodes'] else 0)
        row['timed_eligible_physical_seconds_per_worker_second'] = credited / row['worker_wall_seconds'] if row['worker_wall_seconds'] else None
        assets.append(row)
    targets = validate_outcome_hours(config['task_hours'], config.get('task_outcome_hours'), config['count_failed_episodes'])
    tasks = {}
    for task in TASKS:
        grouped = [r for r in assets if r['task'] == task]
        counts = {key: sum(r[key] for r in grouped) for key in (
            'claimed_attempts', 'timed_attempts', 'unmeasured_attempts', 'unmeasured_completed_attempts',
            'in_progress_untimed_attempts', 'worker_wall_seconds',
            'success_seconds', 'failure_seconds', 'timed_success_seconds', 'timed_failure_seconds')}
        counts['known_pre_dedup_candidate_seconds'] = sum(r['known_pre_dedup_candidate_seconds'] for r in grouped)
        counts['unknown_candidate_attempts'] = sum(r['unknown_candidate_attempts'] for r in grouped)
        counts['pre_dedup_candidate_seconds'] = (None if counts['unknown_candidate_attempts'] or recovered_history
                                               else counts['known_pre_dedup_candidate_seconds'])
        counts['post_dedup_registered_seconds'] = counts['success_seconds'] + counts['failure_seconds']
        if targets is None:
            remaining = max(0., config['task_hours'][task] * 3600 - counts['success_seconds'] - (
                counts['failure_seconds'] if config['count_failed_episodes'] else 0))
            rate = (counts['timed_success_seconds'] + (counts['timed_failure_seconds'] if config['count_failed_episodes'] else 0))
            requirements = [(remaining, rate)]
        else:
            requirements = [(max(0., targets[task][outcome] * 3600 - counts[outcome + '_seconds']),
                             counts['timed_' + outcome + '_seconds']) for outcome in ('success', 'failure')]
        estimates = []
        for remaining, produced in requirements:
            estimates.append(0. if not remaining else remaining * counts['worker_wall_seconds'] / produced
                             if produced and counts['worker_wall_seconds'] and not counts['unmeasured_completed_attempts'] and not recovered_history else None)
        counts['remaining_worker_seconds_at_observed_mix'] = max(estimates) if all(v is not None for v in estimates) else None
        tasks[task] = counts
    projection = remaining_time_projection(tasks, calibration,
        coverage_missing=covered_assets < config['minimum_independent_assets'], recovered_history=recovered_history)
    return {'available': True, 'assets': sorted(assets, key=lambda r: (r['task'], r['asset_hash'])),
            'tasks': tasks, 'wall_clock_eta_seconds': projection['wall_clock_eta_seconds'],
            'remaining_time_projection': projection,
            'candidate_duration_note': 'Candidate durations observed per attempt before dedup; recovery can observe the same output again. Not new execution hours or accepted quota. Unknown histories are not zero.',
            'historical_attempt_counts_unknown_after_rebuild': recovered_history,
            'attempt_count_note': 'Reconstructed quarantine counts consume budget conservatively; they are not measured executions.' if recovered_history
            else 'Attempt counts come from queue claims.',
            'eta_note': 'Conditional wall-clock projection requires complete session timing and task/outcome evidence; asset-coverage completion time remains unknown until coverage is met.',
            'scope': 'SQL monitoring counters, not independent final file acceptance; lost/interrupted timings remain unknown.'}
