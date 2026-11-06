"""Measured coordinator concurrency for conditional remaining-time projections."""

import json
import math
import os
import time
import uuid

from source.data.production_queue import process_identity


def _counters(db):
    row = db.execute('SELECT count(*),coalesce(sum(wall_seconds),0) FROM attempt_metrics').fetchone()
    return {'timed_attempts': row[0], 'worker_seconds': row[1],
            'claimed_attempts': db.execute('SELECT coalesce(sum(attempts),0) FROM jobs').fetchone()[0]}


def begin_session(queue, workers, *, started=None):
    identifier = 'runtime-session-' + uuid.uuid4().hex
    started = time.monotonic() if started is None else started
    with queue.transaction() as db:
        queue.event(db, identifier, 'RUNTIME_SESSION_STARTED', {**_counters(db), 'workers': workers,
                    'owner': process_identity(os.getpid()), 'started_monotonic': started})
    return identifier, started


def finish_session(queue, session):
    identifier, started = session
    with queue.transaction() as db:
        if db.execute("SELECT 1 FROM events WHERE job_id=? AND state='RUNTIME_SESSION_FINISHED'", (identifier,)).fetchone():
            raise ValueError('Session timing is immutable')
        queue.event(db, identifier, 'RUNTIME_SESSION_FINISHED',
                    {**_counters(db), 'wall_seconds': time.monotonic() - started})


def live_untimed_attempts(db):
    grouped = {}
    for row in db.execute("SELECT j.* FROM jobs j WHERE state IN ('RUNNING','VALIDATING') "
                          "AND NOT EXISTS (SELECT 1 FROM attempt_metrics m WHERE m.job_id=j.job_id AND m.attempt=j.attempts)"):
        event = db.execute("SELECT details FROM events WHERE job_id=? AND state='RUNNING' ORDER BY id DESC LIMIT 1",
                           (row['job_id'],)).fetchone()
        details = json.loads(event['details']) if event else {}
        owner = details.get('owner')
        if (owner is not None and process_identity(owner['pid']) == owner
                and details.get('lease_token') == row['lease_token'] and row['lease_until'] > time.time()):
            key = (row['task'], row['asset_hash'])
            grouped[key] = grouped.get(key, 0) + 1
    return grouped


def session_calibration(db, workers, *, in_progress=0):
    sessions = {}
    for row in db.execute("SELECT job_id,state,details FROM events WHERE state IN "
                          "('RUNTIME_SESSION_STARTED','RUNTIME_SESSION_FINISHED') ORDER BY id"):
        sessions.setdefault(row['job_id'], {}).setdefault(row['state'], []).append(json.loads(row['details']))
    reasons = []
    worker_seconds = wall_seconds = 0.
    complete = 0
    live = 0
    accounted_metrics = 0
    for pair in sessions.values():
        starts = pair.get('RUNTIME_SESSION_STARTED', [])
        ends = pair.get('RUNTIME_SESSION_FINISHED', [])
        if len(starts) == 1 and not ends:
            start = starts[0]
            owner = start.get('owner')
            if owner is not None and process_identity(owner['pid']) == owner and 'started_monotonic' in start:
                ends = [{**_counters(db), 'wall_seconds': time.monotonic() - start['started_monotonic']}]
                live += 1
            else:
                reasons.append('UNFINISHED_OR_AMBIGUOUS_SESSION')
                continue
        if len(starts) != 1 or len(ends) != 1:
            reasons.append('UNFINISHED_OR_AMBIGUOUS_SESSION')
            continue
        start, end = starts[0], ends[0]
        if start['workers'] != workers:
            reasons.append('WORKER_CONFIGURATION_CHANGED')
            continue
        duration = end['wall_seconds']
        work = end['worker_seconds'] - start['worker_seconds']
        timed = end['timed_attempts'] - start['timed_attempts']
        claimed = end['claimed_attempts'] - start['claimed_attempts']
        if (not all(math.isfinite(v) and v >= 0 for v in (duration, work, timed, claimed))
                or duration <= 0 or work > duration * workers * 1.01
                or timed != claimed - (in_progress if not pair.get('RUNTIME_SESSION_FINISHED') else 0)):
            reasons.append('INCOMPLETE_OR_INVALID_SESSION_TIMING')
            continue
        complete += int(bool(pair.get('RUNTIME_SESSION_FINISHED')))
        worker_seconds += work
        wall_seconds += duration
        accounted_metrics += timed
    counters = _counters(db)
    if (accounted_metrics != counters['timed_attempts']
            or counters['claimed_attempts'] - in_progress != counters['timed_attempts']):
        reasons.append('HISTORICAL_OR_IN_PROGRESS_TIMING_GAP')
    if not complete + live or worker_seconds <= 0:
        reasons.append('NO_MEASURED_SESSION_THROUGHPUT')
    if live > 1:
        reasons.append('OVERLAPPING_LIVE_SESSIONS')
    return {'completed_sessions': complete, 'live_sessions': live, 'coordinator_wall_seconds': wall_seconds,
            'worker_seconds': worker_seconds, 'configured_workers': workers,
            'measured_worker_seconds_per_wall_second': worker_seconds / wall_seconds if wall_seconds else None,
            'unavailable_reasons': sorted(set(reasons))}


def remaining_time_projection(tasks, calibration, *, coverage_missing, recovered_history):
    reasons = list(calibration['unavailable_reasons'])
    if recovered_history:
        reasons.append('REBUILT_EXECUTION_HISTORY_UNKNOWN')
    for task, row in tasks.items():
        if row['remaining_worker_seconds_at_observed_mix'] is None:
            reasons.append('UNMEASURED_TASK_OR_REQUIRED_OUTCOME:' + task)
        elif row['remaining_worker_seconds_at_observed_mix'] > 0 and row['timed_attempts'] < 3:
            reasons.append('INSUFFICIENT_TASK_ATTEMPTS:' + task)
    projected = None
    if not reasons:
        total = sum(row['remaining_worker_seconds_at_observed_mix'] for row in tasks.values())
        projected = total / calibration['measured_worker_seconds_per_wall_second']
    return {'schema': 'production-conditional-eta-v1', 'calibration': calibration,
            'hours_only_wall_seconds_at_observed_mix': projected,
            'wall_clock_eta_seconds': projected if not coverage_missing else None,
            'unavailable_reasons': sorted(set(reasons + (['ASSET_COVERAGE_TIME_UNKNOWN'] if coverage_missing else []))),
            'assumptions': 'Conditional point projection at the observed task/outcome mix and measured coordinator concurrency. Not a confidence interval or finish-date promise; future object difficulty, downtime and final audit cost can change it.'}
