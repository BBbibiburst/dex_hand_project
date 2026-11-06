"""Persistent bounded rotation over already registered episode files."""

import json
import os
import time

from source.data.production_inventory import verify_inventory_row


def audit_after_attempt(queue, root, *, interval=10, force=False, count_attempt=True):
    if isinstance(interval, bool) or not isinstance(interval, int) or interval < 1:
        raise ValueError('Integrity audit interval must be a positive integer')
    key = 'periodic_file_audit_v1'
    with queue.transaction() as db:
        stored = db.execute('SELECT value FROM queue_meta WHERE key=?', (key,)).fetchone()
        state = json.loads(stored['value']) if stored else {'attempts': 0, 'cursor': 0, 'ceiling': 0, 'cycle': 0}
        state['attempts'] += int(count_attempt)
        due = force or state.get('failed', False) or (count_attempt and state['attempts'] % interval == 0)
        row = None
        if due:
            if state['cursor'] >= state['ceiling']:
                state.update(cursor=0, ceiling=db.execute('SELECT coalesce(max(rowid),0) FROM episodes').fetchone()[0],
                             cycle=state['cycle'] + 1)
            row = db.execute('SELECT e.rowid AS audit_rowid,e.*,j.payload AS planned_payload FROM episodes e '
                             'LEFT JOIN jobs j ON e.job_id=j.job_id WHERE e.rowid>? AND e.rowid<=? '
                             'ORDER BY e.rowid LIMIT 1', (state['cursor'], state['ceiling'])).fetchone()
        db.execute('INSERT INTO queue_meta VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value',
                   (key, json.dumps(state)))
    if not due or row is None:
        return {'state': 'NOT_DUE' if not due else 'NO_REGISTERED_EPISODE'}
    started = time.monotonic()
    report = {'schema': 'periodic-file-audit-v1', 'job_id': row['job_id'], 'cycle': state['cycle'],
              'scope': 'One registered episode, not complete inventory acceptance'}
    try:
        verified = verify_inventory_row(root, row)
        report.update(passed=True, frames=verified['frames'])
    except Exception as error:
        report.update(passed=False, error=f'{type(error).__name__}: {error}')
    report['wall_seconds'] = time.monotonic() - started
    directory = root / 'runtime/integrity'
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f'{time.time_ns()}.json'
    with path.open('x') as stream:
        json.dump(report, stream, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    state['failed'] = not report['passed']
    if report['passed']:
        state['cursor'] = row['audit_rowid']
    with queue.transaction() as db:
        db.execute('UPDATE queue_meta SET value=? WHERE key=?', (json.dumps(state), key))
    return {'state': 'PASSED' if report['passed'] else 'FAILED', 'record_path': str(path), **report}
