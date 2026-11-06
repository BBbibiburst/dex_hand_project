"""Recover a candidate inventory from verified files without replacing the old index."""

import json
from pathlib import Path

from source.data.five_task_dataset import canonical_hash
from source.data.production_artifacts import verify_episode
from source.data.production_inventory import audit_inventory
from source.data.production_queue import ProductionQueue


def recover_payload(root, directory, metadata):
    paths = [directory / 'job_payload.json']
    paths.extend(sorted((root / 'work' / metadata['job_id']).glob('job_*.json')))
    for path in paths:
        if path.is_file():
            payload = json.loads(path.read_text())
            if canonical_hash(payload) == metadata['job_payload_hash']:
                return payload
    if metadata['job_payload_hash'] == canonical_hash({}):
        return {}  # Explicit legacy development fixture contract only.
    raise ValueError('Cannot recover immutable job payload from files')


def rebuild_inventory(root, destination, *, quarantine_unfinished=False, maximum_retries=3):
    if isinstance(maximum_retries, bool) or not isinstance(maximum_retries, int) or maximum_retries < 1:
        raise ValueError('Recovery retry budget must be a positive integer')
    root, destination = Path(root).resolve(), Path(destination).resolve()
    if destination.exists() or destination.is_relative_to(root):
        raise ValueError('Rebuild requires a fresh output directory outside the dataset')
    rows, failures, contents, lineages = [], [], set(), set()
    for directory in sorted((root / 'episodes').glob('*')):
        try:
            if not directory.is_dir() or directory.is_symlink():
                raise ValueError('Unexpected episode entry')
            result = verify_episode(directory)
            metadata = json.loads((directory / 'episode.json').read_text())
            if metadata['job_id'] != directory.name:
                raise ValueError('Episode directory and job identity disagree')
            payload = recover_payload(root, directory, metadata)
            if quarantine_unfinished and (not payload.get('template_id')
                    or payload.get('task') != result['task'] or payload.get('asset_hash') != result['asset_hash']
                    or metadata['job_id'] != payload['template_id'] + '_' + canonical_hash(payload)[:20]
                    or isinstance(payload.get('candidate_index'), bool)
                    or not isinstance(payload.get('candidate_index'), int) or payload['candidate_index'] < 0):
                raise ValueError('Automatic reconstruction requires a versioned candidate identity')
            if result['content_hash'] in contents or result['lineage_hash'] in lineages:
                raise ValueError('Duplicate content or lineage among recovered files')
            contents.add(result['content_hash'])
            lineages.add(result['lineage_hash'])
            rows.append((directory, metadata, payload, result))
        except Exception as error:
            failures.append({'path': str(directory), 'error': f'{type(error).__name__}: {error}'})
    has_work = any(path.is_dir() for path in (root / 'work').glob('*'))
    if failures or (not rows and not (quarantine_unfinished and has_work)):
        raise ValueError(json.dumps({'reason': 'File recovery refused', 'errors': failures,
                                     'verified_episodes': len(rows)}))
    destination.mkdir(parents=True)
    queue = ProductionQueue(destination / 'inventory.sqlite')
    for directory, metadata, payload, result in rows:
        job_id = metadata['job_id']
        queue.enqueue(job_id, result['task'], result['asset_hash'], payload, max_attempts=1)
        # Reconstruct terminal rows directly. Do not fabricate a RUNNING event or
        # optimizer/simulation attempt to make the existing artifact register.
        with queue.transaction() as db:
            row = db.execute('SELECT * FROM jobs WHERE job_id=?', (job_id,)).fetchone()
            db.execute('INSERT INTO episodes VALUES(?,?,?,?,?,?,?,?,?)',
                       (job_id, result['task'], result['asset_hash'], result['lineage_hash'],
                        result['content_hash'], str(directory.relative_to(root)),
                        result['duration_seconds'], int(result['success']), json.dumps(result)))
            db.execute("UPDATE jobs SET state='EXPORTED',error=? WHERE job_id=?",
                       ('Historical attempt count unknown after file-only recovery', job_id))
            queue._terminal(db, row, 'EXPORTED', result['success'])
            queue.event(db, job_id, 'INDEX_REBUILT', {'historical_attempt_count': None,
                                                    'new_simulation_attempts': 0})
    # No pending job is guessed or retried: its attempt/retry history may be lost.
    recovered_ids = {metadata['job_id'] for _, metadata, _, _ in rows}
    unresolved = sorted(path.name for path in (root / 'work').glob('*')
                        if path.is_dir() and path.name not in recovered_ids)
    quarantined = []
    if quarantine_unfinished:
        for job_id in unresolved:
            if (root / 'work' / job_id).is_symlink():
                raise ValueError('Unfinished work must not escape through a symlink')
            files = sorted((root / 'work' / job_id).glob('job_*.json'))
            if not files:
                raise ValueError(f'Unfinished source has no recoverable payload: {job_id}')
            payloads = [json.loads(path.read_text()) for path in files]
            payload = payloads[0]
            if (any(p != payload for p in payloads)
                    or job_id != payload['template_id'] + '_' + canonical_hash(payload)[:20]
                    or isinstance(payload['candidate_index'], bool)
                    or not isinstance(payload['candidate_index'], int) or payload['candidate_index'] < 0):
                raise ValueError(f'Unfinished job identity or candidate index differs: {job_id}')
            queue.enqueue(job_id, payload['task'], payload['asset_hash'], payload, max_attempts=maximum_retries)
            with queue.transaction() as db:
                row = db.execute('SELECT * FROM jobs WHERE job_id=?', (job_id,)).fetchone()
                db.execute("UPDATE jobs SET state='QUARANTINED',attempts=max_attempts,error=? WHERE job_id=?",
                           ('Recovery consumed unknown historical retry budget conservatively', job_id))
                queue._terminal(db, row, 'QUARANTINED')
                queue.event(db, job_id, 'INDEX_REBUILT_QUARANTINE', {'historical_attempt_count': None,
                    'accounted_as_budget_exhausted': True, 'new_simulation_attempts': 0})
            quarantined.append(job_id)
    with queue.transaction() as db:
        db.execute('INSERT INTO queue_meta VALUES(?,?)',
                   ('file_rebuild_history', json.dumps({'historical_attempt_counts': 'UNKNOWN',
                                                       'unresolved_work': unresolved})))
    audit = audit_inventory(root, queue.path)
    report = {'schema': 'five-task-inventory-rebuild-v1', 'dataset_root': str(root),
              'candidate_database': str(queue.path), 'adopted': False,
              'new_simulation_seconds': 0, 'historical_attempt_counts': 'UNKNOWN',
              'unresolved_work': unresolved, 'quarantined_unfinished_jobs': quarantined, 'audit': audit,
              'automatic_resume_allowed': bool(quarantine_unfinished and audit['passed']),
              'reason': 'File-verified recovery; unfinished retry budgets consumed' if quarantine_unfinished
              else 'Candidate only; lost retry history requires reconciliation before adoption'}
    (destination / 'rebuild_report.json').write_text(json.dumps(report, indent=2))
    return report
