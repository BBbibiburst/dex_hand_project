"""Build current execution evidence by task and independent asset, not file count."""

import json
from pathlib import Path
import sqlite3


def compatibility_matrix(database, file_audit=None):
    db = sqlite3.connect(Path(database).resolve().as_uri() + '?mode=ro', uri=True)
    db.row_factory = sqlite3.Row
    try:
        rows = db.execute('SELECT j.job_id,j.task,j.asset_hash,j.payload,j.state,j.error,e.success '
                          'FROM jobs j LEFT JOIN episodes e ON j.job_id=e.job_id ORDER BY j.created_at,j.job_id').fetchall()
    finally:
        db.close()
    verified = {row['job_id']: row for row in (file_audit or {}).get('episodes', []) if row.get('passed')}
    result = []
    for row in rows:
        payload = json.loads(row['payload'])
        result.append({'job_id': row['job_id'], 'task': row['task'], 'asset_hash': row['asset_hash'],
                       'object_id': payload.get('object_id'), 'template_id': payload.get('template_id'),
                       'zero_perturbation': not any(payload.get('conditions', {}).values()),
                       'state': row['state'], 'source_success': None if row['success'] is None else bool(row['success']),
                       'error': row['error'],
                       'task_outcome': verified.get(row['job_id'], {}).get('task_outcome'),
                       'compatibility': 'CURRENT_SINGLE_CONDITION_SUCCESS' if row['state'] == 'EXPORTED' and row['success'] and row['job_id'] in verified
                       else 'REGISTERED_SUCCESS_REQUIRES_FILE_AUDIT' if row['state'] == 'EXPORTED' and row['success']
                       else 'CURRENT_TASK_FAILURE' if row['state'] == 'EXPORTED' else 'NO_ACCEPTED_EPISODE'})
    return {'rows': result,
            'current_success_assets': len({r['asset_hash'] for r in result if r['compatibility'] == 'CURRENT_SINGLE_CONDITION_SUCCESS'}),
            'file_audit_passed': (file_audit or {}).get('passed', False),
            'all_selected_cases_passed': bool(result) and (file_audit or {}).get('passed', False)
            and all(r['compatibility'] == 'CURRENT_SINGLE_CONDITION_SUCCESS' for r in result),
            'scope': 'One-condition current execution evidence; does not prove perturbation robustness or 100-asset coverage'}
