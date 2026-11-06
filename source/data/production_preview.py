"""Run the actual scheduler against a disposable SQLite snapshot, never claim work."""

from pathlib import Path
import json
import sqlite3
import tempfile

from source.data.production_queue import ProductionQueue
from source.data.production_scheduler import enqueue_next


def preview(config, templates, sampling_seed):
    from source.data.production_index import active_database

    database = active_database(config.get('dataset_root', '.'), config['queue_database'])
    with tempfile.TemporaryDirectory(prefix='five-task-preview-') as temporary:
        snapshot = Path(temporary) / 'inventory.sqlite'
        if database.exists():
            # SQLite backup includes committed WAL pages and provides a consistent
            # view even while the real coordinator is updating its inventory.
            source = sqlite3.connect(database.resolve().as_uri() + '?mode=ro', uri=True)
            destination = sqlite3.connect(snapshot)
            try:
                source.backup(destination)
            finally:
                destination.close()
                source.close()
        queue = ProductionQueue(snapshot)
        decision = enqueue_next(queue, templates, config, sampling_seed)
        payload = None
        if decision['state'] in ('ENQUEUED', 'ALREADY_PLANNED'):
            with queue.connect() as db:
                payload = json.loads(db.execute('SELECT payload FROM jobs WHERE job_id=?',
                                               (decision['job_id'],)).fetchone()['payload'])
        return {'mode': 'SCHEDULER_PREVIEW', 'queue_exists': database.exists(),
                'source_database': str(database), 'sampling_seed': sampling_seed,
                'decision': decision, 'candidate_payload': payload,
                'jobs_persisted': 0, 'simulation_started': False,
                'production_ready': False,
                'scope': 'Next scheduler decision only; no lease recovery, claim, rendering or production preflight'}
