"""Durable startup intent before dataset creation, with idempotent metadata."""

import fcntl
import json
import os
from pathlib import Path
import time

from source.data.five_task_dataset import canonical_hash


def intent_path(root):
    root = Path(root).resolve()
    return root.parent / ('.' + root.name + '.bootstrap.json')


def bootstrap_pending(root):
    return intent_path(root).is_file() and not (Path(root) / 'bootstrap_complete.json').exists()


def install_json(path, value):
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError(f'Frozen initialization metadata differs: {path.name}')
        return
    temporary = path.with_name(path.name + f'.{time.time_ns()}.tmp')
    with temporary.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def reserve_bootstrap(root, database, config, contract, seed, initial_jobs, *, resume):
    """Return a startup lock; caller closes it only after taking coordinator lock."""
    root = Path(root).resolve()
    path = intent_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    lock = path.with_suffix('.lock').open('a')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if root.exists() and not resume:
            raise FileExistsError('Pilot root exists; use --resume or a fresh root')
        expected = {'schema': 'production-bootstrap-v1', 'dataset_root': str(root),
                    'database': str(Path(database).resolve()), 'configuration': config,
                    'run_contract': contract, 'sampling_seed': seed}
        if path.exists():
            record = json.loads(path.read_text())
            if any(record.get(key) != value for key, value in expected.items()):
                raise ValueError('Resume configuration, seed or frozen initialization contract differs')
            if isinstance(record.get('initial_jobs'), bool) or not isinstance(record.get('initial_jobs'), int) or not 1 <= record['initial_jobs'] <= 40:
                raise ValueError('Invalid initialization job count')
        else:
            if resume or root.exists():
                raise ValueError('Missing startup intent; cannot adopt an unowned existing dataset')
            record = {**expected, 'initial_jobs': initial_jobs}
            install_json(path, record)
        return lock, record
    except BaseException:
        lock.close()
        raise


def install_metadata(root, record):
    root = Path(root)
    complete = root / 'bootstrap_complete.json'
    expected = {'schema': 'production-bootstrap-complete-v1', 'intent_hash': canonical_hash(record)}
    if complete.exists() and json.loads(complete.read_text()) != expected:
        raise ValueError('Initialization completion proof differs')
    files = {'config.json': record['configuration']}
    if record['run_contract'] is not None:
        files['run_contract.json'] = record['run_contract']
    for name, value in files.items():
        if complete.exists() and not (root / name).exists():
            raise ValueError('Completed dataset lost frozen initialization metadata')
        install_json(root / name, value)
    return complete.exists()


def finish_bootstrap(root, record, queue, enqueue):
    complete = Path(root) / 'bootstrap_complete.json'
    if complete.exists():
        return
    with queue.connect() as db:
        if db.execute('SELECT count(*) FROM jobs WHERE attempts>0 OR state<>?', ('PENDING',)).fetchone()[0]:
            raise ValueError('Incomplete initialization has execution history; refusing reset')
        if db.execute('SELECT count(*) FROM episodes').fetchone()[0]:
            raise ValueError('Incomplete initialization already has registered data')
        count = db.execute('SELECT count(*) FROM jobs').fetchone()[0]
    if count > record['initial_jobs']:
        raise ValueError('Incomplete initialization has unexpected extra jobs')
    for _ in range(record['initial_jobs'] - count):
        if enqueue()['state'] != 'ENQUEUED':
            break
    install_json(complete, {'schema': 'production-bootstrap-complete-v1', 'intent_hash': canonical_hash(record)})
