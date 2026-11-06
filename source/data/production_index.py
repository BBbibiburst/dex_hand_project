"""Preserve corrupt inventories and activate independently verified recovery files."""

import json
import os
from pathlib import Path
import sqlite3
import time

from source.data.production_queue import ProductionQueue
from source.data.five_task_dataset import sha256_file


def recovery_base(root):
    root = Path(root).resolve()
    return root.parent / (root.name + '_index_recovery')


def pointer_path(database):
    database = Path(database)
    return database.with_name(database.name + '.active.json')


def is_database_corruption(error):
    if not isinstance(error, sqlite3.DatabaseError):
        return False
    code = getattr(error, 'sqlite_errorcode', None)
    return (code & 255 in (11, 26) if code is not None else str(error) in (
        'file is not a database', 'database disk image is malformed'))


def active_database(root, database):
    database = Path(database).resolve()
    pointer = pointer_path(database)
    if not pointer.exists():
        return database
    record = json.loads(pointer.read_text())
    selected = Path(record['active_database']).resolve()
    if (record.get('schema') != 'production-active-index-v1'
            or record['original_database'] != str(database)
            or record['dataset_root'] != str(Path(root).resolve())
            or not selected.is_relative_to(recovery_base(root).resolve())
            or not selected.is_file()):
        raise ValueError('Active inventory pointer identity is invalid')
    return selected


def live_dataset_processes(root):
    found = []
    root = Path(root).resolve()
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit() or int(entry.name) == os.getpid():
            continue
        try:
            args = (entry / 'cmdline').read_bytes().decode(errors='replace').split('\0')
            if any(Path(part).name == 'production_worker.py' for part in args) and '--output' in args:
                candidates = args[args.index('--output') + 1:args.index('--output') + 2]
            elif args and Path(args[0]).name.startswith('ffmpeg'):
                candidates = [arg for arg in args if arg.endswith('.mp4')]
            else:
                continue
            cwd = (entry / 'cwd').resolve()
            if any((cwd / arg).resolve().is_relative_to(root) for arg in candidates):
                found.append(int(entry.name))
        except (FileNotFoundError, ProcessLookupError):
            continue
        except PermissionError as error:
            raise RuntimeError('Cannot prove dataset workers are stopped') from error
    return found


def open_inventory(root, database, *, auto_rebuild=True, maximum_retries=3, initializing=False):
    """Caller must hold the exclusive coordinator lock throughout this operation."""
    database = Path(database).resolve()
    selected = active_database(root, database)
    corrupted = None
    if selected.exists():
        db = None
        try:
            if not selected.stat().st_size and not initializing:
                corrupted = 'Existing inventory is empty'
            else:
                db = sqlite3.connect(selected.as_uri() + '?mode=ro', uri=True)
                check = [row[0] for row in db.execute('PRAGMA quick_check')]
                if check != ['ok']:
                    corrupted = '; '.join(check)
        except sqlite3.DatabaseError as error:
            # Python 3.10 does not expose SQLite error codes. Accept only the
            # two exact SQLite corruption messages on that runtime.
            if not is_database_corruption(error):
                raise
            corrupted = f'{type(error).__name__}: {error}'
        finally:
            if db is not None:
                db.close()
    if corrupted is None:
        return ProductionQueue(selected, initializing=initializing)
    if not auto_rebuild:
        raise RuntimeError('Inventory corrupt; automatic recovery disabled: ' + corrupted)
    live = live_dataset_processes(root)
    if live:
        raise RuntimeError(f'Cannot rebuild while dataset processes are alive: {live}')
    from source.data.production_rebuild import rebuild_inventory

    output = recovery_base(root) / f'recovered_{time.time_ns()}'
    report = rebuild_inventory(root, output, quarantine_unfinished=True, maximum_retries=maximum_retries)
    if not report['automatic_resume_allowed']:
        raise RuntimeError('File reconstruction did not pass independent verification')
    record = {'schema': 'production-active-index-v1', 'dataset_root': str(Path(root).resolve()),
              'original_database': str(database), 'previous_database': str(selected),
              'active_database': report['candidate_database'], 'corruption': corrupted,
              'old_database_sha256': sha256_file(selected), 'adopted': True,
              'rebuild_report': str(output / 'rebuild_report.json')}
    # Immutable activation evidence precedes the small atomic active pointer.
    with (output / 'activation.json').open('x') as stream:
        json.dump(record, stream, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    for directory in (output, output.parent):
        descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    pointer = pointer_path(database)
    temporary = pointer.with_name(pointer.name + f'.{time.time_ns()}.tmp')
    with temporary.open('x') as stream:
        json.dump(record, stream, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(pointer)
    directory_fd = os.open(pointer.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    return ProductionQueue(Path(report['candidate_database']))
