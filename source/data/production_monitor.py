"""Atomic, replaceable runtime snapshots; never used for quota accounting."""

import json
import os
import time
import uuid


def write_worker_status(root, slot, payload):
    directory = root / 'runtime'
    directory.mkdir(exist_ok=True)
    target = directory / f'worker_{slot}.json'
    temporary = directory / f'.worker_{slot}_{uuid.uuid4().hex}.tmp'
    with temporary.open('x') as stream:
        json.dump({'updated_unix_seconds': time.time(), 'coordinator_pid': os.getpid(), **payload},
                  stream, indent=2, allow_nan=False)
    temporary.replace(target)


def record_runtime_failure(root, slot, payload):
    directory = root / 'runtime/failures'
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f'{time.time_ns()}_{slot}.json'
    with path.open('x') as stream:
        json.dump({'schema': 'runtime-failure-v1', 'recorded_unix_seconds': time.time(),
                   'coordinator_pid': os.getpid(), **payload}, stream, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    return str(path)
