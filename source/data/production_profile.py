"""Wall-clock diagnostics; inclusive/thread-overlap spans must not be summed."""

from contextlib import contextmanager
import json
import math
import threading
import time


class PhaseProfile:
    def __init__(self):
        self._lock = threading.Lock()
        self._phases = {}

    def add(self, name, seconds):
        with self._lock:
            row = self._phases.setdefault(name, {'seconds': 0., 'calls': 0})
            row['seconds'] += seconds
            row['calls'] += 1

    @contextmanager
    def measure(self, name):
        started = time.monotonic()
        try:
            yield
        finally:
            self.add(name, time.monotonic() - started)

    def call(self, name, function, *args, **kwargs):
        with self.measure(name):
            return function(*args, **kwargs)

    def snapshot(self):
        with self._lock:
            phases = {name: dict(row) for name, row in self._phases.items()}
        return {'schema': 'production-wall-profile-v1', 'phases': phases,
                'timing_semantics': 'Wall spans. Inclusive callers and async encoder spans overlap; do not sum across scopes.'}


def summarize_profiles(root):
    scopes = {}
    errors = []
    paths = [(path, 'coordinator_attempt', 'profile') for path in (root / 'runtime/profiles').glob('*.json')]
    paths += [(path, 'worker', 'wall_profile') for path in (root / 'work').glob('*/attempt_*/worker_result.json')]
    paths += [(path, scope, field) for path in (root / 'work').glob('*/attempt_*/result.json')
              for scope, field in (('online_execution', 'wall_profile'), ('video', 'video_wall_profile'))]
    for path, scope, field in paths:
        try:
            record = json.loads(path.read_text()).get(field)
            if record is None:
                continue
            if record.get('schema') != 'production-wall-profile-v1':
                raise ValueError('Unknown timing schema')
            for row in record['phases'].values():
                if (not isinstance(row['seconds'], (int, float)) or not math.isfinite(row['seconds'])
                        or row['seconds'] < 0 or type(row['calls']) is not int or row['calls'] < 0):
                    raise ValueError('Invalid timing value or call count')
            aggregate = scopes.setdefault(scope, {'profile_records': 0, 'phases': {}})
            aggregate['profile_records'] += 1
            for name, row in record['phases'].items():
                total = aggregate['phases'].setdefault(name, {'seconds': 0., 'calls': 0})
                total['seconds'] += row['seconds']
                total['calls'] += row['calls']
        except Exception as error:
            errors.append({'path': str(path), 'scope': scope, 'error': f'{type(error).__name__}: {error}'})
    return {'available': bool(scopes), 'scopes': scopes, 'errors': errors,
            'timing_semantics': 'Inclusive callers and async encoder phases overlap; no cross-scope total is computed.'}
