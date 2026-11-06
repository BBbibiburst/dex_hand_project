"""Resource-adaptive worker admission for different production hosts."""

from functools import wraps
import math
import os
from pathlib import Path
import threading
import time


POLICY = 'resource-adaptive-v1'


def validate_worker_config(config):
    workers = config['workers']
    adaptive = config.get('worker_policy') == POLICY
    if config.get('worker_policy') not in (None, POLICY):
        raise ValueError('Unknown worker policy')
    if not adaptive:
        if type(workers) is not int or workers not in (1, 2):
            raise ValueError('More workers or auto require explicit resource-adaptive-v1 policy')
        return
    if workers != 'auto' and (type(workers) is not int or not 1 <= workers <= 128):
        raise ValueError('Adaptive workers must be auto or an integer from 1 to 128')
    for key, default, low, high in (
        ('maximum_workers', 64, 1, 128), ('cpu_threads_per_worker', 2, 1, 64),
        ('estimated_worker_memory_bytes', 2 * 1024**3, 256 * 1024**2, 1024**4)):
        value = config.get(key, default)
        if type(value) is not int or not low <= value <= high:
            raise ValueError(f'Invalid {key}')


def _number(path):
    try:
        return int(Path(path).read_text().strip())
    except (OSError, ValueError):
        return None


def hardware_resources():
    import psutil

    memory = psutil.virtual_memory()
    cpus = os.cpu_count() or 1
    if hasattr(os, 'sched_getaffinity'):
        cpus = min(cpus, len(os.sched_getaffinity(0)))
    try:
        quota, period = Path('/sys/fs/cgroup/cpu.max').read_text().split()
        if quota != 'max':
            cpus = min(cpus, max(1, math.floor(int(quota) / int(period))))
    except (OSError, ValueError, ZeroDivisionError):
        pass
    total, available = memory.total, memory.available
    limit = _number('/sys/fs/cgroup/memory.max')
    used = _number('/sys/fs/cgroup/memory.current')
    if limit is not None and used is not None:
        total = min(total, limit)
        available = min(available, max(0, limit - used))
    return {'cpu_capacity': max(1, cpus), 'total_memory_bytes': total,
            'available_memory_bytes': available}


def worker_plan(config, resources=None):
    validate_worker_config(config)
    if config.get('worker_policy') != POLICY:
        return {'policy': 'fixed-legacy', 'pool_workers': config['workers'],
                'initial_admitted_workers': config['workers']}
    resources = hardware_resources() if resources is None else resources
    reserve = config.get('minimum_available_memory_bytes', 1024**3)
    estimate = config.get('estimated_worker_memory_bytes', 2 * 1024**3)
    requested = config.get('maximum_workers', 64) if config['workers'] == 'auto' else config['workers']
    ceiling = min(requested, config.get('maximum_workers', 64),
                  max(1, resources['cpu_capacity'] // config.get('cpu_threads_per_worker', 2)),
                  max(1, (resources['total_memory_bytes'] - reserve) // estimate))
    initial = max(0, min(ceiling, (resources['available_memory_bytes'] - reserve) // estimate))
    return {'policy': POLICY, 'pool_workers': int(ceiling), 'initial_admitted_workers': int(initial),
            'estimated_worker_memory_bytes': estimate, 'reserve_memory_bytes': reserve,
            'resources': dict(resources),
            'note': 'CPU/RAM resource ceiling, not measured throughput or a GPU scaling guarantee.'}


class WorkerAdmission:
    def __init__(self, config):
        self.config = config
        self.plan = worker_plan(config)
        self._lock = threading.Lock()
        self.active = {}

    def attach_process(self, slot, pid):
        with self._lock:
            if slot in self.active:
                self.active[slot] = pid

    def _pending_memory(self):
        import psutil

        pending = 0
        for pid in self.active.values():
            resident = 0
            if pid is not None:
                try:
                    process = psutil.Process(pid)
                    resident = process.memory_info().rss
                    for child in process.children(recursive=True):
                        try:
                            resident += child.memory_info().rss
                        except psutil.Error:
                            pass
                except psutil.Error:
                    pass
            pending += max(0, self.plan['estimated_worker_memory_bytes'] - resident)
        return pending

    def acquire(self, slot, stopped, *, deadline=None):
        while not stopped.is_set():
            if deadline is not None and time.monotonic() >= deadline:
                stopped.set()
                return False
            with self._lock:
                if self.plan['policy'] == 'fixed-legacy':
                    admitted = True
                else:
                    resources = hardware_resources()
                    admitted = (len(self.active) < self.plan['pool_workers'] and
                        resources['available_memory_bytes'] - self.plan['reserve_memory_bytes'] - self._pending_memory()
                        >= self.plan['estimated_worker_memory_bytes'])
                if admitted:
                    self.active[slot] = None
                    return True
            stopped.wait(.5)
        return False

    def release(self, slot):
        with self._lock:
            self.active.pop(slot, None)

    def cleanup_on_exit(self, function):
        @wraps(function)
        def guarded(slot):
            try:
                return function(slot)
            finally:
                self.release(slot)
        return guarded
