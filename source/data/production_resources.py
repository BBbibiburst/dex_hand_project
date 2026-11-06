"""System memory water level for bounded worker admission and runtime checks."""


def retained_file_bytes(root, *, maximum_scans=3):
    """Bounded telemetry scans; retry disappearing paths during atomic rename.

    Permission/I/O errors remain errors. This is a sampled byte count, not an
    atomic filesystem snapshot or an alternative to file-based acceptance.
    """
    import os
    import stat

    for _ in range(maximum_scans):
        retained = 0
        pending = [root]
        disappeared = False
        while pending:
            directory = pending.pop()
            try:
                with os.scandir(directory) as entries:
                    for entry in entries:
                        try:
                            item = entry.stat(follow_symlinks=False)
                        except FileNotFoundError:
                            disappeared = True
                            continue
                        if stat.S_ISDIR(item.st_mode):
                            pending.append(entry.path)
                        elif stat.S_ISREG(item.st_mode):
                            retained += item.st_size
            except FileNotFoundError:
                if directory == root:
                    raise
                disappeared = True
        if not disappeared:
            return retained
    raise RuntimeError('Retained-byte scan remained unstable during directory rename')

def memory_status(config):
    import psutil

    minimum = config.get('minimum_available_memory_bytes', 1024 ** 3)
    if isinstance(minimum, bool) or not isinstance(minimum, int) or minimum < 256 * 1024 ** 2:
        raise ValueError('Memory reserve must be an integer of at least 256 MiB')
    memory = psutil.virtual_memory()
    return {'ready': memory.available >= minimum, 'available_bytes': memory.available,
            'total_bytes': memory.total, 'required_available_bytes': minimum,
            'system_percent': memory.percent}


def canary_storage_status(root, device, reserve=.2):
    import os

    info = os.statvfs(root)
    total = info.f_blocks * info.f_frsize
    available = info.f_bavail * info.f_frsize
    retained = retained_file_bytes(root)
    errors = []
    if root.stat().st_dev != device:
        errors.append('Development test filesystem identity changed')
    if available - total * reserve < 12_000_000_000:
        errors.append('Insufficient test workspace above disk reserve')
    if retained >= 4_000_000_000:
        errors.append('Development retained data reached the 4GB stop water level')
    return {'ready': not errors, 'errors': errors, 'development_volume_only': True,
            'retained_bytes': retained, 'total_bytes': total, 'available_bytes': available,
            'reserve_fraction': reserve, 'retained_stop_bytes': 4_000_000_000}
