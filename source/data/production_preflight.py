"""Read-only checks for the intended F volume and runtime dependencies."""

import importlib.util
import os
from pathlib import Path
import subprocess


def mount_info(path):
    resolved = path.resolve()
    matches = []
    for line in Path('/proc/self/mountinfo').read_text().splitlines():
        parts = line.split()
        split = parts.index('-')
        mount = Path(parts[4].replace('\\040', ' '))
        if resolved.is_relative_to(mount):
            matches.append((len(mount.parts), {'mountpoint': str(mount),
                            'filesystem': parts[split + 1], 'source': parts[split + 2],
                            'options': parts[split + 3]}))
    if not matches:
        raise ValueError('No mount identity found')
    return max(matches, key=lambda item: item[0])[1]


def check_volume(dataset_root, reserve=.2):
    try:
        return _check_volume(dataset_root, reserve)
    except OSError as error:
        return {'ready': False, 'errors': [f'F volume does not exist or is unavailable: {error}']}


def _check_volume(dataset_root, reserve=.2):
    mount_root = Path('/mnt/f')
    target = dataset_root.resolve()
    errors = []
    if not target.is_relative_to(mount_root):
        return {'ready': False, 'errors': ['Formal dataset must remain under /mnt/f']}
    if not mount_root.is_dir():
        return {'ready': False, 'errors': ['F mount directory does not exist']}
    info = mount_info(mount_root)
    stat = os.statvfs(mount_root)
    total = stat.f_blocks * stat.f_frsize
    available = stat.f_bavail * stat.f_frsize
    safe = available - int(total * reserve)
    if info['mountpoint'] != '/mnt/f' or os.stat(mount_root).st_dev == os.stat('/').st_dev:
        errors.append('F path falls back to WSL root filesystem')
    identity = (info['source'] + ' ' + info['options']).upper()
    if not (info['source'].upper().startswith('F:') or 'PATH=F:' in identity):
        errors.append('Mount identity does not identify Windows F volume')
    if total < 19_000_000_000_000:
        errors.append('Volume is below the expected 19TB capacity class')
    workspace = 6_000_000_000  # Two bounded workers, model/video copies and transaction staging.
    if safe < workspace:
        errors.append('Available space after reserve is below the two-worker workspace budget')
    return {**info, 'dataset_root': str(target), 'total_bytes': total,
            'available_bytes': available, 'safe_new_bytes': safe, 'reserve_fraction': reserve,
            'minimum_workspace_bytes': workspace,
            'total_TB': total / 1e12, 'total_TiB': total / 2**40,
            'ready': not errors, 'errors': errors}


def check_dataset_volume(config):
    try:
        return _check_dataset_volume(config)
    except OSError as error:
        return {'ready': False, 'errors': [f'Configured data volume is unavailable: {error}']}


def _check_dataset_volume(config):
    if config.get('storage_policy') != 'configured-volume-v1':
        return check_volume(Path(config['dataset_root']), config['disk_reserve_fraction'])
    target = Path(config['dataset_root']).expanduser().resolve()
    # Still reject the known Windows-drive fallback; Linux server data paths
    # can use their configured local mount without pretending to be Windows F.
    if target.is_relative_to(Path('/mnt/f')):
        return check_volume(target, config['disk_reserve_fraction'])
    existing = target
    while not existing.exists():
        if existing == existing.parent:
            raise ValueError('No existing dataset parent')
        existing = existing.parent
    info = mount_info(existing)
    usage = os.statvfs(existing)
    total = usage.f_blocks * usage.f_frsize
    available = usage.f_bavail * usage.f_frsize
    safe = available - int(total * config['disk_reserve_fraction'])
    expected = config.get('expected_volume_source')
    errors = []
    if expected and info['source'] != expected:
        errors.append('Configured volume source differs from expected_volume_source')
    if safe < 6_000_000_000:
        errors.append('Insufficient workspace above configured disk reserve')
    return {**info, 'dataset_root': str(target), 'total_bytes': total, 'available_bytes': available,
            'safe_new_bytes': safe, 'reserve_fraction': config['disk_reserve_fraction'],
            'total_TB': total / 1e12, 'total_TiB': total / 2**40, 'ready': not errors, 'errors': errors}


def preflight(config):
    from source.data.production_resources import memory_status
    from source.data.production_parallel import worker_plan

    dependencies = {name: importlib.util.find_spec(name) is not None
                    for name in ('mujoco', 'numpy', 'scipy', 'imageio_ffmpeg', 'psutil')}
    codec = {'ready': False}
    if dependencies['imageio_ffmpeg']:
        import imageio_ffmpeg

        executable = imageio_ffmpeg.get_ffmpeg_exe()
        try:
            result = subprocess.run([executable, '-hide_banner', '-encoders'], timeout=10,
                                    check=True, capture_output=True, text=True)
            codec = {'executable': executable, 'ready': 'libx264' in result.stdout}
        except (OSError, subprocess.SubprocessError) as error:
            codec['error'] = str(error)
    volume = check_dataset_volume(config)
    memory = memory_status(config) if dependencies['psutil'] else {'ready': False, 'error': 'psutil not installed'}
    parallel = worker_plan(config) if dependencies['psutil'] else {'initial_admitted_workers': 0}
    return {'volume': volume, 'dependencies': dependencies, 'encoder': codec, 'memory': memory,
            'worker_plan': parallel,
            'infrastructure_ready': volume['ready'] and all(dependencies.values()) and codec['ready'] and memory['ready'] and parallel['initial_admitted_workers'] > 0,
            'production_ready': False,
            'production_gate': 'Current asset coverage, endurance and final storage/volume validation remain incomplete'}
