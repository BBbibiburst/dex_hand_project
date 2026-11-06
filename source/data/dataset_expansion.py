"""Self-contained repeated-trajectory datasets with explicit nominal accounting."""

from concurrent.futures import ThreadPoolExecutor
import json
import math
from pathlib import Path
import shutil

import numpy as np

from source.data.five_task_dataset import sha256_file
from source.data.production_parallel import worker_plan
from source.data.production_templates import load_templates


def inventory_sources(config):
    rows = load_templates(Path(config['template_library']), Path(config['object_registry']))

    def inspect(row):
        manifest = Path(row['reference_manifest'])
        path = manifest.parent / 'episode.npz'
        digest = sha256_file(path)
        with np.load(path, allow_pickle=False) as data:
            n = len(data['action'])
            if n < 2 or data['action'].ndim != 2 or not np.isfinite(data['action']).all():
                raise ValueError('Invalid existing actions: ' + str(path))
            has_time = 'timestamp' in data.files
            if has_time:
                t = data['timestamp']
                if len(t) != n or not np.isfinite(t).all() or np.any(np.diff(t) <= 0):
                    raise ValueError('Invalid recorded timestamps: ' + str(path))
                seconds = float(data['next_timestamp'][-1] - t[0]) if 'next_timestamp' in data.files else float(t[-1] - t[0])
            else:
                seconds = n / config['control_hz']
            if not math.isfinite(seconds) or seconds <= 0:
                raise ValueError('Invalid source duration')
            fields = {name: {'shape': list(data[name].shape), 'dtype': str(data[name].dtype)} for name in data.files}
        return {'source_id': row['task'] + '_' + row['asset_hash'][:12] + '_' + digest[:16],
            'task': row['task'], 'asset_hash': row['asset_hash'], 'object_id': row['object_id'],
            'array_sha256': digest, 'manifest_sha256': sha256_file(manifest),
            'origin_arrays': str(path.resolve()), 'origin_manifest': str(manifest.resolve()),
            'duration_seconds': seconds, 'frames': n, 'recorded_timestamps': has_time,
            'duration_basis': 'recorded_time_span' if has_time else 'frame_count_at_nominal_control_hz',
            'nominal_control_hz': config['control_hz'], 'fields': fields,
            'rgb_available': False,
            'note': 'Original native arrays preserved. Missing RGB/tactile are not fabricated.'}

    plan = worker_plan(config)
    with ThreadPoolExecutor(max_workers=plan['pool_workers']) as pool:
        scanned = list(pool.map(inspect, rows))
    unique = {row['source_id']: row for row in scanned}
    sources = list(unique.values())
    return {'sources': sources, 'worker_plan': plan,
        'independent_assets': len({row['asset_hash'] for row in sources}),
        'available_source_hours': sum({s['array_sha256']: s['duration_seconds'] for s in sources}.values()) / 3600,
        'all_sources_have_recorded_timestamps': all(s['recorded_timestamps'] for s in sources),
        'source_hours_note': 'Legacy sources without clocks use nominal frame_count/control_hz, not verified physical time.'}


def build_expanded_dataset(config, output, *, hours=1000., minimum_assets=100):
    output = Path(output)
    if output.exists() or not math.isfinite(hours) or hours <= 0:
        raise ValueError('Use a fresh output and positive finite sample hours')
    inventory = inventory_sources(config)
    sources = inventory['sources']
    if inventory['independent_assets'] < minimum_assets:
        raise ValueError('Insufficient actual source assets; repetitions cannot increase object count')
    tasks = config['task_hours']
    by_task = {task: [s for s in sources if s['task'] == task] for task in tasks}
    if any(not rows for rows in by_task.values()):
        raise ValueError('Existing sources must cover all requested tasks')
    unique_arrays = {s['array_sha256']: s for s in sources}
    required = sum(Path(s['origin_arrays']).stat().st_size for s in unique_arrays.values()) + 100_000_000
    parent = output.parent
    while not parent.exists():
        parent = parent.parent
    usage = shutil.disk_usage(parent)
    if usage.free - usage.total * config['disk_reserve_fraction'] < required:
        raise ValueError('Insufficient storage above disk reserve')
    output.mkdir(parents=True)
    (output/'sources').mkdir()
    (output/'metadata').mkdir()
    for row in sources:
        row['arrays'] = 'sources/' + row['array_sha256'] + '.npz'
        row['manifest'] = 'metadata/' + row['manifest_sha256'] + '.json'
        for origin, relative, digest in ((row['origin_arrays'], row['arrays'], row['array_sha256']),
                                         (row['origin_manifest'], row['manifest'], row['manifest_sha256'])):
            destination = output/relative
            if not destination.exists():
                shutil.copyfile(origin, destination)
                if sha256_file(destination) != digest:
                    raise ValueError('Copied source hash differs')
    (output/'sources.json').write_text(json.dumps(inventory, indent=2)+'\n')
    accumulated = {task: 0. for task in tasks}
    target = {task: hours*3600*weight/sum(tasks.values()) for task, weight in tasks.items()}
    used = set()
    seen_assets = set()
    sample_count = 0
    with (output/'samples.jsonl').open('x') as stream:
        def emit(source):
            nonlocal sample_count
            record = {'sample_id': sample_count, 'source_id': source['source_id'],
                'task': source['task'], 'duration_seconds': source['duration_seconds'],
                'reused_source': source['source_id'] in used, 'new_simulation': False}
            stream.write(json.dumps(record, separators=(',', ':'))+'\n')
            sample_count += 1
            accumulated[source['task']] += source['duration_seconds']
            seen_assets.add(source['asset_hash'])
            used.add(source['source_id'])

        for source in sources:
            if source['asset_hash'] not in seen_assets:
                emit(source)
        for task, candidates in by_task.items():
            index = 0
            while accumulated[task] < target[task]:
                emit(candidates[index % len(candidates)])
                index += 1
    summary = {'schema': 'expanded-trajectory-view-v1', 'dataset_kind': 'reused_existing_native_trajectories',
        'expanded_sample_hours': sum(accumulated.values())/3600,
        'task_expanded_hours': {task: seconds/3600 for task, seconds in accumulated.items()},
        'target_sample_hours': hours, 'samples': sample_count, 'independent_assets': len(seen_assets),
        'available_source_hours': inventory['available_source_hours'], 'new_simulation_hours': 0,
        'all_sources_have_recorded_timestamps': inventory['all_sources_have_recorded_timestamps'],
        'rgb_included': False, 'native_fields_preserved': True,
        'warning': 'Expanded/repeated sample duration is NOT new or independent simulation duration. Legacy clocks/modalities can be missing. This is not the old full-modal production acceptance.',
        'sources_sha256': sha256_file(output/'sources.json'), 'samples_sha256': sha256_file(output/'samples.jsonl')}
    (output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    (output/'README.md').write_text('# Existing-trajectory expansion\n\n'
        'This dataset repeats existing native trajectories via samples.jsonl. sources/ contains the actual NPZ data once. '
        'Use source.data.dataset_expansion.ExpandedTrajectoryDataset to read logical samples. '
        'See summary.json for expanded hours, original nominal source hours and object identities. '
        'No new simulation is claimed. Legacy sources often have no timestamps or RGB; absent modalities are not fabricated.\n')
    return summary


class ExpandedTrajectoryDataset:
    def __init__(self, root):
        self.root = Path(root)
        inventory = json.loads((self.root/'sources.json').read_text())
        self.sources = {s['source_id']: s for s in inventory['sources']}
        self.samples = [json.loads(line) for line in (self.root/'samples.jsonl').read_text().splitlines()]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        source = self.sources[sample['source_id']]
        with np.load(self.root/source['arrays'], allow_pickle=False) as data:
            arrays = {name: data[name] for name in data.files}
        return {**sample, 'asset_hash': source['asset_hash'], 'object_id': source['object_id'], 'arrays': arrays}
