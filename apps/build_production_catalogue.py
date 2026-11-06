"""Freeze historically successful task templates and their exact dependencies."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np

from source.data.production_templates import load_templates
from source.grasping.contracts import DemonstrationEpisode
from tools.grasping.transfer_lift_to_pick_place import load_lift_source


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def local_path(value):
    path = Path(value)
    if path.is_file():
        return path
    if 'outputs' in path.parts:
        return Path(*path.parts[path.parts.index('outputs'):])
    return path


def validate_source(path):
    payload = json.loads(path.read_text())
    if not payload.get('success'):
        raise ValueError('No historical success')
    arrays_path = path.parent / payload.get('arrays', 'episode.npz')
    with np.load(arrays_path, allow_pickle=False) as arrays:
        n = len(arrays['qpos'])
        for key in ('qpos', 'qvel', 'ctrl', 'action', 'object_position', 'object_quaternion_wxyz', 'stage', 'task_success'):
            if len(arrays[key]) != n or not np.isfinite(arrays[key]).all():
                raise ValueError(f'Invalid source field {key}')
        if arrays['action'].shape != (n, 13):
            raise ValueError('Source action is not 13D')
    return payload, digest(arrays_path)


def dependencies(path, seen=None):
    seen = set() if seen is None else seen
    path = local_path(str(path))
    if path.resolve() in seen:
        return []
    seen.add(path.resolve())
    payload = json.loads(path.read_text())
    files = [{'path': str(path), 'sha256': digest(path)}]
    arrays = payload.get('arrays')
    if isinstance(arrays, str):
        array_path = path.parent / arrays
        files.append({'path': str(array_path), 'sha256': digest(array_path)})
    for section in (payload, payload.get('metadata', {})):
        for key in ('source_manifest', 'template_manifest'):
            if section.get(key):
                source = local_path(section[key])
                if source.is_file():
                    files.extend(dependencies(source, seen))
    return files


def successful_lineage(path):
    summary = json.loads((path.parent.parent / 'summary.json').read_text())
    lineage = next((local_path(item['source_manifest']) for item in reversed(summary.get('attempts', []))
                    if item.get('success') and item.get('source_manifest')), None)
    if lineage is None:
        raise ValueError('No successful Lift lineage')
    return lineage


def build(output, registry_path, base_library):
    if output.exists():
        raise FileExistsError(output)
    registry = json.loads(registry_path.read_text())
    assets = {row['object_id']: row for row in registry['objects']}
    rows = list(json.loads(base_library.read_text())['templates'])
    for row in rows:
        row['validation_status'] = 'CURRENT_SINGLE_CONDITION_SUCCESS'
        path = Path(row['reference_manifest'])
        row['object_id'] = json.loads(path.read_text())['object_id']
        row['dependencies'] = dependencies(path)
        if row['task'] in ('pick_place', 'stack', 'push'):
            lineage = successful_lineage(path)
            row.update(lift_source_manifest=str(lineage), lift_source_manifest_sha256=digest(lineage))
            row['dependencies'].extend(dependencies(lineage))
        if row.get('handle_manifest'):
            row['dependencies'].extend(dependencies(Path(row['handle_manifest'])))
    excluded = []
    seen = {(row['task'], str(Path(row['reference_manifest']).resolve())) for row in rows}

    def add(task, path):
        try:
            key = (task, str(path.resolve()))
            if key in seen:
                return
            source, array_hash = validate_source(path)
            object_id = source['object_id']
            row = {'template_id': task + '_' + digest(path)[:20], 'task': task,
                   'object_id': object_id, 'base_seed': int(source['seed']),
                   'reference_manifest': str(path), 'reference_manifest_sha256': digest(path),
                   'source_arrays_sha256': array_hash, 'asset_hash': assets[object_id]['asset_hash'],
                   'validation_status': 'HISTORICAL_SUCCESS_REQUIRES_CURRENT_EXECUTION'}
            if task == 'lift':
                DemonstrationEpisode.load(path)
                for field in ('object_initial_position', 'object_initial_quaternion_wxyz', 'execution_config'):
                    if field not in source['metadata']:
                        raise ValueError(f'No reinitialization field {field}')
                row['perturbation_fields'] = ['dx_m', 'dy_m', 'yaw_deg']
            else:
                lineage = successful_lineage(path)
                lifted = load_lift_source(lineage)
                if lifted.object_id != object_id:
                    raise ValueError('Lift lineage object differs')
                if task == 'stack' and lifted.metadata.get('ppo_source_manifest'):
                    raise ValueError('Stack RL lineage requires a newly executed held-state source')
                row.update(lift_source_manifest=str(lineage), lift_source_manifest_sha256=digest(lineage))
                row['dependencies'] = dependencies(lineage)
                row['perturbation_fields'] = (['target_dx_m', 'target_dy_m'] if task in ('pick_place', 'stack')
                                              else ['dx_m', 'dy_m', 'yaw_deg', 'push_direction_delta_deg'])
            row.setdefault('dependencies', []).extend(dependencies(path))
            rows.append(row)
            seen.add(key)
        except (OSError, KeyError, ValueError, TypeError) as error:
            excluded.append({'task': task, 'manifest': str(path), 'reason': f'{type(error).__name__}: {error}'})

    for index in sorted(Path('outputs/dex_hand_top100_v2/lattice').glob('*/index.json')):
        candidates = [row for row in json.loads(index.read_text()).get('templates', []) if row.get('success')]
        for row in sorted(candidates, key=lambda row: -row.get('source_lift', 0))[:3]:
            add('lift', local_path(row['manifest']))
    for task, root in [('pick_place', 'outputs/pick_place_top100_final'),
                       ('stack', 'outputs/stack_task_valid_v24'), ('push', 'outputs/push_top100_v30')]:
        for path in sorted(Path(root).glob('*/best_trajectory/manifest.json')):
            add(task, path)
    output.mkdir(parents=True)
    library = output / 'templates.json'
    library.write_text(json.dumps({'schema': 'five-task-template-library-v1', 'templates': rows}, indent=2) + '\n')
    # Validate all selected identities against live asset contents once.
    checked = load_templates(library, registry_path)
    matrix = {}
    for row in checked:
        asset = row['asset_hash']
        matrix.setdefault(asset, {'asset_hash': asset, 'tasks': {}, 'formal_episodes': 0})
        matrix[asset]['tasks'].setdefault(row['task'], []).append(row['template_id'])
    summary = {'templates': len(rows), 'independent_candidate_assets': len(matrix),
               'by_task': dict(Counter(row['task'] for row in rows)),
               'excluded_candidates': len(excluded), 'current_100_object_coverage_proven': False,
               'registry_sha256': digest(registry_path),
               'note': 'Historical compatibility candidates, not current successful dataset coverage'}
    for name, value in [('summary.json', summary), ('task_object_matrix.json', matrix),
                        ('excluded_templates.json', excluded)]:
        (output / name).write_text(json.dumps(value, indent=2) + '\n')
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--registry', type=Path, default=Path('outputs/five_task_p31/object_registry.json'))
    parser.add_argument('--base-library', type=Path, default=Path('configs/five_task_templates.json'))
    args = parser.parse_args()
    print(json.dumps(build(args.output, args.registry, args.base_library), indent=2))
