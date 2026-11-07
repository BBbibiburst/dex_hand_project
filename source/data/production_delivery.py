"""Bounded, hash-verifiable delivery candidate; never launches production."""

import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import zipfile

from source.data.five_task_dataset import sha256_file
from source.data.production_execution_contract import execution_fingerprint, application_dependencies
from source.data.production_templates import load_templates


MAXIMUM_BYTES = 2 * 1024 ** 3


def safe_name(name):
    path = PurePosixPath(name)
    if path.is_absolute() or '..' in path.parts or '\\' in name or not path.parts or ':' in path.parts[0]:
        raise ValueError('Unsafe delivery member path')
    return path


def verify_archive(archive):
    with zipfile.ZipFile(archive) as bundle:
        names = bundle.namelist()
        if len(names) != len(set(names)):
            raise ValueError('Duplicate delivery entries')
        for item in bundle.infolist():
            safe_name(item.filename)
            if (item.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError('Delivery archive must not contain symlinks')
        if sum(i.file_size for i in bundle.infolist()) > MAXIMUM_BYTES:
            raise ValueError('Delivery exceeds uncompressed budget')
        if bundle.getinfo('DELIVERY_MANIFEST.json').file_size > 16 * 1024 ** 2:
            raise ValueError('Delivery manifest is too large')
        manifest = json.loads(bundle.read('DELIVERY_MANIFEST.json'))
        if manifest.get('schema') != 'five-task-delivery-v1':
            raise ValueError('Unsupported delivery schema')
        if set(names) != set(manifest['files']) | {'DELIVERY_MANIFEST.json'}:
            raise ValueError('Delivery has missing or unlisted files')
        for name, expected in manifest['files'].items():
            digest = hashlib.sha256()
            size = 0
            with bundle.open(name) as stream:
                for block in iter(lambda: stream.read(1024 ** 2), b''):
                    digest.update(block)
                    size += len(block)
            if digest.hexdigest() != expected['sha256'] or size != expected['bytes']:
                raise ValueError(f'Delivery file differs: {name}')
    return {'passed': True, 'files': len(manifest['files']),
            'archive_sha256': sha256_file(archive), 'bytes': archive.stat().st_size,
            'production_started': False, 'data_target_certified': False}


def build_delivery(project, output):
    project, output = Path(project).resolve(), Path(output).resolve()
    if Path.cwd().resolve() != project or output.exists() or not output.is_relative_to(project / 'outputs'):
        raise ValueError('Run from project root with a fresh outputs delivery directory')
    sources = {}

    def add(path):
        path = Path(path)
        path = path if path.is_absolute() else project / path
        logical = Path(os.path.abspath(path))
        if not logical.is_relative_to(project) or not path.resolve().is_relative_to(project):
            raise ValueError(f'Delivery source escapes project: {path}')
        if not path.is_file():
            raise FileNotFoundError(path)
        sources[str(logical.relative_to(project))] = path

    fingerprint = execution_fingerprint(project)
    for name in (*fingerprint['source_sha256'], *fingerprint['shared_resource_sha256']):
        add(name)
    management = ['run_production_canary.py', 'probe_production_compatibility.py',
                  'verify_production_inventory.py', 'build_production_catalogue.py',
                  'build_p31_registry.py', 'build_production_delivery.py',
                  'expand_existing_dataset.py', 'restore_production_inputs.py']
    for path in application_dependencies(project, [project / 'apps' / name for name in management]):
        add(path)
    for path in (project / 'tests').glob('test_production_*.py'):
        add(path)
    for name in ('pyproject.toml', 'README.md', 'assets/__init__.py', 'docs/five_task_production_user_guide.md',
                 'docs/five_task_episode_schema.md', 'docs/fast_collection.md'):
        add(name)
    seen_libraries = set()
    rows = []
    config_paths = set((project / 'configs').glob('five_task_production*.json'))
    config_paths.update(project / 'configs' / name for name in ('five_task_collection.json', 'five_task_speed.json')
                        if (project / 'configs' / name).is_file())
    for config_path in sorted(config_paths):
        add(config_path)
        config = json.loads(config_path.read_text())
        add(config['template_library'])
        add(config['object_registry'])
        key = (config['template_library'], config['object_registry'])
        if key not in seen_libraries:
            rows.extend(load_templates(Path(key[0]), Path(key[1])))
            seen_libraries.add(key)
    visited = set()

    def source_manifest(path):
        path = Path(path)
        if path.is_absolute() and 'outputs' in path.parts:
            local = project.joinpath(*path.parts[path.parts.index('outputs'):])
            if local.is_file():
                path = local
        if not path.is_absolute():
            path = project / path
        if path.resolve() in visited:
            return
        visited.add(path.resolve())
        add(path)
        record = json.loads(path.read_text())
        arrays = record.get('arrays', 'episode.npz')
        if isinstance(arrays, str) and (path.parent / arrays).is_file():
            add(path.parent / arrays)
        for section in (record, record.get('metadata', {})):
            for field in ('source_manifest', 'template_manifest'):
                if section.get(field):
                    candidate = Path(section[field])
                    if candidate.is_absolute() and 'outputs' in candidate.parts:
                        candidate = Path(*candidate.parts[candidate.parts.index('outputs'):])
                    candidate = candidate if candidate.is_absolute() else project / candidate
                    if candidate.is_file():
                        source_manifest(candidate)

    for row in rows:
        for field in ('reference_manifest', 'handle_manifest', 'lift_source_manifest'):
            if row.get(field):
                source_manifest(row[field])
        for dependency in row.get('dependencies', []):
            add(dependency['path'])
        summary = Path(row['reference_manifest']).parent.parent / 'summary.json'
        if summary.is_file():
            add(summary)
        asset = Path(row['asset_path'])
        for path in ([asset] if asset.is_file() else asset.rglob('*')):
            if path.is_file():
                add(path)
    total = sum(path.stat().st_size for path in sources.values())
    disk = shutil.disk_usage(project / 'outputs')
    if total > MAXIMUM_BYTES - 10 * 1024 ** 2 or disk.free - total * 2 < .2 * disk.total:
        raise ValueError('Delivery exceeds 2GiB bound or disk reserve')
    environment = {'python': fingerprint['python'], 'packages': {name: version(name) for name in (
        'mujoco', 'numpy', 'scipy', 'gymnasium', 'imageio-ffmpeg', 'psutil', 'trimesh', 'Pillow')}}
    generated = {'RUNTIME_ENVIRONMENT.json': json.dumps(environment, indent=2).encode(),
                 'START_HERE.md': (
        '# Five-task production delivery candidate\n\n'
        'Extract into a NEW directory. Never overwrite the original project.\n'
        'Install Python >=3.10 and project production dependencies: pip install -e ".[production]".\n'
        'For faster collection read docs/fast_collection.md; set DEX_DATASET_ROOT to the server data disk, '
        'then preflight with --config configs/five_task_speed.json (automatic CPU/RAM worker admission).\n'
        'The older HD/strict workflow is documented in docs/five_task_production_user_guide.md.\n'
        'Contains 101 historical candidate assets, not 100 verified current assets or 1000 hours.\n'
        'This archive does not authorize production. F-volume, development acceptance and remaining tests still apply.\n'
        'No simulation is launched by building or verifying this archive.\n').encode()}
    manifest = {'schema': 'five-task-delivery-v1', 'status': 'DEVELOPMENT_CANDIDATE_NOT_FINAL_CERTIFICATION',
                'independent_candidate_assets': len({r['asset_hash'] for r in rows}),
                'data_target_certified': False, 'files': {}}
    for name, path in sorted(sources.items()):
        manifest['files'][name] = {'bytes': path.stat().st_size, 'sha256': sha256_file(path)}
    for name, data in generated.items():
        manifest['files'][name] = {'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}
    output.mkdir(parents=True)
    archive = output / 'five_task_production.zip'
    with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=1) as bundle:
        for name, path in sorted(sources.items()):
            bundle.write(path, name)
        for name, data in generated.items():
            bundle.writestr(name, data)
        bundle.writestr('DELIVERY_MANIFEST.json', json.dumps(manifest, indent=2))
    result = verify_archive(archive)
    result.update(status=manifest['status'], independent_candidate_assets=manifest['independent_candidate_assets'],
                  uncompressed_input_bytes=total)
    (output / 'verification.json').write_text(json.dumps(result, indent=2))
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    return result
