"""Upgrade the known lease-runtime fix while retaining the existing inventory."""

from contextlib import ExitStack
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile

from apps.five_task_production import load_config
from source.data.five_task_dataset import canonical_hash
from source.data.production_bootstrap import intent_path
from source.data.production_execution_contract import execution_fingerprint
from source.data.production_index import live_dataset_processes


APPROVED_BASE = '44187c5'
RUNTIME_FILES = {'source/data/production_coordinator.py', 'source/data/production_execution_contract.py',
                 'source/data/production_liveness.py', 'source/data/production_queue.py'}


def git_hash(project, revision, name):
    return hashlib.sha256(subprocess.check_output(['git', 'show', f'{revision}:{name}'], cwd=project)).hexdigest()


def validate_change(project, before, after):
    if any(before[k] != after[k] for k in ('python', 'packages', 'shared_resource_sha256')):
        raise ValueError('Runtime environment or physical resources differ; this upgrade only permits the lease fix')
    if set(before['source_sha256']) != set(after['source_sha256']):
        raise ValueError('Source file set changed; cannot apply the restricted runtime upgrade')
    changed = [name for name, digest in after['source_sha256'].items() if before['source_sha256'][name] != digest]
    if set(changed) - RUNTIME_FILES:
        raise ValueError('Non-runtime source changes are not permitted: ' + str(changed))
    for name in changed:
        if before['source_sha256'][name] != git_hash(project, APPROVED_BASE, name):
            raise ValueError('Frozen source is not the approved previous release: ' + name)
        if after['source_sha256'][name] != git_hash(project, 'HEAD', name):
            raise ValueError('Local source differs from the checked-out release: ' + name)
    return changed


def write_json(path, value):
    from source.data.production_bootstrap import install_json
    temporary = path.with_name(path.name + '.upgrade.tmp')
    if temporary.exists():
        temporary.unlink()
    install_json(temporary, value)
    temporary.replace(path)


def upgrade(config, project):
    root = Path(config['dataset_root']).resolve()
    paths = {'intent': intent_path(root), 'contract': root / 'run_contract.json',
             'complete': root / 'bootstrap_complete.json'}
    if not root.is_dir():
        raise ValueError('Existing dataset directory is required')
    with ExitStack() as locks:
        control = Path(config['queue_database']).resolve().parent
        for path in (control / 'supervisor.lock', paths['intent'].with_suffix('.lock'), root / 'coordinator.lock'):
            stream = locks.enter_context(path.open('a'))
            try:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise ValueError('Collection is still running; stop it before upgrading') from None
        if live_dataset_processes(root):
            raise ValueError('Dataset workers or encoders are still running')
        current = execution_fingerprint(project)
        pending = root / 'runtime_upgrade_pending.json'
        if pending.exists():
            journal = json.loads(pending.read_text())
            if (journal.get('schema') != 'lease-runtime-upgrade-v1'
                    or not Path(journal['backup_directory']).resolve().is_relative_to(root / 'runtime/upgrades')):
                raise ValueError('Invalid interrupted-upgrade journal')
            if journal['new']['contract']['execution'] != current:
                raise ValueError('Interrupted upgrade targets another runtime')
            validate_change(project, journal['old']['contract']['execution'], current)
        else:
            old = {key: json.loads(path.read_text()) for key, path in paths.items()}
            intent = old['intent']
            if (intent['configuration'] != config or intent['dataset_root'] != str(root)
                    or json.loads((root / 'config.json').read_text()) != config
                    or old['contract'] != intent['run_contract']
                    or old['complete']['intent_hash'] != canonical_hash(intent)):
                raise ValueError('Existing startup/configuration proofs do not agree')
            for field, file in (('template_sha256', 'template_library'), ('registry_sha256', 'object_registry')):
                if hashlib.sha256(Path(config[file]).read_bytes()).hexdigest() != old['contract'][field]:
                    raise ValueError('Frozen template or registry changed')
            changed = validate_change(project, old['contract']['execution'], current)
            if not changed:
                return {'upgraded': False, 'reason': 'Already running the current code contract'}
            new_contract = {**old['contract'], 'execution': current}
            new_intent = {**intent, 'run_contract': new_contract}
            new = {'contract': new_contract, 'intent': new_intent,
                   'complete': {**old['complete'], 'intent_hash': canonical_hash(new_intent)}}
            backup_parent = root / 'runtime/upgrades'
            backup_parent.mkdir(parents=True, exist_ok=True)
            backup = Path(tempfile.mkdtemp(prefix='lease_fix_', dir=backup_parent))
            journal = {'schema': 'lease-runtime-upgrade-v1', 'old': old, 'new': new,
                       'changed_sources': changed, 'backup_directory': str(backup)}
            write_json(backup / 'upgrade.json', journal)
            write_json(pending, journal)
        # Recovery accepts only the recorded old/new metadata, never arbitrary changes.
        for key, path in paths.items():
            if json.loads(path.read_text()) not in (journal['old'][key], journal['new'][key]):
                raise ValueError('Unexpected metadata during upgrade: ' + str(path))
        if journal['new']['intent']['configuration'] != config:
            raise ValueError('Upgrade configuration differs')
        for key in ('contract', 'intent', 'complete'):
            write_json(paths[key], journal['new'][key])
        write_json(Path(journal['backup_directory']) / 'complete.json', {'complete': True})
        pending.unlink()
        return {'upgraded': True, 'backup': journal['backup_directory'],
                'changed_sources': journal['changed_sources'], 'inventory_modified': False,
                'episodes_modified': False, 'collection_started': False}


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(upgrade(load_config(args.config), Path.cwd()), indent=2, ensure_ascii=False))
