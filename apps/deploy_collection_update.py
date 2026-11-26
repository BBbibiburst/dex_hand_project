"""Approve only the pinned Git quality update and relocate frozen mount metadata.

Run on the original server, with collectors stopped. Never rebuild the inventory.
"""
import argparse
from contextlib import ExitStack
import fcntl
import json
import os
from pathlib import Path
import sqlite3
import sys
import time

from apps.apply_collection_quality_patch import read, sha, canonical, write_json


def approve_execution(old, current, manifest):
    expected = dict(old['source_sha256'])
    for name, entry in manifest['files'].items():
        if expected.get(name) not in (entry['before'], entry['after']):
            raise ValueError('Unrecognized frozen patch base: ' + name)
        expected[name] = entry['after']
    if current != {**old, 'source_sha256': expected}:
        raise ValueError('Unapproved source, Python, package or physical-resource difference')


def relocate_pointer(value, oldroot, root, database):
    if value['original_database'] != str(database) or value['dataset_root'] != str(oldroot):
        raise ValueError('Inventory pointer identity differs')
    oldbase = oldroot.parent / (oldroot.name + '_index_recovery')
    selected = Path(value['active_database'])
    relative = selected.relative_to(oldbase)
    target = root.parent / (root.name + '_index_recovery') / relative
    if not target.is_file() or not target.resolve().is_relative_to((root.parent / (root.name + '_index_recovery')).resolve()):
        raise ValueError('Relocated recovery database missing or unsafe')
    return {**value, 'dataset_root': str(root), 'active_database': str(target)}


def restore(journal):
    for name, item in journal['metadata'].items():
        p = Path(name)
        if read(p) not in (item['old'], item['new']):
            raise ValueError('Concurrent metadata change; manual recovery required: ' + name)
    for name, item in journal['metadata'].items():
        write_json(Path(name), item['old'])


def deploy(project, root, manifest, *, fault=False):
    from source.data.production_execution_contract import execution_fingerprint
    from source.data.production_index import pointer_path, live_dataset_processes
    from source.data.production_preflight import mount_info
    if not root.is_dir():
        raise ValueError('Existing dataset directory missing; refusing to create one')
    # Require a separately mounted volume, never an empty directory on the OS disk.
    volume = mount_info(root)
    if not volume or Path(volume['mountpoint']).resolve() == Path('/'):
        raise ValueError('Dataset is not on a separately mounted external volume')
    if os.statvfs(root).f_blocks * os.statvfs(root).f_frsize < 19_000_000_000_000:
        raise ValueError('External volume is below the expected 19 TB capacity class')
    config = read(root / 'config.json')
    database = (project / config['queue_database']).resolve()
    control = database.parent
    if not control.is_dir():
        raise ValueError('Original server control directory is missing; use the original checkout')
    intentpath = root.parent / ('.' + root.name + '.bootstrap.json')
    pending = control / 'git_deployment_pending.json'
    with ExitStack() as stack:
        for p in (control/'autorecovery.manager.lock', control/'supervisor.lock',
                  intentpath.with_suffix('.lock'), root/'coordinator.lock'):
            stream = stack.enter_context(p.open('a'))
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if live_dataset_processes(root) or live_dataset_processes(Path(config['dataset_root'])):
            raise ValueError('Stop all collection workers before deployment')
        if pending.exists():
            restore(read(pending)); pending.unlink()
        config = read(root/'config.json')
        contract = read(root/'run_contract.json')
        intent = read(intentpath)
        complete = read(root/'bootstrap_complete.json')
        oldroot = Path(config['dataset_root'])
        if (intent['configuration'] != config or intent['run_contract'] != contract
                or complete['intent_hash'] != canonical(intent)
                or intent['dataset_root'] != str(oldroot)
                or intent['database'] != str(database) or root.name != oldroot.name):
            raise ValueError('Original frozen proofs/project/database identity differs')
        for field, key in [('template_sha256','template_library'), ('registry_sha256','object_registry')]:
            if sha(project/config[key]) != contract[field]:
                raise ValueError('Frozen registry/templates differ')
        current = execution_fingerprint(project)
        approve_execution(contract['execution'], current, manifest)
        for name, item in manifest['files'].items():
            if sha(project/name) != item['after']:
                raise ValueError('Pinned Git patch is incomplete: ' + name)
        updated_config = {**config, 'dataset_root': str(root)}
        updated_contract = {**contract, 'execution': current}
        updated_intent = {**intent, 'configuration': updated_config,
                          'dataset_root': str(root), 'run_contract': updated_contract}
        metadata = {}
        def change(path, old, new):
            metadata[str(path)] = {'old':old, 'new':new}
        change(root/'config.json', config, updated_config)
        change(root/'run_contract.json', contract, updated_contract)
        change(intentpath, intent, updated_intent)
        change(root/'bootstrap_complete.json', complete, {**complete,'intent_hash':canonical(updated_intent)})
        selected = database
        pointer = pointer_path(database)
        if pointer.exists():
            before = read(pointer)
            after = relocate_pointer(before, oldroot, root, database)
            change(pointer, before, after)
            selected = Path(after['active_database'])
        for folder in ('autorecovery_quality_v3', 'autorecovery_v2'):
            statefile = control/folder/'state.json'
            if statefile.exists():
                stream = stack.enter_context((statefile.parent/'watchdog.lock').open('a'))
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
                state = read(statefile)
                if state.get('dataset_root') != str(oldroot):
                    raise ValueError('Watchdog state belongs to another dataset')
                change(statefile, state, {**state,'dataset_root':str(root)})
        if not selected.is_file():
            raise ValueError('Original queue missing; never create a replacement')
        backup = control/'git_deployments'/str(time.time_ns())
        backup.mkdir(parents=True)
        with sqlite3.connect(selected.as_uri()+'?mode=ro', uri=True) as db:
            if db.execute('PRAGMA quick_check').fetchall() != [('ok',)]:
                raise ValueError('Queue integrity check failed')
            with sqlite3.connect(backup/'queue_before.sqlite') as target:
                db.backup(target)
        journal = {'schema':'collection-git-deployment-v1','metadata':metadata,'backup':str(backup)}
        write_json(backup/'migration.json', journal)
        write_json(pending, journal)
        try:
            for name, item in metadata.items():
                write_json(Path(name), item['new'])
            if fault:
                raise RuntimeError('Injected migration failure')
            # Create only the new schema/reference ledger; preserve all progress and quotas.
            from source.data.production_queue import ProductionQueue
            from source.data.production_artifacts import verify_episode
            queue = ProductionQueue(selected)
            before = queue.status(count_failures=config['count_failed_episodes'])
            refs = []
            legacy = root.parent/'dex_hand_dataset'
            if (legacy/'episodes').is_dir():
                for directory in sorted((legacy/'episodes').iterdir()):
                    if not directory.is_dir() or directory.is_symlink():
                        raise ValueError('Unexpected legacy episode entry')
                    verified = verify_episode(directory)
                    meta = read(directory/'episode.json')
                    refs.append({'batch_id':str(oldroot.parent/'dex_hand_dataset'), 'job_id':meta['job_id'],
                                 'content_hash':verified['content_hash'],'lineage_hash':verified['lineage_hash']})
            queue.import_external_references(refs)
            if before != queue.status(count_failures=config['count_failed_episodes']):
                raise ValueError('Quota statistics changed')
            deployment = {'project':str(project),'dataset_root':str(root),
                          'python':sys.executable,'backup':str(backup),'started_collection':False}
            write_json(control/'collection_deployment.json',deployment)
            pending.unlink()
            return deployment
        except BaseException:
            restore(journal)
            pending.unlink(missing_ok=True)
            raise


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project',type=Path,required=True)
    p.add_argument('--dataset-root',type=Path,required=True)
    a=p.parse_args(); project=a.project.resolve(); os.chdir(project)
    print(json.dumps(deploy(project,a.dataset_root.resolve(),read(project/'configs/collection_quality_patch_20261126.json')),ensure_ascii=False,indent=2))
