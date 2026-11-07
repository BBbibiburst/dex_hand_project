import copy
import json
from pathlib import Path

import pytest

import apps.upgrade_collection_runtime as module
from source.data.production_bootstrap import reserve_bootstrap, install_metadata, intent_path
from source.data.five_task_dataset import canonical_hash, sha256_file


def fixture(tmp_path, monkeypatch):
    root = tmp_path / 'data'
    control = tmp_path / 'control'
    control.mkdir()
    template = tmp_path / 'templates.json'
    registry = tmp_path / 'registry.json'
    template.write_text('{}')
    registry.write_text('{}')
    cfg = {'dataset_root': str(root), 'queue_database': str(control / 'queue.sqlite'),
           'template_library': str(template), 'object_registry': str(registry)}
    before = {'python': 'test', 'packages': {}, 'shared_resource_sha256': {},
              'source_sha256': {p: 'old-' + p for p in module.RUNTIME_FILES}}
    after = copy.deepcopy(before)
    after['source_sha256'] = {p: 'new-' + p for p in module.RUNTIME_FILES}
    contract = {'execution': before, 'template_sha256': sha256_file(template), 'registry_sha256': sha256_file(registry)}
    lock, record = reserve_bootstrap(root, control / 'queue.sqlite', cfg, contract, 32000, 24, resume=False)
    root.mkdir()
    install_metadata(root, record)
    (root / 'bootstrap_complete.json').write_text(json.dumps(
        {'schema': 'production-bootstrap-complete-v1', 'intent_hash': canonical_hash(record)}))
    (control / 'queue.sqlite').write_bytes(b'inventory untouched')
    (root / 'episodes').mkdir()
    (root / 'episodes/sentinel').write_bytes(b'episode unchanged')
    lock.close()
    monkeypatch.setattr(module, 'execution_fingerprint', lambda _: after)
    monkeypatch.setattr(module, 'live_dataset_processes', lambda _: [])
    monkeypatch.setattr(module, 'git_hash', lambda project, revision, name:
                        ('old-' if revision == module.APPROVED_BASE else 'new-') + name)
    return cfg, before, after, record


def test_restricted_upgrade_preserves_episodes_inventory_and_resume_contract(tmp_path, monkeypatch):
    cfg, before, after, old = fixture(tmp_path, monkeypatch)
    root = Path(cfg['dataset_root'])
    result = module.upgrade(cfg, tmp_path)
    assert result['upgraded']
    assert Path(cfg['queue_database']).read_bytes() == b'inventory untouched'
    assert (root / 'episodes/sentinel').read_bytes() == b'episode unchanged'
    backup = json.loads((Path(result['backup']) / 'upgrade.json').read_text())
    assert backup['old']['intent'] == old
    new_contract = {**old['run_contract'], 'execution': after}
    lock, new = reserve_bootstrap(root, Path(cfg['queue_database']), cfg, new_contract, 32000, 24, resume=True)
    lock.close()
    assert json.loads((root / 'bootstrap_complete.json').read_text())['intent_hash'] == canonical_hash(new)
    assert not module.upgrade(cfg, tmp_path)['upgraded']


def test_interrupted_upgrade_can_finish_without_resetting_data(tmp_path, monkeypatch):
    cfg, _, _, _ = fixture(tmp_path, monkeypatch)
    root = Path(cfg['dataset_root'])
    original = module.write_json

    def interrupted(path, value):
        original(path, value)
        if path == root / 'run_contract.json':
            raise RuntimeError('Injected interruption after first metadata replacement')

    monkeypatch.setattr(module, 'write_json', interrupted)
    with pytest.raises(RuntimeError, match='Injected interruption'):
        module.upgrade(cfg, tmp_path)
    assert (root / 'runtime_upgrade_pending.json').exists()
    monkeypatch.setattr(module, 'write_json', original)
    assert module.upgrade(cfg, tmp_path)['upgraded']
    assert not (root / 'runtime_upgrade_pending.json').exists()
    assert Path(cfg['queue_database']).read_bytes() == b'inventory untouched'


def test_upgrade_rejects_physical_or_unapproved_source_change(tmp_path, monkeypatch):
    cfg, before, after, _ = fixture(tmp_path, monkeypatch)
    original = intent_path(Path(cfg['dataset_root'])).read_bytes()
    after['shared_resource_sha256']['physics.xml'] = 'changed'
    with pytest.raises(ValueError, match='physical resources'):
        module.upgrade(cfg, tmp_path)
    assert intent_path(Path(cfg['dataset_root'])).read_bytes() == original
    after['shared_resource_sha256'] = {}
    after['source_sha256']['source/policy.py'] = 'new'
    before['source_sha256']['source/policy.py'] = 'old'
    # Test the validator directly because the on-disk old manifest predates this extra field.
    with pytest.raises(ValueError, match='Non-runtime'):
        module.validate_change(tmp_path, before, after)
