from pathlib import Path

from tools.grasping.transfer_lift_to_pick_place import project_source_path


def test_local_frozen_copy_wins_over_existing_absolute_historical_path(tmp_path, monkeypatch):
    old = tmp_path / 'old/outputs/episode/manifest.json'
    old.parent.mkdir(parents=True)
    old.write_text('old data must not be consulted')
    project = tmp_path / 'new'
    local = project / 'outputs/episode/manifest.json'
    local.parent.mkdir(parents=True)
    local.write_text('packaged frozen copy')
    monkeypatch.chdir(project)
    assert project_source_path(old) == local
    assert project_source_path(Path('outputs/episode/manifest.json')) == local


def test_template_loader_relocates_all_pinned_inputs_without_original_machine(tmp_path, monkeypatch):
    import hashlib
    import json
    from source.data.production_templates import load_templates, asset_digest

    monkeypatch.chdir(tmp_path)
    source = tmp_path/'outputs/example/manifest.json'
    source.parent.mkdir(parents=True)
    source.write_text(json.dumps({'object_id': 'fixture:object', 'success': True, 'metadata': {}}))
    asset = tmp_path/'assets/object.bin'
    asset.parent.mkdir()
    asset.write_bytes(b'artificial asset')
    asset_hash = asset_digest(asset)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    old = '/unavailable_original_project/outputs/example/manifest.json'
    library = {'templates': [{'template_id': 'fixture', 'task': 'lift', 'object_id': 'fixture:object',
        'perturbation_fields': [], 'reference_manifest': old, 'reference_manifest_sha256': digest,
        'lift_source_manifest': old, 'lift_source_manifest_sha256': digest, 'asset_hash': asset_hash,
        'dependencies': [{'path': old, 'sha256': digest}]}]}
    path = tmp_path/'library.json'
    path.write_text(json.dumps(library))
    registry = tmp_path/'registry.json'
    registry.write_text(json.dumps({'objects': [
        {'object_id': 'fixture:object', 'asset_hash': asset_hash, 'asset_path': 'assets/object.bin'}]}))
    loaded = load_templates(path, registry)[0]
    assert loaded['reference_manifest'] == str(source)
    assert loaded['lift_source_manifest'] == str(source)
    assert loaded['dependencies'][0]['path'] == str(source)
    assert json.loads(path.read_text()) == library
