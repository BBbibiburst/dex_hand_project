import json

import numpy as np
import pytest

from source.data.dataset_expansion import build_expanded_dataset, ExpandedTrajectoryDataset


def fixtures(tmp_path, monkeypatch):
    """Artificial format fixtures, not actual five-task simulation evidence."""
    rows = []
    for index, task in enumerate(('lift', 'pick_place', 'stack', 'push', 'nut_assembly')):
        directory = tmp_path/'inputs'/task
        directory.mkdir(parents=True)
        np.savez_compressed(directory/'episode.npz', action=np.full((3, 13), index, np.float32),
                            qpos=np.zeros((3, 26), np.float32))
        manifest = directory/'manifest.json'
        manifest.write_text(json.dumps({'success': True, 'object_id': f'fixture:{index%2}'}))
        rows.append({'reference_manifest': str(manifest), 'task': task,
                     'object_id': f'fixture:{index%2}', 'asset_hash': str(index%2)*64})
    monkeypatch.setattr('source.data.dataset_expansion.load_templates', lambda *_: rows)
    monkeypatch.setattr('source.data.dataset_expansion.worker_plan', lambda _: {'pool_workers': 1})
    return {'template_library': 'artificial', 'object_registry': 'artificial', 'control_hz': 20,
            'task_hours': {r['task']: 200 for r in rows}, 'disk_reserve_fraction': .2}


def test_reuse_increases_sample_hours_not_assets_or_new_simulation(tmp_path, monkeypatch):
    config = fixtures(tmp_path, monkeypatch)
    output = tmp_path/'expanded'
    result = build_expanded_dataset(config, output, hours=.001, minimum_assets=2)
    assert result['expanded_sample_hours'] >= .001
    assert result['independent_assets'] == 2
    assert result['new_simulation_hours'] == 0
    assert result['all_sources_have_recorded_timestamps'] is False
    dataset = ExpandedTrajectoryDataset(output)
    assert len(dataset) > 5
    assert any(row['reused_source'] for row in dataset.samples)
    assert dataset[0]['arrays']['action'].shape == (3, 13)
    assert len(list((output/'sources').glob('*.npz'))) == 5
    assert all(value > 0 for value in result['task_expanded_hours'].values())


def test_expansion_cannot_create_missing_assets_and_never_overwrites(tmp_path, monkeypatch):
    config = fixtures(tmp_path, monkeypatch)
    output = tmp_path/'not_created'
    with pytest.raises(ValueError, match='Insufficient actual source assets'):
        build_expanded_dataset(config, output, hours=.001, minimum_assets=3)
    assert not output.exists()
    output.mkdir()
    with pytest.raises(ValueError, match='fresh'):
        build_expanded_dataset(config, output, hours=.001, minimum_assets=2)
