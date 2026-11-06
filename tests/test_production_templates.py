import json
from pathlib import Path

import pytest

from source.data.production_templates import candidate, load_templates, validate_candidate_sampling
from source.data.production_conditions import sampling_contract


def test_pilot_library_tracks_real_identity_without_inflating_asset_count():
    rows = load_templates(Path('configs/five_task_templates.json'),
                         Path('outputs/five_task_p31/object_registry.json'))
    assert len(rows) == 5
    assert len({row['asset_hash'] for row in rows}) == 4
    pick = next(row for row in rows if row['task'] == 'pick_place')
    job = candidate(pick, 32000, 0)
    assert job['conditions']['dx_m'] == job['conditions']['dy_m'] == 0
    assert job['conditions']['target_dx_m'] != 0


def test_template_hash_drift_rejected(tmp_path):
    payload = json.loads(Path('configs/five_task_templates.json').read_text())
    payload['templates'][0]['reference_manifest_sha256'] = '0' * 64
    path = tmp_path / 'templates.json'
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match='changed'):
        load_templates(path, Path('outputs/five_task_p31/object_registry.json'))


def test_payload_binds_ranges_and_enabled_fields():
    row = {'task': 'lift', 'base_seed': 7, 'perturbation_fields': ['dx_m']}
    contract = sampling_contract()
    contract['tasks']['lift']['dx_m'] = [.001, .001]
    payload = candidate(row, 42, 0, contract)
    assert payload['conditions'] == {'dx_m': .001, 'dy_m': 0., 'yaw_deg': 0.}
    validate_candidate_sampling(payload)
    payload['conditions']['dx_m'] = .002
    with pytest.raises(ValueError, match='immutable sampling contract'):
        validate_candidate_sampling(payload)
