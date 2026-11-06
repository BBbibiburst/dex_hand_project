import copy
import json
from pathlib import Path

import pytest

from apps.five_task_production import load_config


def test_default_plan_is_1000h_and_success_only():
    config = load_config(Path("configs/five_task_production.json"))
    assert sum(config["task_hours"].values()) == 1000
    assert not config["count_failed_episodes"]


def test_explicit_outcome_example_preserves_all_five_200h_targets():
    config = load_config(Path('configs/five_task_production_outcome_example.json'))
    assert sum(row['success'] for row in config['task_outcome_hours'].values()) == 900
    assert sum(row['failure'] for row in config['task_outcome_hours'].values()) == 100
    assert config['coverage_requires_success']


@pytest.mark.parametrize("key,value", [("workers", 4), ("rgb_fps", 5),
                                      ("tactile_dimension", 0), ("disk_reserve_fraction", .1),
                                      ('integrity_audit_interval_attempts', 0),
                                      ('integrity_audit_interval_attempts', True)])
def test_contract_cannot_be_silently_reduced(tmp_path, key, value):
    config = copy.deepcopy(json.loads(Path("configs/five_task_production.json").read_text()))
    config[key] = value
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError):
        load_config(path)


def test_recovery_policy_must_be_explicit_boolean_if_present(tmp_path):
    config = json.loads(Path('configs/five_task_production.json').read_text())
    config['auto_rebuild_inventory'] = 'yes'
    path = tmp_path / 'bad_recovery.json'
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match='boolean'):
        load_config(path)
