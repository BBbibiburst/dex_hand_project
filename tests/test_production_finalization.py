import json
from pathlib import Path
import sys

import pytest

from apps.five_task_production import load_config
from source.data.production_finalization import finalize_dataset
from source.data.production_queue import ProductionQueue


def test_sql_completion_without_files_is_rejected_by_automatic_report(tmp_path, monkeypatch):
    import source.data.production_coordinator as coordinator

    config = load_config(Path('configs/five_task_production.json'))
    root = tmp_path / 'empty_dataset'
    original = ProductionQueue.quota_status

    def bogus_quota(self, *args, **kwargs):
        return {**original(self, *args, **kwargs), 'complete': True, 'needed_tasks': []}

    monkeypatch.setattr(ProductionQueue, 'quota_status', bogus_quota)
    monkeypatch.setattr(coordinator, 'load_templates', lambda *_: [])
    monkeypatch.setattr(coordinator.subprocess, 'Popen', lambda *a, **kw: (_ for _ in ()).throw(
        AssertionError('No worker should start')))
    result = coordinator._coordinate(config, root, 1, 42)
    assert result['quota_complete']
    assert result['termination'] == 'QUOTA_COMPLETE_FILE_REJECTED'
    acceptance = result['final_file_acceptance']
    saved = json.loads((Path(acceptance['report_directory']) / 'report.json').read_text())
    assert saved['acceptance']['unique_seconds'] == 0
    assert not saved['acceptance']['formal_dataset_accepted']
    assert not Path(acceptance['report_directory']).is_relative_to(root)
    assert json.loads((root / 'summary.json').read_text())['termination'] == result['termination']


def test_report_exception_retains_first_error_and_rejects_acceptance(tmp_path, monkeypatch):
    import source.data.production_finalization as module

    root = tmp_path / 'dataset'
    root.mkdir()
    monkeypatch.setattr(module, 'build_report', lambda _: (_ for _ in ()).throw(ValueError('bad video')))
    result = finalize_dataset({}, root, tmp_path / 'queue.sqlite')
    assert result['state'] == 'REPORT_FAILED'
    assert not result['formal_dataset_accepted']
    assert json.loads(Path(result['failure_record']).read_text())['error'] == 'ValueError: bad video'


def test_cli_returns_nonzero_when_final_file_acceptance_fails(monkeypatch, capsys):
    import apps.five_task_production as cli
    import source.data.production_coordinator as coordinator

    monkeypatch.setattr(coordinator, 'production', lambda *args, **kwargs: {
        'termination': 'QUOTA_COMPLETE_FILE_REJECTED', 'quota_complete': True})
    monkeypatch.setattr(sys, 'argv', ['five_task_production.py', 'run'])
    with pytest.raises(SystemExit) as stopped:
        cli.main()
    assert stopped.value.code == 2
    assert json.loads(capsys.readouterr().out)['termination'] == 'QUOTA_COMPLETE_FILE_REJECTED'
