import threading
import json
import sys

import pytest

from source.data.production_profile import PhaseProfile, summarize_profiles


def test_phase_profile_retains_exception_spans_and_concurrent_counts():
    profile = PhaseProfile()
    with pytest.raises(ValueError):
        with profile.measure('failed_work'):
            raise ValueError('artificial failure')
    assert profile.snapshot()['phases']['failed_work']['seconds'] >= 0
    threads = [threading.Thread(target=lambda: [profile.add('async', .001) for _ in range(100)]) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert profile.snapshot()['phases']['async']['calls'] == 200
    snapshot = profile.snapshot()
    snapshot['phases']['async']['calls'] = -1
    assert profile.snapshot()['phases']['async']['calls'] == 200


def test_profile_summary_keeps_inclusive_scopes_separate_and_reports_corruption(tmp_path):
    profile = PhaseProfile()
    profile.add('inclusive_work', 1.)
    attempt = tmp_path / 'work/fixture/attempt_001'
    attempt.mkdir(parents=True)
    (attempt / 'worker_result.json').write_text(json.dumps({'wall_profile': profile.snapshot()}))
    (attempt / 'result.json').write_text(json.dumps({'wall_profile': profile.snapshot()}))
    result = summarize_profiles(tmp_path)
    assert result['scopes']['worker']['phases']['inclusive_work']['seconds'] == 1.
    assert result['scopes']['online_execution']['phases']['inclusive_work']['seconds'] == 1.
    assert 'total_seconds' not in result
    (attempt / 'result.json').write_text('{')
    assert summarize_profiles(tmp_path)['errors']


def test_pilot_cli_passes_explicit_wall_budget(monkeypatch, capsys):
    import apps.five_task_production as cli
    import source.data.production_coordinator as coordinator

    observed = {}

    def pilot(*args, **kwargs):
        observed.update(kwargs)
        return {'artificial_cli_test': True}

    monkeypatch.setattr(coordinator, 'pilot', pilot)
    monkeypatch.setattr(sys, 'argv', ['five_task_production.py', 'pilot', '--pilot-root',
        'outputs/artificial_cli_not_created', '--max-episodes', '1', '--wall-budget-seconds', '240'])
    cli.main()
    assert observed['wall_budget'] == 240.
    assert json.loads(capsys.readouterr().out)['artificial_cli_test']
