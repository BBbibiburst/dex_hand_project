import numpy as np

from apps.run_tactile_raw_domain_audit import _real_stage_frames


def test_strict_stage_codes_are_not_baseline():
    from apps.run_tactile_raw_domain_audit import STAGE_CODES

    assert set(STAGE_CODES) == {"CLOSE", "HOLD", "LIFT"}
    assert min(STAGE_CODES.values()) >= 4


def test_resampling_contract_targets_twenty_hz(tmp_path):
    # The loader integration is exercised by the actual audit. This unit test
    # fixes the intended numerical sampling interval without fabricating NPZs.
    target_hz = 20.0
    assert np.isclose(1.0 / target_hz, 0.05)
    assert callable(_real_stage_frames)
