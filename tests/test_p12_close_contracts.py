from types import SimpleNamespace

import numpy as np

from apps.diagnose_p12_close_takeover import PHASE_STEPS, TAKEOVER_END, VARIANTS, _hash
from apps.search_p12_close_sequences import alignment_candidates


def test_p12_cross_takeover_contract_is_complete_and_bounded():
    assert set(VARIANTS) == {
        "bc_xyz_bc_hand",
        "expert_xyz_bc_hand",
        "bc_xyz_expert_hand",
        "expert_xyz_expert_hand",
    }
    assert PHASE_STEPS == {"close_early": 55, "close_mid": 76}
    assert TAKEOVER_END - PHASE_STEPS["close_mid"] == 15


def test_alignment_candidates_use_current_controls_without_future_state():
    env = SimpleNamespace(
        controller=SimpleNamespace(
            arm_controller=SimpleNamespace(actuator_ids=np.arange(7)),
            hand_controller=SimpleNamespace(actuator_ids=np.arange(7, 13)),
        )
    )
    expert = np.zeros((120, 13), dtype=np.float32)
    expert[:, :7] = 1.0
    expert[63, :7] = 0.25
    expert[63, 7:] = 0.5
    snapshot = {"ctrl": np.r_[np.full(7, 0.25), np.full(6, 0.5)]}
    candidates = alignment_candidates(env, expert, snapshot, 55)
    assert len(candidates) == 4
    assert candidates[0] == ("aligned_current_orientation", 56, False)
    assert candidates[2][1] == 63
    assert candidates[3][1] == 63


def test_action_hash_is_dtype_stable():
    values = np.arange(18).reshape(2, 9)
    assert _hash(values) == _hash(values.astype(np.float32))
