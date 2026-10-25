from apps.build_p17_state_conditioned_dataset import SPLITS, _boundary
from apps.train_cartesian_action_chunk_bc import build_parser

import numpy as np


def test_p17_split_roots_are_disjoint():
    roots = [root for values in SPLITS.values() for root in values]
    assert len(roots) == len(set(roots)) == 5
    assert len(SPLITS["test"]) == 1


def test_p17_boundary_uses_preframe_plus_bc_prefix():
    ctrl = np.zeros((15, 13), dtype=np.float32)
    ctrl[12, :7] = 0.02
    ctrl[12, 7:] = 0.0001
    result = _boundary({"ctrl": ctrl}, query_step=10)
    assert np.isclose(result["arm_l2_rad"], np.sqrt(7) * 0.02)
    assert np.isclose(result["hand_l2_m"], np.sqrt(6) * 0.0001)


def test_cartesian_trainer_accepts_explicit_root_split():
    actions = build_parser()._actions
    assert any(action.dest == "split_file" for action in actions)
