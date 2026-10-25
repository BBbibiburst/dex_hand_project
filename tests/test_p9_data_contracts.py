import numpy as np
import pytest

from apps.collect_bc_hold_snapshots import snapshot_payload
from apps.search_grasp_preparation import preparation_validated


def _trace():
    return {
        "bc_qpos": np.zeros((10, 26)),
        "bc_qvel": np.zeros((10, 25)),
        "bc_ctrl": np.zeros((10, 13)),
        "bc_qacc_warmstart": np.zeros((10, 25)),
        "bc_wrist_quat_wxyz": np.zeros((10, 4)),
        "bc_tactile": np.zeros((10, 712)),
        "bc_arm_qpos": np.zeros((10, 7)),
        "bc_arm_ctrl": np.zeros((10, 7)),
    }


def test_complete_snapshot_contract_and_history_boundary():
    snapshot = snapshot_payload(_trace(), 3)
    assert snapshot["qpos"].shape == (26,)
    assert snapshot["qacc_warmstart"].shape == (25,)
    assert snapshot["tactile_history"].shape == (4, 712)


def test_incomplete_snapshot_is_rejected():
    trace = _trace()
    del trace["bc_ctrl"]
    with pytest.raises(ValueError, match="bc_ctrl"):
        snapshot_payload(trace, 3)


def test_preparation_label_requires_load_transfer_and_contacts():
    assert preparation_validated(0.0031, 0.6, 0.9, 0.9)
    assert not preparation_validated(0.0031, 0.4, 1.0, 1.0)
    assert not preparation_validated(0.001, 0.8, 1.0, 1.0)
