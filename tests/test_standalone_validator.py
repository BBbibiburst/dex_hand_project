from types import SimpleNamespace

import numpy as np
import pytest

from source.grasping import mujoco_safety


def _data() -> SimpleNamespace:
    return SimpleNamespace(
        qpos=np.zeros(2),
        qvel=np.zeros(2),
        qacc=np.zeros(2),
    )


def test_checked_step_rejects_mujoco_warning(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mujoco_safety.mujoco, "mj_step", lambda model, data: None)

    with pytest.raises(FloatingPointError, match="QACC at DOF 31"):
        mujoco_safety.checked_mj_step(
            object(),
            _data(),
            ["Nan, Inf or huge value in QACC at DOF 31"],
            phase="hold",
            step=7,
        )


def test_checked_step_rejects_nonfinite_state(monkeypatch: pytest.MonkeyPatch) -> None:
    data = _data()
    data.qacc[0] = np.nan
    monkeypatch.setattr(mujoco_safety.mujoco, "mj_step", lambda model, data: None)

    with pytest.raises(FloatingPointError, match="non-finite or unbounded"):
        mujoco_safety.checked_mj_step(
            object(),
            data,
            [],
            phase="settle",
            step=1,
        )
