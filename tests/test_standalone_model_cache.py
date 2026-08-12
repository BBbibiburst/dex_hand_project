"""Regression checks for process-local evolution model reuse."""

from pathlib import Path

import numpy as np

from source.grasping.standalone_validator import (
    _compiled_direct_hold_model,
    build_cached_direct_hold_model,
)


def test_direct_hold_model_is_compiled_once_but_data_is_fresh() -> None:
    mesh = Path("assets/grippers/dex_hand/meshes/v3_base_link.STL").resolve()
    _compiled_direct_hold_model.cache_clear()
    kwargs = {
        "object_mesh": mesh,
        "mesh_center": np.zeros(3),
        "mesh_scale": 0.1,
        "hand_translation": np.asarray([0.2, 0.0, 0.0]),
        "hand_rotation_matrix": np.eye(3),
        "object_table_height": None,
        "end_effector_name": "dex_hand",
    }

    first_model, first_data = build_cached_direct_hold_model(**kwargs)
    second_model, second_data = build_cached_direct_hold_model(**kwargs)

    assert first_model is second_model
    assert first_data is not second_data
    assert _compiled_direct_hold_model.cache_info().misses == 1
    assert _compiled_direct_hold_model.cache_info().hits == 1
