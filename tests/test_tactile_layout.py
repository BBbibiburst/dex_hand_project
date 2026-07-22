"""Regression tests for tactile fitting layout metadata."""

from __future__ import annotations

import pytest

from source.sensors.tactile.fitting.layout import dex_hand_patch_info


def test_dex_hand_patch_info_is_read_only() -> None:
    patch_info = dex_hand_patch_info()

    assert patch_info["skin_0_0_p"] == (4, 8, "segment")
    with pytest.raises(TypeError):
        patch_info["skin_0_0_p"] = (1, 1, "segment")  # type: ignore[index]
