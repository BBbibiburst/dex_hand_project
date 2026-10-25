import numpy as np
import torch

from source.imitation.dual_vision_bc import DualVisionActionChunkBC, DualVisionBCConfig
from source.imitation.vision_input import crop_normalized_rgb


def test_dual_vision_checkpoint_roundtrip():
    config = DualVisionBCConfig(state_dim=39, action_dim=9, horizon=4)
    model = DualVisionActionChunkBC(config).eval()
    global_rgb = torch.rand(2, 3, 120, 160)
    local_rgb = torch.rand(2, 3, 256, 96)
    state = torch.rand(2, 39)
    expected = model.sample(global_rgb, local_rgb, state)
    restored = DualVisionActionChunkBC.from_checkpoint(model.checkpoint()).eval()
    actual = restored.sample(global_rgb, local_rgb, state)
    assert expected.shape == (2, 4, 9)
    torch.testing.assert_close(actual, expected)


def test_fixed_roi_is_rgb_hwc_and_bounded():
    image = np.zeros((1080, 1920, 3), dtype=np.uint8)
    image[449:845, 947:1091, 1] = 255
    crop = crop_normalized_rgb(
        image,
        (947 / 1920, 449 / 1080, 1091 / 1920, 845 / 1080),
        output_width=96,
        output_height=256,
    )
    assert crop.shape == (256, 96, 3)
    assert crop.dtype == np.uint8
    assert crop[..., 1].min() == 255
