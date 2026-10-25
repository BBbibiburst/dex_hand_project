import numpy as np
import pytest

from source.imitation.vision_input import crop_normalized_rgb, resize_rgb_uint8


def test_resize_rgb_preserves_contract_and_is_deterministic():
    image = np.arange(12 * 16 * 3, dtype=np.uint8).reshape(12, 16, 3)
    first = resize_rgb_uint8(image, width=8, height=6)
    second = resize_rgb_uint8(image, width=8, height=6)
    assert first.shape == (6, 8, 3)
    assert first.dtype == np.uint8
    np.testing.assert_array_equal(first, second)


def test_resize_identity_returns_independent_copy():
    image = np.zeros((4, 5, 3), dtype=np.uint8)
    result = resize_rgb_uint8(image, width=5, height=4)
    result[0, 0] = 255
    assert not np.array_equal(result, image)


def test_fixed_normalized_roi_has_requested_output_shape():
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    image[25:75, 50:150] = 200
    roi = crop_normalized_rgb(image, (0.25, 0.25, 0.75, 0.75), output_width=40, output_height=30)
    assert roi.shape == (30, 40, 3)
    assert int(roi.mean()) == 200


@pytest.mark.parametrize("box", [(-0.1, 0, 1, 1), (0.5, 0.5, 0.5, 1), (0, 0, 1.1, 1)])
def test_invalid_fixed_roi_is_rejected(box):
    with pytest.raises(ValueError):
        crop_normalized_rgb(
            np.zeros((10, 10, 3), dtype=np.uint8),
            box,
            output_width=8,
            output_height=8,
        )
