import numpy as np

from source.sensors.tactile.dex_hand import create_dex_hand_tactile_sensor
from source.sensors.tactile.signal_processing import TactileSignalProcessor


def test_read_raw_does_not_advance_signal_processor_state():
    sensor = create_dex_hand_tactile_sensor(signal_processor={"lowpass_alpha": 0.4})
    sensor.signal_processor.reset(sensor.total_taxels)
    before = sensor.signal_processor._filtered.copy()
    assert before.shape == (712,)
    np.testing.assert_array_equal(sensor.signal_processor._filtered, before)


def test_cloned_processor_reconstructs_same_sequence():
    config = {"deadzone": 0.002, "saturation": 1.0, "nonlinear_exponent": 0.7,
              "lowpass_alpha": 0.45, "crosstalk": 0.12, "normalize": True}
    sensor = create_dex_hand_tactile_sensor(signal_processor=config)
    clone = TactileSignalProcessor(config)
    sensor.signal_processor.reset(sensor.total_taxels)
    clone.reset(sensor.total_taxels)
    rng = np.random.default_rng(8)
    for _ in range(5):
        raw = rng.random(712, dtype=np.float32)
        expected = sensor.signal_processor.process(raw, sensor.patches)
        actual = clone.process(raw, sensor.patches)
        np.testing.assert_array_equal(actual, expected)
