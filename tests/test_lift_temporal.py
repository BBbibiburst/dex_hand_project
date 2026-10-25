import torch

from source.imitation.lift_temporal import LiftTemporalConfig, LiftTemporalController
from apps.build_lift_temporal_dataset import _pad_history


def test_lift_temporal_shapes():
    for kind in ("single_mlp", "history_mlp", "gru"):
        config = LiftTemporalConfig(kind=kind)
        model = LiftTemporalController(config)
        state = torch.zeros(2, config.history, config.state_dim)
        action = torch.zeros(2, config.history, 3)
        assert model(state, action).shape == (2, config.horizon, 3)


def test_history_padding_repeats_first_available_value():
    values = [torch.tensor([float(index)]).numpy() for index in range(3)]
    result = _pad_history(values, 1, 4)
    assert result[:, 0].tolist() == [0.0, 0.0, 0.0, 1.0]
