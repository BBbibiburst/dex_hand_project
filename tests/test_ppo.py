import torch

from source.rl.ppo import ActorCritic, PPOConfig, RunningNormalizer


def test_running_normalizer_stays_finite() -> None:
    normalizer = RunningNormalizer(3)
    values = torch.tensor([[1.0, 2.0, 3.0], [3.0, 4.0, 5.0]])
    normalizer.update(values)
    result = normalizer(values)
    assert result.shape == values.shape
    assert torch.isfinite(result).all()


def test_actor_critic_actions_are_bounded_and_re_evaluable() -> None:
    config = PPOConfig(hidden_sizes=(32, 32), initial_std=0.4)
    model = ActorCritic(7, 3, config)
    observations = torch.randn(8, 7)
    model.normalizer.update(observations)
    actions, log_prob, value = model.act(observations)
    evaluated_log_prob, entropy, evaluated_value = model.evaluate_actions(observations, actions)
    assert actions.shape == (8, 3)
    assert torch.all(actions.abs() <= 1.0)
    assert torch.isfinite(log_prob).all()
    assert torch.isfinite(evaluated_log_prob).all()
    assert torch.isfinite(entropy).all()
    assert torch.isfinite(value).all()
    assert torch.isfinite(evaluated_value).all()
