from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch

from apps.collect_generated_lerobot import recorded_pre_action_indices
from apps.generate_verified_lift_demonstrations import (
    _explicit_source_manifests,
    classify_failure,
    perturbations,
)
from apps.train_diffusion import split_episode_indices
from apps.trace_action_chunk_bc import execution_prefix
from apps.diagnose_hold_cross_control import _longest_true
from apps.analyze_tactile_failure_warning import _groups
from source.sensors.tactile.dex_hand import SimpleBoxTactileSensor
from source.imitation.observation import real_observable_state, state_layout
from source.imitation.action_chunk_bc import ActionChunkBC, ActionChunkBCConfig
from source.imitation.cartesian_action import (
    POSE12,
    XYZ9,
    CartesianActionContract,
    left_rotation_delta,
    quat_from_rotvec,
)
from source.imitation.cartesian_residual import (
    CartesianResidualConfig,
    CartesianResidualMLP,
    tactile_region_features,
)
from source.geometry import quat_multiply
from source.imitation.diffusion_policy import (
    DiffusionPolicy,
    DiffusionPolicyConfig,
    diffusion_betas,
)


class _Dataset:
    def episode_indices(self):
        return {0: [0, 1, 2], 1: [3, 4], 2: [5, 6], 3: [7]}


def test_episode_split_has_no_frame_leakage():
    train, val, train_episodes, val_episodes = split_episode_indices(
        _Dataset(), val_fraction=0.25, seed=7
    )
    assert set(train_episodes).isdisjoint(val_episodes)
    assert set(train).isdisjoint(val)
    assert sorted(train + val) == list(range(8))


def test_generated_export_pairs_position_control_with_pre_action_state():
    # Two setup states precede three strict-replay controls.  The matching
    # actuator target is stored after each action, while policy observations
    # must come from the preceding rows.
    controls = np.asarray([[10.0, 11.0], [20.0, 21.0], [30.0, 31.0]], np.float32)
    recorded = {
        "ctrl": np.asarray(
            [[0.0, 1.0], [2.0, 3.0], [10.0, 11.0], [20.0, 21.0], [30.0, 31.0]],
            np.float32,
        ),
        # These are deliberately Cartesian commands and must never be used as
        # low-level position actions merely because both happen to be 2-D.
        "action": np.full((5, 2), 999.0, np.float32),
    }
    assert recorded_pre_action_indices(recorded, controls).tolist() == [1, 2, 3]


def test_real_observable_state_excludes_uncontrolled_coordinates():
    arm = SimpleNamespace(actuator_ids=np.asarray([1, 3], dtype=np.int32))
    hand = SimpleNamespace(actuator_ids=np.asarray([4], dtype=np.int32))
    controller = SimpleNamespace(
        arm_controller=arm,
        hand_controller=hand,
        actuator_names=("arm_a", "arm_b", "hand_a"),
    )
    env = SimpleNamespace(
        controller=controller,
        data=SimpleNamespace(
            actuator_length=np.arange(6, dtype=np.float64),
            actuator_velocity=np.arange(10, 16, dtype=np.float64),
            ctrl=np.arange(20, 26, dtype=np.float64),
        ),
    )
    assert real_observable_state(env).tolist() == [1, 3, 4, 11, 13, 14, 21, 23, 24]
    assert len(state_layout(env)) == 9


def test_perturbation_schedule_is_bounded_and_distinct():
    values = perturbations(5, 0.005, 5.0)
    assert len(values) == len(set(values)) == 5
    assert all(abs(x) <= 0.005 and abs(y) <= 0.005 and abs(yaw) <= 5.0 for x, y, yaw in values)


def test_explicit_grasp_sources_preserve_order_and_provenance(tmp_path, monkeypatch):
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    first.write_text("{}", encoding="utf-8")
    second.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        "apps.generate_verified_lift_demonstrations.DemonstrationEpisode.load",
        lambda _path: SimpleNamespace(object_id="ycb:004_sugar_box", success=True),
    )
    sources = _explicit_source_manifests(
        [second, first, second],
        "ycb:004_sugar_box",
    )
    assert sources == (second.resolve(), first.resolve())


def test_generation_failure_categories_are_explicit():
    assert classify_failure("pregrasp IK residual is too large", None).startswith("IK_")
    assert classify_failure("object did not satisfy the lift success criterion", None) == (
        "LIFT_FAILURE"
    )
    assert classify_failure("FINAL_REJECTED", {"success": False}) == ("STRICT_REPLAY_INSTABILITY")


def test_vision_state_policy_accepts_empty_tactile():
    policy = DiffusionPolicy(
        DiffusionPolicyConfig(
            state_dim=6, tactile_dim=0, action_dim=3, horizon=2, diffusion_steps=2, feature_dim=8
        )
    )
    loss = policy.loss(
        image=torch.zeros(1, 3, 16, 16),
        tactile=torch.zeros(1, 0),
        state=torch.zeros(1, 6),
        action=torch.zeros(1, 2, 3),
    )
    assert torch.isfinite(loss)


def test_reduced_step_ddim_sampling_preserves_action_shape():
    policy = DiffusionPolicy(
        DiffusionPolicyConfig(
            state_dim=6, tactile_dim=0, action_dim=3, horizon=2, diffusion_steps=8, feature_dim=8
        )
    )
    action = policy.sample(
        torch.zeros(1, 3, 16, 16),
        torch.zeros(1, 0),
        torch.zeros(1, 6),
        inference_steps=3,
    )
    assert action.shape == (1, 2, 3)
    assert torch.isfinite(action).all()


def test_short_cosine_schedule_reaches_gaussian_terminal_distribution():
    cosine = diffusion_betas(10, "cosine")
    legacy_linear = diffusion_betas(10, "linear")
    assert torch.cumprod(1.0 - cosine, dim=0)[-1] < 1e-3
    assert torch.cumprod(1.0 - legacy_linear, dim=0)[-1] > 0.8


def test_deterministic_bc_preserves_action_chunk_contract_and_roundtrip():
    policy = ActionChunkBC(ActionChunkBCConfig(state_dim=6, action_dim=3, horizon=4, feature_dim=8))
    policy.set_normalization(
        torch.arange(6, dtype=torch.float32),
        torch.ones(6),
        torch.tensor([1.0, 2.0, 3.0]),
        torch.tensor([0.5, 1.0, 2.0]),
    )
    image = torch.zeros(2, 3, 16, 16)
    state = torch.zeros(2, 6)
    action = policy.sample(image, state)
    assert action.shape == (2, 4, 3)
    assert torch.isfinite(action).all()
    normalized = (action - policy.action_mean) / policy.action_std
    restored = normalized * policy.action_std + policy.action_mean
    assert torch.equal(restored, action)


def test_deterministic_bc_checkpoint_roundtrip():
    policy = ActionChunkBC(ActionChunkBCConfig(state_dim=6, action_dim=3, horizon=2, feature_dim=8))
    restored = ActionChunkBC.from_checkpoint(policy.checkpoint())
    assert restored.config == policy.config
    assert restored.sample(torch.zeros(1, 3, 16, 16), torch.zeros(1, 6)).shape == (1, 2, 3)


def test_action_chunk_execution_uses_each_prefix_step_once_in_order():
    chunk = np.arange(5 * 3, dtype=np.float32).reshape(5, 3)
    executed = execution_prefix(chunk, 4)
    assert np.array_equal(executed, chunk[[0, 1, 2, 3]])
    assert len({tuple(row) for row in executed}) == 4


def test_opposing_contact_survival_counts_longest_contiguous_run():
    values = np.asarray([False, True, True, False, True, True, True, False])
    assert _longest_true(values) == 3


def test_tactile_semantic_groups_cover_each_taxel_once():
    sensor = SimpleBoxTactileSensor()
    groups = _groups(sensor)
    indices = np.concatenate(list(groups.values()))
    assert len(indices) == sensor.total_taxels == 712
    assert sorted(indices.tolist()) == list(range(712))


def test_ddpm_uses_true_posterior_variance_and_zero_at_final_step():
    policy = DiffusionPolicy(
        DiffusionPolicyConfig(
            state_dim=2,
            tactile_dim=0,
            action_dim=1,
            horizon=1,
            diffusion_steps=8,
            feature_dim=8,
            noise_schedule="cosine",
        )
    )
    expected = policy.betas[1:] * (1.0 - policy.alpha_bars[:-1]) / (1.0 - policy.alpha_bars[1:])
    assert policy.posterior_variance[0] == 0
    assert torch.allclose(policy.posterior_variance[1:], expected)
    assert torch.all(policy.posterior_variance[1:] < policy.betas[1:])


def test_exact_noise_reconstructs_clean_action_at_timestep_zero():
    policy = DiffusionPolicy(
        DiffusionPolicyConfig(
            state_dim=2,
            tactile_dim=0,
            action_dim=1,
            horizon=2,
            diffusion_steps=8,
            feature_dim=8,
            noise_schedule="cosine",
            clip_sample=5.0,
        )
    )
    clean = torch.tensor([[[0.25], [-0.5]]])
    noise = torch.tensor([[[0.2], [0.3]]])
    alpha_bar = policy.alpha_bars[0]
    noisy = alpha_bar.sqrt() * clean + (1.0 - alpha_bar).sqrt() * noise
    reconstructed = policy._predict_clean(noisy, noise, 0)
    assert torch.allclose(reconstructed, clean, atol=1e-6)


def test_clean_prediction_clipping_is_explicit_and_bounded():
    policy = DiffusionPolicy(
        DiffusionPolicyConfig(
            state_dim=2,
            tactile_dim=0,
            action_dim=1,
            horizon=1,
            diffusion_steps=4,
            feature_dim=8,
            noise_schedule="cosine",
            clip_sample=2.0,
        )
    )
    reconstructed = policy._predict_clean(torch.full((1, 1, 1), 100.0), torch.zeros(1, 1, 1), 3)
    assert reconstructed.item() == 2.0


def test_cartesian_action_contract_dimensions_and_shared_anchor():
    assert CartesianActionContract(XYZ9).action_dim == 9
    assert CartesianActionContract(POSE12).action_dim == 12
    assert "shared" in CartesianActionContract(XYZ9).chunk_anchor


def test_left_rotation_vector_round_trip():
    anchor = quat_from_rotvec(np.asarray([0.2, -0.1, 0.05]))
    delta = np.asarray([-0.03, 0.04, 0.02])
    target = quat_multiply(quat_from_rotvec(delta), anchor)
    np.testing.assert_allclose(left_rotation_delta(anchor, target), delta, atol=1e-7)


def test_cartesian_residual_is_bounded_and_chunk_shaped():
    model = CartesianResidualMLP(
        CartesianResidualConfig(input_dim=5, horizon=4, maximum_residual_m=0.003)
    )
    output = model(torch.randn(2, 5))
    assert output.shape == (2, 4, 3)
    assert float(output.detach().abs().max()) <= 0.0030001


def test_tactile_region_features_use_all_semantic_groups():
    sensor = SimpleBoxTactileSensor()
    groups = _groups(sensor)
    values = np.ones(sensor.total_taxels, dtype=np.float32)
    features = tactile_region_features(values, groups)
    assert features.shape == (8,)
    assert features[0] == sensor.total_taxels
    assert features[1] == len(groups["digit_4"])
    assert features[2] == sum(len(groups[f"digit_{index}"]) for index in range(4))
