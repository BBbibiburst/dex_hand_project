from source.grasping.dexevolve_mjwarp import MjWarpLifetimeConfig


def test_mjwarp_lifetime_defaults_apply_five_window_pull_scale() -> None:
    config = MjWarpLifetimeConfig()
    assert config.disturbance_steps == 80
    assert config.upward_force_ratio > 1.0
    assert config.lateral_force > 0.0
    assert config.maximum_drift > 0.0
    assert config.require_opposed_contact
