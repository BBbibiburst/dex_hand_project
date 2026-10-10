"""Unit contracts for adaptive Push motion."""

from source.grasping.push_transfer import PushTransferConfig, _push_advance


def test_push_advance_decelerates_near_target() -> None:
    config = PushTransferConfig()

    assert _push_advance(0.10, config) == config.push_increment
    assert config.minimum_push_increment <= _push_advance(0.04, config) < config.push_increment
    assert _push_advance(config.target_stop_radius, config) == 0.0


def test_push_config_rejects_invalid_adaptive_step_range() -> None:
    config = PushTransferConfig(minimum_push_increment=0.005)

    try:
        config.validate()
    except ValueError as exc:
        assert "minimum_push_increment" in str(exc)
    else:
        raise AssertionError("invalid adaptive step range was accepted")


def test_push_approach_uses_a_faster_free_space_probe() -> None:
    config = PushTransferConfig()

    assert config.approach_probe_increment > config.contact_probe_increment
    assert config.approach_increment_steps < config.increment_steps
    old_probe_frames = (
        int(config.initial_maximum_contact_probe / config.contact_probe_increment)
        * config.increment_steps
    )
    new_probe_frames = (
        int(config.initial_maximum_contact_probe / config.approach_probe_increment)
        * config.approach_increment_steps
    )
    assert new_probe_frames <= old_probe_frames // 4
