"""Cross-module configuration semantics."""

from types import SimpleNamespace

from source.cli import robot_config


def test_robot_config_preserves_tactile_setting_without_cli_override(monkeypatch) -> None:
    monkeypatch.setattr(
        robot_config,
        "load_robot_config",
        lambda path: {"enable_tactile_sensors": False},
    )
    args = SimpleNamespace(robot_config=None, no_tactile=False)

    assert robot_config.load_configured_robot(args)["enable_tactile_sensors"] is False


def test_no_tactile_cli_option_is_an_explicit_override(monkeypatch) -> None:
    monkeypatch.setattr(
        robot_config,
        "load_robot_config",
        lambda path: {"enable_tactile_sensors": True},
    )
    args = SimpleNamespace(robot_config=None, no_tactile=True)

    assert robot_config.load_configured_robot(args)["enable_tactile_sensors"] is False
