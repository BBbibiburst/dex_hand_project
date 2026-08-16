"""Shared dataset contracts and recorders.

This package is intentionally neutral: teleoperation, scripted collection, and
imitation learning may all depend on it without depending on each other.
"""

from importlib import import_module

_EXPORTS = {
    "LeRobotEpisodeRecorder": ("source.data.lerobot_recorder", "LeRobotEpisodeRecorder"),
    "ACTION_KEY": ("source.data.lerobot_schema", "ACTION_KEY"),
    "AGENTVIEW_IMAGE_KEY": ("source.data.lerobot_schema", "AGENTVIEW_IMAGE_KEY"),
    "STATE_KEY": ("source.data.lerobot_schema", "STATE_KEY"),
    "TACTILE_KEY": ("source.data.lerobot_schema", "TACTILE_KEY"),
}
__all__ = list(_EXPORTS)


def __getattr__(name: str):
    try:
        module_name, attribute = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(name) from exc
    value = getattr(import_module(module_name), attribute)
    globals()[name] = value
    return value
