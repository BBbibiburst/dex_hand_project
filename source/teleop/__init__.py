"""Human teleoperation and demonstration-recording utilities."""

from importlib import import_module

_EXPORT_MODULES = {
    "GloveSample": "source.teleop.devices",
    "ViveSample": "source.teleop.devices",
    "MockStretchGlove": "source.teleop.devices",
    "MockViveTracker": "source.teleop.devices",
    "SineStretchGlove": "source.teleop.devices",
    "SineViveTracker": "source.teleop.devices",
    "StretchGloveApiDevice": "source.teleop.devices",
    "ViveApiTracker": "source.teleop.devices",
    "TeleopMapper": "source.teleop.mapping",
    "LeRobotEpisodeRecorder": "source.teleop.lerobot_recorder",
}

__all__ = [
    "GloveSample",
    "ViveSample",
    "MockStretchGlove",
    "MockViveTracker",
    "SineStretchGlove",
    "SineViveTracker",
    "StretchGloveApiDevice",
    "ViveApiTracker",
    "TeleopMapper",
    "LeRobotEpisodeRecorder",
]


def __getattr__(name: str):
    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value
