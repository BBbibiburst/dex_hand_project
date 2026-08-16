"""Gymnasium environments and task plug-ins."""

from importlib import import_module

__all__ = ["RobotGymEnv", "make_env"]


def __getattr__(name: str):
    if name not in __all__:
        raise AttributeError(name)
    value = getattr(import_module("source.envs.rl_env"), name)
    globals()[name] = value
    return value
