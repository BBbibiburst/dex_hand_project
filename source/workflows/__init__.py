"""High-level, reusable project workflows."""

from importlib import import_module

__all__ = ["GraspBenchmarkConfig", "run_grasp_benchmark"]


def __getattr__(name: str):
    if name not in __all__:
        raise AttributeError(name)
    value = getattr(import_module("source.workflows.grasp_benchmark"), name)
    globals()[name] = value
    return value
