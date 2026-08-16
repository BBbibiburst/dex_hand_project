"""High-level, resumable catalogue grasp benchmark workflow."""

from importlib import import_module

from source.workflows.grasp_benchmark.config import GraspBenchmarkConfig

__all__ = ["GraspBenchmarkConfig", "run_grasp_benchmark"]


def __getattr__(name: str):
    if name != "run_grasp_benchmark":
        raise AttributeError(name)
    value = getattr(import_module("source.workflows.grasp_benchmark.runner"), name)
    globals()[name] = value
    return value
