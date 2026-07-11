"""Vision-tactile imitation-learning policies."""

__all__ = ["DiffusionPolicy", "DiffusionPolicyConfig"]


def __getattr__(name: str):
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from source.imitation import diffusion_policy

    value = getattr(diffusion_policy, name)
    globals()[name] = value
    return value
