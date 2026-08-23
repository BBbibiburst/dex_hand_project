"""Authoritative C MuJoCo verification for generated grasp demonstrations."""

from source.verification.profiles import FINAL_REJECTED, FINAL_VERIFIED
from source.verification.strict_replay import (
    StrictReplayResult,
    load_replay_controls,
    strict_replay_manifest,
)

__all__ = [
    "FINAL_REJECTED",
    "FINAL_VERIFIED",
    "StrictReplayResult",
    "load_replay_controls",
    "strict_replay_manifest",
]
