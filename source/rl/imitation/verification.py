"""Public verification states for expert admission and final evaluation."""

from __future__ import annotations

EXPERT_PROFILE = "expert"
FINAL_PROFILE = "final"

EXPERT_POOL_VALID = "EXPERT_POOL_VALID"
EXPERT_POOL_REJECTED = "EXPERT_POOL_REJECTED"
FINAL_VERIFIED = "FINAL_VERIFIED"
FINAL_REJECTED = "FINAL_REJECTED"


def verification_status(profile: str, success: bool) -> str:
    """Map one strict-replay profile result to an unambiguous public state."""

    if profile == EXPERT_PROFILE:
        return EXPERT_POOL_VALID if success else EXPERT_POOL_REJECTED
    if profile == FINAL_PROFILE:
        return FINAL_VERIFIED if success else FINAL_REJECTED
    raise ValueError(f"Unknown verification profile: {profile!r}.")
