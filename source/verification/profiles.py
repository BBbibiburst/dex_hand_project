"""Public verification states for generated demonstrations."""

from __future__ import annotations

FINAL_PROFILE = "final"

FINAL_VERIFIED = "FINAL_VERIFIED"
FINAL_REJECTED = "FINAL_REJECTED"


def verification_status(profile: str, success: bool) -> str:
    """Map one strict-replay profile result to an unambiguous public state."""

    if profile == FINAL_PROFILE:
        return FINAL_VERIFIED if success else FINAL_REJECTED
    raise ValueError(f"Unknown verification profile: {profile!r}.")
