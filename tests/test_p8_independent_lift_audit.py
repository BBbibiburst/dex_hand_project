from pathlib import Path

from apps.audit_independent_lift_demonstrations import audit


def test_empty_audit_fails_independence_gate(tmp_path: Path):
    result = audit(tmp_path)
    assert result["attempts"] == 0
    assert result["strict_accepted"] == 0
    assert result["independence_gate"] is False
