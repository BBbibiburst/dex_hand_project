from pathlib import Path

import pytest

from apps.run_production_canary import run


def test_canary_driver_cannot_use_formal_or_unbounded_target():
    with pytest.raises(ValueError, match='development output'):
        run(Path('/mnt/f/must_not_create_test'), 3600, 150, 80)
    target = Path('outputs/must_not_create_invalid_canary_driver')
    with pytest.raises(ValueError, match='Finite canary'):
        run(target, 3601, 150, 80)
    with pytest.raises(ValueError, match='Finite canary'):
        run(target, 3600, 150, 81)
    assert not target.exists()
