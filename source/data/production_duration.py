"""Observed candidate duration for monitoring, never for quota acceptance."""

import json

import numpy as np

from source.data.production_execution_contract import file_digest


def candidate_duration(case):
    """Count a contract-checked candidate, including later-rejected duplicates.

    Missing/partial legacy outputs remain unknown. A reused recovery output is
    a candidate seen again, not newly executed simulation time.
    """
    if case is None:
        return None
    try:
        record = json.loads((case / 'result.json').read_text())
        arrays = case / 'modalities.npz'
        if (not isinstance(record, dict) or record.get('contract_passed') is not True
                or record.get('modalities_sha256') != file_digest(arrays)):
            return None
        with np.load(arrays, allow_pickle=False) as data:
            timestamp = data['timestamp']
            following = data['next_timestamp']
            if (timestamp.ndim != 1 or len(timestamp) < 2 or following.shape != timestamp.shape
                    or not np.isfinite(timestamp).all() or not np.isfinite(following).all()
                    or not np.allclose(np.diff(timestamp), .05, rtol=0, atol=1e-7)
                    or not np.allclose(following - timestamp, .05, rtol=0, atol=1e-7)):
                return None
            return float(following[-1] - timestamp[0])
    except (OSError, ValueError, KeyError, TypeError):
        return None
