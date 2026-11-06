"""Validate optional per-task success/failure physical-hour targets."""

import math


def validate_outcome_hours(task_hours, outcome_hours, count_failures):
    if outcome_hours is None:
        return None
    if set(outcome_hours) != set(task_hours):
        raise ValueError('Outcome quotas must cover exactly the five configured tasks')
    result = {}
    for task, hours in task_hours.items():
        targets = outcome_hours[task]
        if set(targets) != {'success', 'failure'} or any(
                isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0
                for v in targets.values()):
            raise ValueError('Outcome quotas require finite nonnegative success/failure hours')
        if not math.isclose(sum(targets.values()), hours, rel_tol=0, abs_tol=1e-9):
            raise ValueError('Success and failure quotas must sum to each task hour target')
        if targets['failure'] > 0 and not count_failures:
            raise ValueError('Positive failure quotas require explicit count_failed_episodes=true')
        result[task] = dict(targets)
    return result
