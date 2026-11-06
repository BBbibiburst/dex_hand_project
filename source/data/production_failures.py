"""Versioned execution-failure policy; task failure outcomes remain valid data."""

import errno


def reported_task_outcome(success, reason):
    """Classify an executor's message without claiming a physical root cause.

    A valid failure episode is not a worker exception and gets no extra retry
    budget from these diagnostic categories.
    """
    reason = reason if isinstance(reason, str) else None
    category = None
    if not success:
        text = (reason or '').lower()
        category = 'UNKNOWN_TASK_FAILURE'
        if 'ik residual is too large:' in text:
            category = 'IK_RESIDUAL_GATE'
        elif text.startswith('environment ended during '):
            category = 'ENVIRONMENT_TERMINATION'
        elif text == 'hold has no thumb-opposed skin contact':
            category = 'CONTACT_REQUIREMENT'
        elif text == 'reused grasp did not retain the object to transport clearance':
            category = 'GRASP_RETENTION_GATE'
        elif text == 'object did not satisfy the lift success criterion':
            category = 'LIFT_CRITERION'
        elif text == 'object was not released stably inside target region':
            category = 'PLACEMENT_CRITERION'
        elif text == 'object was not stacked stably on the support':
            category = 'STACK_CRITERION'
        elif text == 'object was not stably pushed by the four-finger tactile plane':
            category = 'PUSH_CRITERION'
        elif text == 'grasped bolt was not stably inserted in the fixed nut':
            category = 'ASSEMBLY_CRITERION'
    return {'schema': 'reported-task-outcome-v1', 'success': bool(success),
            'failure_category': category, 'failure_reason': reason,
            'classification_basis': 'Executor message, not inferred collision/slip/physical root cause'}

POLICIES = {
    'UNKNOWN_EXECUTION_ERROR': (False, 1, False),
    'SESSION_INTERRUPTED': (True, None, True),
    'RUNTIME_PROTECTION': (True, None, True),
    'RESOURCE_EXHAUSTED': (True, None, True),
    'EXECUTION_TIMEOUT': (True, 2, False),
    'ENCODER_PIPE_FAILURE': (True, 2, False),
    'TRANSIENT_IO': (True, 3, False),
    'INPUT_CONTRACT_ERROR': (False, 1, False),
    'OUTPUT_VALIDATION_ERROR': (False, 1, False)}


class WorkerFailure(RuntimeError):
    def __init__(self, failure):
        expected = POLICIES.get(failure.get('category')) if isinstance(failure, dict) else None
        if (expected is None or failure.get('schema') != 'production-failure-policy-v1'
                or not isinstance(failure.get('error'), str) or not isinstance(failure.get('phase'), str)
                or type(failure.get('retryable')) is not bool or type(failure.get('stop_coordinator')) is not bool
                or (failure.get('retryable'), failure.get('attempt_limit'), failure.get('stop_coordinator')) != expected
                or isinstance(failure.get('attempt_limit'), bool)):
            raise ValueError('Malformed worker failure policy')
        super().__init__(failure['error'])
        self.failure = dict(failure)


def failure_policy(error, phase, *, interrupted=False):
    if isinstance(error, WorkerFailure):
        return dict(error.failure)
    category, retryable, limit, stop = 'UNKNOWN_EXECUTION_ERROR', False, 1, False
    if interrupted or isinstance(error, InterruptedError):
        category, retryable, limit, stop = 'SESSION_INTERRUPTED', True, None, True
    elif phase == 'RUNTIME_GUARD':
        category, retryable, limit, stop = 'RUNTIME_PROTECTION', True, None, True
    elif isinstance(error, MemoryError) or (isinstance(error, OSError) and error.errno in (
            errno.ENOMEM, errno.ENOSPC, errno.EDQUOT)):
        category, retryable, limit, stop = 'RESOURCE_EXHAUSTED', True, None, True
    elif isinstance(error, TimeoutError):
        category, retryable, limit = 'EXECUTION_TIMEOUT', True, 2
    elif isinstance(error, BrokenPipeError):
        category, retryable, limit = 'ENCODER_PIPE_FAILURE', True, 2
    elif isinstance(error, OSError) and error.errno in (errno.EAGAIN, errno.EBUSY, errno.EINTR, errno.ETIMEDOUT):
        category, retryable, limit = 'TRANSIENT_IO', True, 3
    elif phase == 'INPUT_VALIDATION' or isinstance(error, (ValueError, KeyError, FileNotFoundError)):
        category = 'INPUT_CONTRACT_ERROR' if phase == 'INPUT_VALIDATION' else 'OUTPUT_VALIDATION_ERROR'
    elif phase in ('REPLAY', 'STRICT', 'PACKAGING', 'REGISTERING'):
        category = 'OUTPUT_VALIDATION_ERROR'
    return {'schema': 'production-failure-policy-v1', 'category': category, 'phase': phase,
            'error': f'{type(error).__name__}: {error}', 'retryable': retryable,
            'attempt_limit': limit, 'stop_coordinator': stop}
