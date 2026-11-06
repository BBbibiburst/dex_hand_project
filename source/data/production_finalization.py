"""Require independent file acceptance after the queue reports quota completion."""

from pathlib import Path
import time

from source.data.production_monitor import record_runtime_failure
from source.data.production_report import build_report, save_report


def finalize_dataset(config, root, database):
    root = Path(root)
    effective = {**config, 'dataset_root': str(root), 'queue_database': str(database)}
    output = root.parent / (root.name + '_acceptance_reports') / f'final_{time.time_ns()}'
    try:
        report = build_report(effective)
        save_report(report, output)
        accepted = report['acceptance']['formal_dataset_accepted']
        return {'state': 'FILE_ACCEPTED' if accepted else 'FILE_REJECTED',
                'formal_dataset_accepted': accepted, 'report_directory': str(output),
                'file_integrity_passed': report['acceptance']['file_integrity_passed']}
    except Exception as error:
        failure = {'phase': 'FINAL_FILE_ACCEPTANCE', 'error': f'{type(error).__name__}: {error}',
                   'report_directory': str(output)}
        try:
            failure['failure_record'] = record_runtime_failure(root, 'final_report', failure)
        except Exception as logging_error:
            failure['failure_record_error'] = f'{type(logging_error).__name__}: {logging_error}'
        return {'state': 'REPORT_FAILED', 'formal_dataset_accepted': False, **failure}
