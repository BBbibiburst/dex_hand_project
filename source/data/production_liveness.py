"""Keep a claimed job alive through simulation, packaging and file verification."""

import threading
import sys


class LeaseKeeper:
    def __init__(self, queue, job, stopped, *, guard=None, interval=10., lease_seconds=300., on_failure=None):
        self.queue = queue
        self.job = job
        self.stopped = stopped
        self.guard = guard
        self.interval = interval
        self.lease_seconds = lease_seconds
        self.finished = threading.Event()
        self.error = None
        self.thread = None
        self.on_failure = on_failure
        self.failure_reporting_error = None

    def _record_failure(self, error):
        self.error = error
        self.stopped.set()
        if self.on_failure is not None:
            try:
                self.on_failure(error)
            except Exception as report_error:
                self.failure_reporting_error = report_error
                print(f'Runtime failure reporting also failed: {report_error}', file=sys.stderr, flush=True)

    def _pulse(self):
        if self.finished.is_set():
            return
        if self.guard is not None:
            self.guard()
        if self.finished.is_set():
            return
        try:
            self.queue.heartbeat(self.job['job_id'], self.job['lease_token'], self.lease_seconds,
                                 cancelled=self.finished)
        except ValueError:
            if self.finished.is_set():
                return
            # Registration can finish while the last pulse waits for SQLite.
            with self.queue.connect() as db:
                row = db.execute('SELECT state FROM jobs WHERE job_id=?', (self.job['job_id'],)).fetchone()
            if row is not None and row['state'] in ('EXPORTED', 'DUPLICATE', 'QUARANTINED'):
                self.finished.set()
                return
            raise

    def start(self):
        try:
            self._pulse()
        except Exception as error:
            self._record_failure(error)
            raise

        def maintain():
            while not self.finished.wait(self.interval):
                try:
                    self._pulse()
                except Exception as error:
                    self._record_failure(error)
                    return

        self.thread = threading.Thread(target=maintain, name='episode-lease', daemon=True)
        self.thread.start()

    def check(self):
        if self.error is not None:
            raise RuntimeError(f'Episode lease or runtime guard failed: {self.error}') from self.error

    def close(self):
        self.finished.set()
        if self.thread is not None:
            self.thread.join(timeout=35)
            if self.thread.is_alive():
                self.stopped.set()
                raise RuntimeError('Episode lease thread did not stop')
