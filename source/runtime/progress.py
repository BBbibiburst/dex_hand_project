"""Dependency-free live terminal progress for long multi-process workflows."""

from __future__ import annotations

from dataclasses import dataclass
import os
import shutil
import sys
import time


@dataclass
class WorkerProgress:
    object_id: str
    phase: str
    current: int | None
    total: int | None
    detail: str
    started: float
    updated: float


class LiveWorkerProgress:
    """Render one stable terminal row per active worker."""

    def __init__(
        self,
        *,
        total: int,
        workers: int,
        stream=None,
        columns: int | None = None,
    ) -> None:
        self.total = total
        self.workers = workers
        self.stream = stream or sys.stdout
        self.interactive = bool(
            self.stream.isatty()
            and (os.environ.get("TERM") != "dumb" or columns is not None)
        )
        self.states: dict[str, WorkerProgress] = {}
        self.worker_labels: dict[str, str] = {}
        self.completed = 0
        self.solved = 0
        self.started = time.monotonic()
        self._rendered_lines = 0
        self._last_noninteractive: dict[str, tuple[str, float]] = {}
        self._fixed_columns = columns

    @staticmethod
    def _bar(current: int | None, total: int | None, width: int = 14) -> str:
        if current is None or total is None or total <= 0:
            position = int(time.monotonic() * 5) % width
            cells = ["░"] * width
            cells[position] = "█"
            return "".join(cells)
        filled = min(width, max(0, round(width * current / total)))
        return "█" * filled + "░" * (width - filled)

    @staticmethod
    def _duration(seconds: float) -> str:
        seconds = max(0, int(seconds))
        minutes, seconds = divmod(seconds, 60)
        hours, minutes = divmod(minutes, 60)
        return f"{hours:d}h{minutes:02d}m" if hours else f"{minutes:d}m{seconds:02d}s"

    def update(self, event: dict) -> None:
        worker = str(event["worker"])
        if worker not in self.worker_labels:
            self.worker_labels[worker] = f"W{len(self.worker_labels) + 1}"
        now = time.monotonic()
        previous = self.states.get(worker)
        self.states[worker] = WorkerProgress(
            object_id=str(event["object_id"]),
            phase=str(event["phase"]),
            current=event.get("current"),
            total=event.get("total"),
            detail=str(event.get("detail") or ""),
            started=previous.started
            if previous and previous.object_id == event["object_id"]
            else now,
            updated=now,
        )
        if not self.interactive:
            last_phase, last_time = self._last_noninteractive.get(worker, ("", 0.0))
            if last_phase != event["phase"] or now - last_time >= 30.0:
                count = ""
                if event.get("total") is not None:
                    count = f" {event.get('current', 0)}/{event['total']}"
                print(
                    f"PROGRESS {event['object_id']} {event['phase']}{count} "
                    f"{event.get('detail') or ''}",
                    file=self.stream,
                    flush=True,
                )
                self._last_noninteractive[worker] = (str(event["phase"]), now)

    def mark_completed(self, *, object_id: str, solved: bool) -> None:
        self.completed += 1
        self.solved += int(solved)
        self.states = {
            worker: state for worker, state in self.states.items() if state.object_id != object_id
        }

    def clear(self) -> None:
        if not self.interactive or not self._rendered_lines:
            return
        self.stream.write(f"\x1b[{self._rendered_lines}A")
        for index in range(self._rendered_lines):
            self.stream.write("\x1b[2K\r")
            if index + 1 < self._rendered_lines:
                self.stream.write("\x1b[1B")
        if self._rendered_lines > 1:
            self.stream.write(f"\x1b[{self._rendered_lines - 1}A")
        self.stream.flush()
        self._rendered_lines = 0

    def render(self) -> None:
        if not self.interactive:
            now = time.monotonic()
            for worker, state in self.states.items():
                last_phase, last_time = self._last_noninteractive.get(worker, ("", 0.0))
                if now - last_time < 30.0:
                    continue
                count = ""
                if state.total is not None:
                    count = f" {state.current or 0}/{state.total}"
                print(
                    f"PROGRESS {state.object_id} {state.phase}{count} "
                    f"elapsed={self._duration(now - state.started)} {state.detail}",
                    file=self.stream,
                    flush=True,
                )
                self._last_noninteractive[worker] = (last_phase or state.phase, now)
            return
        self.clear()
        columns = self._fixed_columns or shutil.get_terminal_size(fallback=(100, 24)).columns

        def fit(line: str) -> str:
            # Never let a logical worker row wrap into two physical terminal
            # rows; cursor-up redraw relies on this invariant.
            return line[: max(20, columns - 1)]

        elapsed = time.monotonic() - self.started
        eta = (
            None
            if self.completed == 0
            else elapsed / self.completed * (self.total - self.completed)
        )
        eta_text = "warming_up" if eta is None else self._duration(eta)
        lines = [
            fit(
                f"Overall [{self._bar(self.completed, self.total, width=12)}] "
                f"{self.completed}/{self.total} solved={self.solved} "
                f"elapsed={self._duration(elapsed)} eta={eta_text}"
            )
        ]
        for worker in sorted(self.states):
            state = self.states[worker]
            label = self.worker_labels[worker]
            count = "" if state.total is None else f" {state.current or 0}/{state.total}"
            lines.append(
                fit(
                    f"{label:<3} {state.object_id:<27.27} {state.phase:<18.18} "
                    f"[{self._bar(state.current, state.total, width=12)}]{count:<7} "
                    f"{self._duration(time.monotonic() - state.started):>6} {state.detail}"
                ).rstrip()
            )
        self.stream.write("\n".join(lines) + "\n")
        self.stream.flush()
        self._rendered_lines = len(lines)

    def close(self) -> None:
        self.clear()
