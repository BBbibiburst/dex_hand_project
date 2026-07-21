"""Wall-clock pacing for interactive MuJoCo loops."""

from __future__ import annotations

import time


class RealtimePacer:
    """Synchronize simulation time to wall-clock time.

    Control and rendering policies remain owned by the caller.  This helper
    only delays the loop when simulation time is ahead of wall-clock time.
    """

    def __init__(self) -> None:
        self._wall_start = 0.0
        self._sim_start = 0.0

    def reset(self, sim_time: float) -> None:
        self._wall_start = time.perf_counter()
        self._sim_start = float(sim_time)

    def sleep_until(self, sim_time: float) -> None:
        target_wall_time = self._wall_start + float(sim_time) - self._sim_start
        delay = target_wall_time - time.perf_counter()
        if delay > 0.0:
            time.sleep(delay)
