# -*- coding: utf-8 -*-
"""Keyboard state for interactive teleoperation collection."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TeleopUIState:
    recording: bool = False
    save_requested: bool = False
    discard_requested: bool = False
    calibration_requested: bool = False
    quit_requested: bool = False

    def handle_key(self, keycode: int) -> None:
        if keycode == 32:
            self.recording = not self.recording
            print("recording" if self.recording else "paused")
        elif keycode in (ord("N"), ord("n")):
            self.save_requested = True
        elif keycode in (ord("R"), ord("r")):
            self.discard_requested = True
        elif keycode in (ord("C"), ord("c")):
            self.calibration_requested = True
        elif keycode in (ord("Q"), ord("q")):
            self.quit_requested = True

    def consume_save_request(self) -> bool:
        requested = self.save_requested
        self.save_requested = False
        return requested

    def consume_discard_request(self) -> bool:
        requested = self.discard_requested
        self.discard_requested = False
        return requested

    def consume_calibration_request(self) -> bool:
        requested = self.calibration_requested
        self.calibration_requested = False
        return requested
