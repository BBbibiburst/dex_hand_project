"""Matplotlib dashboard for interactive teleoperation and data collection."""

from __future__ import annotations

from collections import deque
from typing import Any, Mapping

import numpy as np


class TeleopDashboard:
    """Camera, tactile heatmaps, status, and keyboard input in one window."""

    def __init__(self, tactile_sensor, *, title: str = "Teleop Data Collection") -> None:
        try:
            import matplotlib.pyplot as plt
        except ImportError as exc:
            raise RuntimeError(
                "The teleoperation dashboard requires matplotlib."
            ) from exc
        self.plt = plt
        self.sensor = tactile_sensor
        self.title = title
        self.figure = plt.figure(title, figsize=(14, 8), facecolor="#181b20")
        self.figure.canvas.manager.set_window_title(title)
        self.figure.canvas.mpl_connect("key_press_event", self._on_key)
        self._keys: deque[int] = deque()
        self._camera_artist = None
        self._heatmap_artists: dict[str, Any] = {}
        self._hand_bars: list[Any] = []
        self._hand_value_texts: list[Any] = []
        self._status_text = None
        self._controls_text = None
        self._message_text = None
        self._patch_names: tuple[str, ...] = ()
        plt.ion()
        plt.show(block=False)
        self._place_window_on_right()

    @property
    def is_open(self) -> bool:
        return self.plt.fignum_exists(self.figure.number)

    def update(
        self,
        camera_rgb: np.ndarray,
        tactile_values: Any,
        *,
        state: str,
        episode: int,
        episodes: int,
        frames: int,
        frame_limit: int,
        success: bool,
        message: str = "",
        target_position: Any = None,
        hand_values: Any = None,
        raw_tactile_values: Any = None,
        grasp_contacts: int = 0,
    ) -> int:
        """Draw one frame and return an ASCII key code, or -1."""
        camera = np.asarray(camera_rgb, dtype=np.uint8)
        tactile = np.asarray(tactile_values, dtype=np.float32)
        patches = self._patches(tactile)
        patch_names = tuple(patches)
        hand = (
            np.zeros(6, dtype=np.float32)
            if hand_values is None
            else np.clip(np.asarray(hand_values, dtype=np.float32).reshape(6), 0, 1)
        )
        if self._camera_artist is None or patch_names != self._patch_names:
            self._build_layout(camera, patches, hand)
        else:
            self._camera_artist.set_data(camera)
            self._update_hand(hand)
            for name, values in patches.items():
                artist = self._heatmap_artists[name]
                artist.set_data(np.asarray(values, dtype=np.float32))
                maximum = max(float(np.max(values, initial=0.0)), 1e-6)
                artist.set_clim(0.0, maximum)

        color = "#ff5252" if state == "REC" else "#ffd54f"
        self._status_text.set_color(color)
        tactile_max = float(np.max(tactile, initial=0.0))
        tactile_active = int(np.count_nonzero(tactile > 0.0))
        raw_tactile_max = (
            tactile_max
            if raw_tactile_values is None
            else float(np.max(np.asarray(raw_tactile_values), initial=0.0))
        )
        self._status_text.set_text(
            f"{state}  |  episode {episode}/{episodes}  |  "
            f"frames {frames}/{frame_limit}  |  success {success}\n"
            f"grasp contacts {int(grasp_contacts)}  |  tactile raw max "
            f"{raw_tactile_max:.3g}  |  processed max {tactile_max:.3g}  "
            f"({tactile_active} active)"
        )
        target_text = ""
        if target_position is not None:
            target = np.asarray(target_position, dtype=float).reshape(3)
            target_text = (
                f"  |  target xyz "
                f"{target[0]:+.3f} {target[1]:+.3f} {target[2]:+.3f} m"
            )
        self._controls_text.set_text(
            "SPACE record/pause  |  C calibrate  |  N save  |  "
            f"R reset  |  Q quit{target_text}"
        )
        self._message_text.set_text(message)
        self.figure.canvas.draw_idle()
        self.figure.canvas.flush_events()
        return self._keys.popleft() if self._keys else -1

    def close(self) -> None:
        if self.is_open:
            self.plt.close(self.figure)

    def _on_key(self, event) -> None:
        key = str(event.key or "").lower()
        mapping = {
            " ": 32,
            "space": 32,
            "c": ord("c"),
            "n": ord("n"),
            "r": ord("r"),
            "q": ord("q"),
        }
        if key in mapping:
            self._keys.append(mapping[key])

    def _place_window_on_right(self) -> None:
        """Keep the dashboard compact and away from the main MuJoCo view."""
        window = getattr(self.figure.canvas.manager, "window", None)
        if window is None:
            return
        # TkAgg, used by the project's Windows environment.
        if hasattr(window, "wm_geometry") and hasattr(window, "winfo_screenwidth"):
            screen_width = int(window.winfo_screenwidth())
            screen_height = int(window.winfo_screenheight())
            width = min(920, max(640, screen_width // 2))
            height = min(820, max(560, screen_height - 100))
            x = max(0, screen_width - width - 10)
            window.wm_geometry(f"{width}x{height}+{x}+40")
            return
        # Qt backends.
        if hasattr(window, "setGeometry") and hasattr(window, "screen"):
            geometry = window.screen().availableGeometry()
            width = min(920, max(640, geometry.width() // 2))
            height = min(820, max(560, geometry.height() - 80))
            x = geometry.x() + geometry.width() - width
            window.setGeometry(x, geometry.y() + 30, width, height)

    def _patches(self, values: np.ndarray) -> Mapping[str, np.ndarray]:
        if hasattr(self.sensor, "patches_from_values"):
            return self.sensor.patches_from_values(values)
        return {"tactile": values.reshape(1, -1)}

    def _build_layout(
        self,
        camera: np.ndarray,
        patches: Mapping[str, np.ndarray],
        hand_values: np.ndarray,
    ) -> None:
        figure = self.figure
        figure.clear()
        names = tuple(patches)
        count = max(1, len(names))
        heat_columns = max(1, int(np.ceil(np.sqrt(count))))
        heat_rows = int(np.ceil(count / heat_columns))
        grid = figure.add_gridspec(
            heat_rows,
            heat_columns + 3,
            width_ratios=[1.45, 1.45, 1.15, *([1.0] * heat_columns)],
            left=0.035,
            right=0.985,
            top=0.93,
            bottom=0.16,
            wspace=0.16,
            hspace=0.28,
        )
        camera_axis = figure.add_subplot(grid[:, :2])
        camera_axis.set_title("AGENTVIEW CAMERA", color="#f0f0f0", fontsize=10)
        camera_axis.axis("off")
        self._camera_artist = camera_axis.imshow(camera)
        hand_axis = figure.add_subplot(grid[:, 2])
        hand_axis.set_title("GLOVE FLEXION", color="#f0f0f0", fontsize=9)
        hand_axis.set_facecolor("#181b20")
        labels = ("INDEX", "MIDDLE", "RING", "PINKY", "THUMB ROT", "THUMB FLEX")
        positions = np.arange(len(labels))[::-1]
        colors = ("#37d5ff", "#4fe391", "#ffae42", "#b388ff", "#ffe066", "#ff7b72")
        self._hand_bars = list(
            hand_axis.barh(
                positions,
                np.zeros(len(labels)),
                height=0.58,
                color=colors,
            )
        )
        hand_axis.set_xlim(0.0, 1.08)
        hand_axis.set_ylim(-0.7, len(labels) - 0.3)
        hand_axis.set_yticks(positions, labels)
        hand_axis.set_xticks((0.0, 0.5, 1.0))
        hand_axis.tick_params(colors="#d6d6d6", labelsize=7)
        hand_axis.grid(axis="x", color="#30363d", linewidth=0.7, alpha=0.8)
        for spine in hand_axis.spines.values():
            spine.set_color("#30363d")
        self._hand_value_texts = [
            hand_axis.text(
                0.02,
                position,
                "0.000",
                va="center",
                ha="left",
                color="#ffffff",
                fontsize=7,
                fontweight="bold",
            )
            for position in positions
        ]
        self._update_hand(hand_values)
        self._heatmap_artists = {}
        for index, name in enumerate(names):
            row, column = divmod(index, heat_columns)
            axis = figure.add_subplot(grid[row, column + 3])
            axis.set_title(name, color="#f0f0f0", fontsize=8)
            axis.set_xticks(())
            axis.set_yticks(())
            values = np.asarray(patches[name], dtype=np.float32)
            maximum = max(float(np.max(values, initial=0.0)), 1e-6)
            self._heatmap_artists[name] = axis.imshow(
                values,
                origin="lower",
                cmap="inferno",
                vmin=0.0,
                vmax=maximum,
                interpolation="nearest",
                aspect="auto",
            )
        self._status_text = figure.text(
            0.035, 0.105, "", fontsize=11, fontweight="bold"
        )
        self._controls_text = figure.text(
            0.035, 0.068, "", color="#d6d6d6", fontsize=8.5
        )
        self._message_text = figure.text(
            0.035, 0.028, "", color="#ffdf4d", fontsize=10, fontweight="bold"
        )
        self._patch_names = names

    def _update_hand(self, values: np.ndarray) -> None:
        for bar, text, value in zip(
            self._hand_bars, self._hand_value_texts, values
        ):
            value = float(value)
            bar.set_width(value)
            text.set_x(min(value + 0.025, 1.015))
            text.set_text(f"{value:.3f}")
