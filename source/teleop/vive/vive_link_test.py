"""Read and dynamically display a Vive pose with the dex_hand model."""

from __future__ import annotations

import argparse
from collections import deque
import time

import numpy as np

from source.teleop.devices import ViveApiTracker
from source.teleop.vive.hand_skeleton import make_hand_lines
from source.teleop.vive.coordinates import (
    remap_pose,
    rotation_matrix_to_rpy_degrees,
)
from source.teleop.vive.vive_plot_style import (
    BG, CYAN, GREEN, ORANGE, PURPLE, RED, WHITE, apply_theme, draw_floor,
    style_3d_axis, style_panel, update_frame_axes,
)


class LivePosePlot:
    """Polished dark dashboard for the Vive hand pose."""

    def __init__(self, axis_range: float, trail_length: int):
        try:
            import matplotlib.pyplot as plt
            from matplotlib.gridspec import GridSpec
        except ImportError as exc:
            raise RuntimeError("需要 matplotlib，请先执行: pip install matplotlib") from exc
        apply_theme(plt)
        self.plt = plt
        self.axis_range = axis_range
        self.trail = deque(maxlen=trail_length)
        self.origin = None
        self.local_hand_lines = make_hand_lines()
        self.figure = plt.figure("Vive Hand 6D Pose", figsize=(13, 8), facecolor=BG)
        self.figure.suptitle("◆  VIVE TRACKER  ·  LIVE 6D HAND POSE", fontsize=13, fontweight="bold")
        grid = GridSpec(2, 3, figure=self.figure, width_ratios=(1.7, 1.7, 1.0), hspace=0.35)
        self.axes = self.figure.add_subplot(grid[:, :2], projection="3d")
        self.position_axis = self.figure.add_subplot(grid[0, 2])
        self.rotation_axis = self.figure.add_subplot(grid[1, 2])
        style_panel(self.position_axis, "POSITION · metres")
        style_panel(self.rotation_axis, "ORIENTATION · degrees")
        axis_colors = (RED, GREEN, CYAN)
        self.position_bars = self.position_axis.barh(
            ("X right", "Y forward", "Z up"), (0, 0, 0), color=axis_colors
        )
        self.rotation_bars = self.rotation_axis.barh(
            ("roll · X", "pitch · Y", "yaw · Z"), (0, 0, 0), color=axis_colors
        )
        self.rotation_axis.set_xlim(-180, 180)
        self.hand_artists = [
            self.axes.plot([], [], [], color=CYAN, linewidth=3.2, marker="o", markersize=3)[0]
            for _ in self.local_hand_lines
        ]
        (self.trajectory,) = self.axes.plot(
            [], [], [], color=ORANGE, linewidth=1.5, alpha=0.75, label="trajectory"
        )
        self.frame_axes = [self.axes.plot([], [], [], linewidth=2)[0] for _ in range(3)]
        self.status = self.axes.text2D(
            0.025, 0.97, "WAITING FOR TRACKER", transform=self.axes.transAxes, va="top",
            fontsize=8, bbox=dict(boxstyle="round,pad=0.5", fc=BG, ec="#30363d", alpha=0.9),
        )
        self.axes.legend(loc="lower left", fontsize=7)
        plt.ion()
        plt.show(block=False)

    @property
    def is_open(self) -> bool:
        return self.plt.fignum_exists(self.figure.number)

    def update(self, position, rotation, rpy_degrees) -> None:
        position = np.asarray(position, dtype=float)
        if self.origin is None:
            self.origin = position.copy()
            style_3d_axis(self.axes, self.origin, self.axis_range)
            draw_floor(self.axes, self.origin, self.axis_range)
        self.trail.append(position.copy())
        trail = np.asarray(self.trail)
        for local_line, artist in zip(self.local_hand_lines, self.hand_artists):
            line = local_line @ rotation.T + position
            artist.set_data_3d(line[:, 0], line[:, 1], line[:, 2])
        self.trajectory.set_data_3d(trail[:, 0], trail[:, 1], trail[:, 2])
        update_frame_axes(self.axes, self.frame_axes, position, rotation, self.axis_range * 0.16)
        roll, pitch, yaw = rpy_degrees
        for bar, value in zip(self.position_bars, position):
            bar.set_width(value)
        limit = max(self.axis_range, float(np.max(np.abs(position))) * 1.15)
        self.position_axis.set_xlim(-limit, limit)
        for bar, value in zip(self.rotation_bars, rpy_degrees):
            bar.set_width(value)
        self.status.set_text(
            f"● TRACKING\nXYZ  {position[0]:+7.3f}  {position[1]:+7.3f}  {position[2]:+7.3f} m\n"
            f"RPY  {roll:+7.1f}  {pitch:+7.1f}  {yaw:+7.1f} deg"
        )
        self.figure.canvas.draw_idle()
        self.figure.canvas.flush_events()
        self.plt.pause(0.001)

    def close(self) -> None:
        if self.is_open:
            self.plt.close(self.figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    selector = parser.add_mutually_exclusive_group()
    selector.add_argument("--device-index", type=int, help="SteamVR tracked-device index.")
    selector.add_argument("--serial", help="Tracker serial number.")
    parser.add_argument("--interval", type=float, default=0.05, help="Refresh interval in seconds.")
    parser.add_argument("--axis-range", type=float, default=0.5, help="Plot radius in metres.")
    parser.add_argument("--trail-length", type=int, default=200, help="Number of trail points.")
    parser.add_argument("--no-plot", action="store_true", help="Only print pose values.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.interval <= 0 or args.axis_range <= 0 or args.trail_length <= 0:
        raise ValueError("interval, axis-range and trail-length must be positive.")

    tracker = ViveApiTracker(device_index=args.device_index, serial=args.serial)
    plot = None if args.no_plot else LivePosePlot(args.axis_range, args.trail_length)
    try:
        tracker.connect()
        print("开始读取 Vive Tracker，关闭图窗或按 Ctrl+C 退出。")
        while plot is None or plot.is_open:
            sample = tracker.read()
            if not sample.valid:
                print("Tracker 位姿暂时无效，请检查遮挡和 SteamVR 状态。")
                if plot is not None:
                    plot.plt.pause(args.interval)
                else:
                    time.sleep(args.interval)
                continue
            position, rotation = remap_pose(sample.position, sample.quaternion_wxyz)
            rpy = rotation_matrix_to_rpy_degrees(rotation)
            if plot is not None:
                plot.update(position, rotation, rpy)
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\n已退出 Vive 动态位姿显示。")
    finally:
        tracker.close()
        if plot is not None:
            plot.close()


if __name__ == "__main__":
    main()
