"""Live Vive 6D pose plus Bluetooth-glove finger control visualization."""

from __future__ import annotations

import argparse
from collections import deque

import numpy as np

from source.teleop.config import load_teleop_config, save_glove_calibration
from source.teleop.devices import StretchGloveApiDevice, ViveApiTracker
from source.teleop.glove_processing import GloveValueFilter, read_latest_glove
from source.teleop.vive.coordinates import remap_pose, rotation_matrix_to_rpy_degrees
from source.teleop.vive.hand_skeleton import FINGER_NAMES, make_hand_lines
from source.teleop.vive.vive_plot_style import (
    BG, CYAN, GREEN, ORANGE, PURPLE, WHITE, YELLOW, apply_theme, draw_floor,
    style_3d_axis, style_panel, update_frame_axes,
)


class LiveGloveHandPlot:
    """Dark live dashboard combining tracked pose and finger flexion."""

    def __init__(
        self,
        axis_range: float,
        trail_length: int,
        smoothing: float,
        deadzone: float,
        closed_deadzone: float,
        thumb_rotation: float,
    ):
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
        self.thumb_rotation = float(thumb_rotation)
        self._calibration_confirmed = False
        self._calibration_pose_index = 0
        self.glove_filter = GloveValueFilter(
            smoothing=smoothing,
            deadzone=deadzone,
            closed_deadzone=closed_deadzone,
        )
        self.figure = plt.figure("Vive + Glove Control", figsize=(14, 8), facecolor=BG)
        self.figure.canvas.mpl_connect("key_press_event", self._on_key)
        self.figure.suptitle(
            "◆  VIVE + BLUETOOTH GLOVE  ·  LIVE HAND CONTROL",
            fontsize=13, fontweight="bold",
        )
        grid = GridSpec(1, 3, figure=self.figure, width_ratios=(1.7, 1.7, 1.15))
        self.axes = self.figure.add_subplot(grid[0, :2], projection="3d")
        self.info_axis = self.figure.add_subplot(grid[0, 2])
        style_panel(self.info_axis, "GLOVE FLEXION + 6D POSE")
        colors = (CYAN, GREEN, ORANGE, PURPLE, YELLOW)
        bar_y = np.asarray((10.5, 9.5, 8.5, 7.5, 6.5))
        self.finger_bars = self.info_axis.barh(
            bar_y, np.zeros(5), height=0.56, color=colors
        )
        self.info_axis.set_xlim(0, 1)
        self.info_axis.set_ylim(0, 11.7)
        self.info_axis.set_yticks(bar_y, FINGER_NAMES)
        self.info_axis.set_xticks((0.0, 0.5, 1.0))
        self.info_axis.xaxis.tick_top()
        self.info_axis.tick_params(axis="x", pad=3)
        self.info_axis.axhline(5.75, color="#30363d", linewidth=1.0)
        self.pose_text = self.info_axis.text(
            0.04, 0.47, "WAITING FOR DEVICES", va="top", fontsize=8.0,
            transform=self.info_axis.transAxes,
        )
        self.glove_text = self.info_axis.text(
            0.04, 0.19, "RAW / FILTERED", va="top", fontsize=7.2,
            transform=self.info_axis.transAxes,
        )
        self.hand_artists = [
            self.axes.plot([], [], [], color=color, linewidth=3.3, marker="o", markersize=3)[0]
            for color in (WHITE, WHITE, CYAN, GREEN, ORANGE, PURPLE, YELLOW)
        ]
        (self.trajectory,) = self.axes.plot(
            [], [], [], color=ORANGE, linewidth=1.4, alpha=0.65, label="trajectory"
        )
        self.frame_axes = [self.axes.plot([], [], [], linewidth=2)[0] for _ in range(3)]
        self.axes.legend(loc="lower left", fontsize=7)
        plt.ion()
        plt.show(block=False)

    @property
    def is_open(self) -> bool:
        return self.plt.fignum_exists(self.figure.number)

    def wait_for_calibration_pose(self, pose: str) -> None:
        """Show calibration instructions in the dashboard instead of the terminal."""
        prompts = (
            "CLOSE YOUR HAND INTO A FIST",
            "OPEN AND FLATTEN ALL FINGERS",
        )
        if self._calibration_pose_index < len(prompts):
            pose = prompts[self._calibration_pose_index]
        self._calibration_pose_index += 1
        self._calibration_confirmed = False
        self.pose_text.set_text(
            "GLOVE CALIBRATION\n\n"
            f"{pose}\n\n"
            "Hold the pose steady,\n"
            "then press C to start sampling."
        )
        self.glove_text.set_text("WAITING FOR C")
        self.figure.canvas.draw_idle()
        self.figure.canvas.flush_events()
        while self.is_open and not self._calibration_confirmed:
            self.plt.pause(0.03)
        if not self.is_open:
            raise KeyboardInterrupt
        self.pose_text.set_text(f"SAMPLING\n\n{pose}\n\nHOLD STEADY...")
        self.glove_text.set_text("CALIBRATING")
        self.figure.canvas.draw_idle()
        self.figure.canvas.flush_events()
        self.plt.pause(0.05)

    def show_calibration_saved(self, path) -> None:
        self.pose_text.set_text("CALIBRATION SAVED\n\nCONNECTING VIVE TRACKER...")
        self.glove_text.set_text(f"SAVED\n{path}")
        self.figure.canvas.draw_idle()
        self.figure.canvas.flush_events()
        self.plt.pause(0.4)

    def _on_key(self, event) -> None:
        if str(event.key or "").lower() == "c":
            self._calibration_confirmed = True

    def update(self, position, rotation, glove_values) -> None:
        position = np.asarray(position, dtype=float)
        raw_values = np.clip(np.asarray(glove_values, dtype=float), 0, 1)
        glove_values = self.glove_filter.update(raw_values)
        glove_values[4] = self.thumb_rotation
        display_raw = np.concatenate((raw_values[:4], raw_values[5:6]))
        display_values = np.concatenate((glove_values[:4], glove_values[5:6]))
        if self.origin is None:
            self.origin = position.copy()
            style_3d_axis(self.axes, self.origin, self.axis_range)
            draw_floor(self.axes, self.origin, self.axis_range)
        self.trail.append(position.copy())
        trail = np.asarray(self.trail)
        local_lines = make_hand_lines(glove_values)
        for local_line, artist in zip(local_lines, self.hand_artists):
            line = local_line @ rotation.T + position
            artist.set_data_3d(line[:, 0], line[:, 1], line[:, 2])
        self.trajectory.set_data_3d(trail[:, 0], trail[:, 1], trail[:, 2])
        update_frame_axes(self.axes, self.frame_axes, position, rotation, self.axis_range * 0.16)
        for bar, value in zip(self.finger_bars, display_values):
            bar.set_width(value)
        roll, pitch, yaw = rotation_matrix_to_rpy_degrees(rotation)
        self.pose_text.set_text(
            "● VIVE  ● GLOVE\n"
            f"XYZ  {position[0]:+7.3f}  {position[1]:+7.3f}  {position[2]:+7.3f} m\n"
            f"RPY  {roll:+7.1f}  {pitch:+7.1f}  {yaw:+7.1f} deg"
        )
        self.glove_text.set_text(
            "RAW / FILTERED\n"
            + "   ".join(
                f"{name[:2].upper()} {raw:.2f}/{filtered:.2f}"
                for name, raw, filtered in zip(
                    FINGER_NAMES[:3], display_raw[:3], display_values[:3]
                )
            )
            + "\n"
            + "   ".join(
                f"{name[:2].upper()} {raw:.2f}/{filtered:.2f}"
                for name, raw, filtered in zip(
                    FINGER_NAMES[3:], display_raw[3:], display_values[3:]
                )
            )
            + f"\nTHUMB ROTATION FIXED {self.thumb_rotation:.2f}"
        )
        self.figure.canvas.draw_idle()
        self.figure.canvas.flush_events()
        self.plt.pause(0.001)

    def close(self) -> None:
        if self.is_open:
            self.plt.close(self.figure)


def parse_args() -> argparse.Namespace:
    config = load_teleop_config()
    calibration = dict(config.get("glove_calibration") or {})
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--glove-mac", default=config.get("glove_mac"))
    parser.add_argument("--glove-serial-port", default=config.get("glove_serial_port"))
    parser.add_argument("--glove-channel", type=int, default=int(config.get("glove_channel", 1)))
    parser.add_argument("--glove-baudrate", type=int, default=int(config.get("glove_baudrate", 9600)))
    parser.add_argument(
        "--glove-calibration-seconds",
        type=float,
        default=float(config.get("glove_calibration_seconds", 3.0)),
    )
    parser.add_argument(
        "--recalibrate-glove",
        action="store_true",
        help="Calibrate fist/open poses now and save the new bounds.",
    )
    parser.add_argument("--vive-device-index", type=int)
    parser.add_argument("--vive-serial")
    parser.add_argument("--interval", type=float, default=0.05)
    parser.add_argument("--axis-range", type=float, default=0.5)
    parser.add_argument("--trail-length", type=int, default=200)
    parser.add_argument(
        "--glove-smoothing",
        type=float,
        default=0.70,
        help="EMA response factor in (0, 1]; larger values respond faster.",
    )
    parser.add_argument(
        "--glove-deadzone",
        type=float,
        default=float(config.get("glove_open_deadzone", 0.10)),
        help="Normalized open-end interval mapped exactly to 0.",
    )
    parser.add_argument(
        "--glove-closed-deadzone",
        type=float,
        default=float(config.get("glove_closed_deadzone", 0.10)),
        help="Normalized closed-end interval mapped exactly to 1.",
    )
    parser.add_argument(
        "--thumb-rotation",
        type=float,
        default=float(config.get("teleop_thumb_rotation", 0.25)),
        help="Fixed normalized thumb-opposition value in [0, 1].",
    )
    parser.set_defaults(
        glove_calibration_minimum=calibration.get("open_minimum"),
        glove_calibration_maximum=calibration.get("fist_maximum"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.glove_mac:
        raise ValueError("请在 configs/teleop.json 或 --glove-mac 中配置手套 MAC。")
    if args.interval <= 0 or args.axis_range <= 0 or args.trail_length <= 0:
        raise ValueError("interval, axis-range and trail-length must be positive.")
    if not 0 < args.glove_smoothing <= 1:
        raise ValueError("--glove-smoothing must be in (0, 1].")
    if (
        args.glove_deadzone < 0
        or args.glove_closed_deadzone < 0
        or args.glove_deadzone + args.glove_closed_deadzone >= 1
    ):
        raise ValueError("Glove endpoint deadzones must be non-negative and sum to < 1.")
    if not 0 <= args.thumb_rotation <= 1:
        raise ValueError("--thumb-rotation must be in [0, 1].")

    calibration_minimum = None if args.recalibrate_glove else args.glove_calibration_minimum
    calibration_maximum = None if args.recalibrate_glove else args.glove_calibration_maximum
    glove = StretchGloveApiDevice(
        args.glove_mac,
        channel=args.glove_channel,
        serial_port=args.glove_serial_port,
        baudrate=args.glove_baudrate,
        calibration_seconds=args.glove_calibration_seconds,
        calibration_confirmation=(
            (lambda pose: plot.wait_for_calibration_pose(pose))
            if args.recalibrate_glove
            else None
        ),
        calibration_minimum=calibration_minimum,
        calibration_maximum=calibration_maximum,
    )
    vive = ViveApiTracker(device_index=args.vive_device_index, serial=args.vive_serial)
    plot = LiveGloveHandPlot(
        args.axis_range,
        args.trail_length,
        smoothing=args.glove_smoothing,
        deadzone=args.glove_deadzone,
        closed_deadzone=args.glove_closed_deadzone,
        thumb_rotation=args.thumb_rotation,
    )
    try:
        print("正在连接蓝牙手套……")
        glove.connect()
        if args.recalibrate_glove:
            minimum, maximum = glove.calibration_bounds()
            saved_path = save_glove_calibration(minimum.tolist(), maximum.tolist())
            plot.show_calibration_saved(saved_path)
            print(f"现场校准已保存到：{saved_path}")
        print("正在连接 Vive Tracker……")
        vive.connect()
        print("连接成功：Vive 控制六维位姿，手套控制五指弯曲。")
        while plot.is_open:
            vive_sample = vive.read()
            if not vive_sample.valid:
                plot.plt.pause(args.interval)
                continue
            # Rendering is slower than the glove stream. Drop queued serial/RFCOMM
            # packets so animation always uses a fresh sample instead of building
            # up several seconds of latency while replaying stale samples.
            glove_values = np.asarray(read_latest_glove(glove).stretch, dtype=float)
            position, rotation = remap_pose(
                vive_sample.position, vive_sample.quaternion_wxyz
            )
            plot.update(position, rotation, glove_values)
    except KeyboardInterrupt:
        print("\n已退出 Vive + 手套控制。")
    finally:
        glove.close()
        vive.close()
        plot.close()


if __name__ == "__main__":
    main()
