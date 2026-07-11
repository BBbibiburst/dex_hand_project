"""Standalone Vive tracker link test using the project's hardware adapter."""

from __future__ import annotations

import argparse
import time

from source.teleop.devices import ViveApiTracker


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    selector = parser.add_mutually_exclusive_group()
    selector.add_argument("--device-index", type=int, help="SteamVR tracked-device index.")
    selector.add_argument("--serial", help="Tracker serial number.")
    parser.add_argument("--interval", type=float, default=0.1, help="Print interval in seconds.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.interval <= 0:
        raise ValueError("--interval must be positive.")
    tracker = ViveApiTracker(device_index=args.device_index, serial=args.serial)
    try:
        tracker.connect()
        origin = None
        frame = 0
        print("开始读取 Vive Tracker，按 Ctrl+C 退出。")
        while True:
            sample = tracker.read()
            if not sample.valid:
                print("Tracker 姿态暂时无效，请检查遮挡和 SteamVR 状态。")
                time.sleep(args.interval)
                continue
            if origin is None:
                origin = sample.position.copy()
            delta = sample.position - origin
            print(
                f"帧 {frame:05d} | pos(m)={sample.position.round(4).tolist()} | "
                f"delta(m)={delta.round(4).tolist()} | "
                f"quat(wxyz)={sample.quaternion_wxyz.round(4).tolist()}"
            )
            frame += 1
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\n已退出 Vive 链路测试。")
    finally:
        tracker.close()


if __name__ == "__main__":
    main()
