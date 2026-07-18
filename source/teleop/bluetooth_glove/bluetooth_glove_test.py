"""Interactive connection and signal test for the Classic-Bluetooth glove."""

from __future__ import annotations

import argparse
import time

import numpy as np

from source.teleop.config import load_teleop_config, save_glove_calibration
from source.teleop.devices import StretchGloveApiDevice


CHANNEL_NAMES = ("食指", "中指", "无名指", "小指", "拇指", "拇指副本")
CHANNEL_SHORT_NAMES = ("食", "中", "无", "小", "拇", "拇副")


def _parse_args() -> argparse.Namespace:
    config = load_teleop_config()
    parser = argparse.ArgumentParser(
        description="连接并检查五传感器经典蓝牙手套，不启动 MuJoCo、Vive 或机械臂。"
    )
    parser.add_argument(
        "--mac",
        default=config.get("glove_mac"),
        help="手套 MAC；默认读取 configs/teleop.json。",
    )
    parser.add_argument(
        "--pin",
        default=str(config.get("glove_pin", "")),
        help="Windows 首次配对 PIN；仅用于引导，不由 RFCOMM 程序发送。",
    )
    parser.add_argument(
        "--channel",
        type=int,
        default=int(config.get("glove_channel", 1)),
        help="RFCOMM channel",
    )
    parser.add_argument(
        "--serial-port",
        default=config.get("glove_serial_port"),
        help="Windows 配对生成的出站串口；配置后优先于 MAC/RFCOMM。",
    )
    parser.add_argument("--baudrate", type=int, default=int(config.get("glove_baudrate", 9600)))
    parser.add_argument(
        "--calibration-seconds",
        type=float,
        default=float(config.get("glove_calibration_seconds", 3.0)),
    )
    parser.add_argument(
        "--pose-delay-seconds",
        type=float,
        default=2.0,
        help="握拳/张手校准开始前的准备时间。",
    )
    parser.add_argument("--duration", type=float, default=2.0, help="每个确认姿势的稳定采集秒数。")
    parser.add_argument("--display-hz", type=float, default=8.0)
    parser.add_argument(
        "--history-weight",
        type=float,
        default=float(config.get("glove_calibration_history_weight", 0.75)),
        help="保存校准时历史结果的权重，范围 0 到 1（默认 0.75）。",
    )
    parser.add_argument(
        "--no-prompt", action="store_true", help="不等待 Enter，适用于已经准备好的设备。"
    )
    parser.add_argument("--no-save", action="store_true", help="测试通过后不保存校准结果。")
    return parser.parse_args()


def _print_preflight(args: argparse.Namespace) -> None:
    print("\n蓝牙手套独立诊断")
    print("1. 打开手套电源（最右侧按键长按）。")
    pin_hint = f"，配对 PIN 为 {args.pin}" if args.pin else ""
    print(f"2. 在 Windows 蓝牙设置中确认已与 HC-06 配对{pin_hint}。")
    print("3. 按手套中间按键开始发送数据。")
    if args.serial_port:
        port_label = "按 MAC 自动发现" if args.serial_port.lower() == "auto" else args.serial_port
        print(f"4. 即将打开串口（{port_label}），baudrate={args.baudrate}。")
    else:
        print(f"4. 即将连接 {args.mac}，RFCOMM channel={args.channel}。")
    print("5. 连接后按提示先握拳、再张开手掌；准备阶段不要提前变换姿势。")
    if not args.no_prompt:
        input("\n准备完成后按 Enter 连接；Ctrl+C 可随时退出：")


def _print_live(values: np.ndarray, sample_count: int) -> None:
    fields = [f"{name}:{value:.2f}" for name, value in zip(CHANNEL_SHORT_NAMES, values)]
    line = f"样本 {sample_count:6d} | " + " | ".join(fields)
    print(f"\r\033[2K{line}", end="", flush=True)


def _read_values(glove: StretchGloveApiDevice) -> np.ndarray:
    values = np.asarray(glove.read().stretch, dtype=np.float32)
    if values.shape != (6,) or not np.all(np.isfinite(values)):
        raise RuntimeError(f"收到无效手套数据: shape={values.shape}, values={values}")
    return values


def _confirm_pose(glove: StretchGloveApiDevice, instruction: str, no_prompt: bool) -> None:
    if no_prompt:
        print(f"{instruction}；2 秒后开始采集。")
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            _read_values(glove)
    else:
        input(f"{instruction}，做好后按 Enter 开始采集：")
    glove.discard_pending()


def _collect_stage(
    glove: StretchGloveApiDevice,
    *,
    label: str,
    duration: float,
    display_hz: float,
) -> tuple[np.ndarray, float]:
    print(f"正在采集【{label}】，请保持当前姿势。")
    samples: list[np.ndarray] = []
    started = time.monotonic()
    next_display = started
    while time.monotonic() - started < duration:
        values = _read_values(glove)
        samples.append(values)
        now = time.monotonic()
        if now >= next_display:
            _print_live(values, len(samples))
            next_display = now + 1.0 / display_hz
    print()
    if not samples:
        raise RuntimeError(f"{label}测试没有收到手套样本。")
    return np.stack(samples), time.monotonic() - started


def _diagnose(
    opened: np.ndarray,
    opened_elapsed: float,
    stage_samples: list[tuple[str, int | None, np.ndarray, float]],
) -> bool:
    samples = np.concatenate([opened, *(item[2] for item in stage_samples)], axis=0)
    elapsed = opened_elapsed + sum(item[3] for item in stage_samples)
    physical = samples[:, :5]
    minimum = physical.min(axis=0)
    maximum = physical.max(axis=0)
    spans = maximum - minimum
    std = physical.std(axis=0)
    print("\n测试汇总")
    print(f"接收 {len(samples)} 个样本，用时 {elapsed:.2f}s，约 {len(samples) / elapsed:.1f} Hz")
    print("\n逐项通道对应检查")
    passed = True
    for label, target_index, flexed, _ in stage_samples:
        pose_delta = np.abs(np.median(flexed[:, :5], axis=0) - np.median(opened[:, :5], axis=0))
        if target_index is None:
            minimum_delta = float(pose_delta.min())
            status = "正常" if minimum_delta >= 0.15 else "至少一路变化过小"
            passed &= minimum_delta >= 0.15
            print(f"{label:<6} 五路最小姿势差={minimum_delta:.3f}  {status}")
            continue
        target_delta = float(pose_delta[target_index])
        other_deltas = np.delete(pose_delta, target_index)
        largest_other = float(other_deltas.max())
        if target_delta < 0.15:
            status = "变化过小"
            passed = False
        else:
            status = "正常"
        print(
            f"{label:<6} 目标通道姿势差={target_delta:.3f} "
            f"自然联动最大姿势差={largest_other:.3f}  {status}"
        )
    print("\n全部动态样本统计")
    print("通道         min    max   span    std")
    for index, name in enumerate(CHANNEL_NAMES[:5]):
        print(
            f"{name:<6} {minimum[index]:6.3f} {maximum[index]:6.3f} "
            f"{spans[index]:6.3f} {std[index]:6.3f}"
        )
    duplicate_ok = np.allclose(samples[:, 4], samples[:, 5], atol=1e-6)
    print(f"拇指复制映射：{'正常' if duplicate_ok else '异常'}")
    passed &= bool(duplicate_ok)
    if not passed:
        print("提示：若目标通道变化过小，请检查对应传感器或重新校准后再测试。")
    return passed


def run(args: argparse.Namespace) -> int:
    if not args.mac and not args.serial_port:
        raise ValueError("未配置手套 MAC 或串口；请设置 configs/teleop.json。")
    if args.duration <= 0 or args.display_hz <= 0:
        raise ValueError("--duration and --display-hz must be positive.")
    if not 0.0 <= args.history_weight <= 1.0:
        raise ValueError("--history-weight must be between 0 and 1.")
    _print_preflight(args)
    glove = StretchGloveApiDevice(
        args.mac,
        channel=args.channel,
        serial_port=args.serial_port,
        baudrate=args.baudrate,
        calibration_seconds=args.calibration_seconds,
        calibration_pose_delay_seconds=args.pose_delay_seconds if args.no_prompt else 0.0,
        calibration_confirmation=(
            None
            if args.no_prompt
            else lambda pose: input(f"请{pose}，做好并稳定后按 Enter 开始校准：")
        ),
    )
    try:
        print("\n正在连接；若长时间无响应，请确认系统配对、传输按键和串口配置。")
        glove.connect()
        print("\n连接及校准成功。下面会逐项测试，每个姿势都由你按 Enter 确认后采集。")
        print("不要求目标手指反复运动，也不要求其他手指完全不动，保持自然即可。")
        print("显示顺序是手部控制顺序，最后两项都来自同一个拇指传感器。")
        print("完全张手基线只采集一次，后面的所有动作都与这份基线比较。")
        _confirm_pose(glove, "请自然、完全张开手掌并保持", args.no_prompt)
        opened, opened_elapsed = _collect_stage(
            glove,
            label="全张手基线",
            duration=args.duration,
            display_hz=args.display_hz,
        )
        stages = (("拇指", 4), ("食指", 0), ("中指", 1), ("无名指", 2), ("小指", 3))
        stage_samples: list[tuple[str, int | None, np.ndarray, float]] = []
        for label, target_index in stages:
            _confirm_pose(
                glove,
                f"请舒适地弯曲{label}；允许其他手指自然联动并保持",
                args.no_prompt,
            )
            flexed, flexed_elapsed = _collect_stage(
                glove,
                label=f"{label}-弯曲姿势",
                duration=args.duration,
                display_hz=args.display_hz,
            )
            stage_samples.append((label, target_index, flexed, flexed_elapsed))
        _confirm_pose(glove, "请舒适地完整握拳并保持", args.no_prompt)
        flexed, flexed_elapsed = _collect_stage(
            glove,
            label="完整握拳-握拳姿势",
            duration=args.duration,
            display_hz=args.display_hz,
        )
        stage_samples.append(("完整握拳", None, flexed, flexed_elapsed))
        passed = _diagnose(opened, opened_elapsed, stage_samples)
        if passed and not args.no_save:
            minimum, maximum = glove.calibration_bounds()
            saved_path = save_glove_calibration(
                minimum.tolist(),
                maximum.tolist(),
                history_weight=args.history_weight,
            )
            print(f"校准结果已保存：{saved_path}")
            print(
                f"校准融合权重：历史={args.history_weight:.2f}，"
                f"本次={1.0 - args.history_weight:.2f}"
            )
            print("后续正式遥操作会直接加载该结果，不再要求重新握拳/张手校准。")
        return 0 if passed else 1
    except KeyboardInterrupt:
        print("\n用户中止测试。")
        return 130
    except Exception as exc:
        print(f"\n测试失败：{type(exc).__name__}: {exc}")
        print("检查顺序：Windows 配对 → COM 端口 → 手套发送按键 → baudrate。")
        return 1
    finally:
        glove.close()
        print("蓝牙连接已关闭。")


def main() -> None:
    raise SystemExit(run(_parse_args()))


if __name__ == "__main__":
    main()
