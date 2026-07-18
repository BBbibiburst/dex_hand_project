"""Automated per-taxel contact, hold, crosstalk, and release validation."""

from __future__ import annotations

import argparse
import csv
import time
from dataclasses import asdict, dataclass
from pathlib import Path
import mujoco
import numpy as np

from source.demos.common import add_robot_config_args, load_demo_robot_config
from source.demos.tactile_probe_demo import (
    PROBE_GEOM_NAME,
    PROBE_JOINT_NAME,
    _add_probe_to_spec,
)
from source.robots.builder import build_robot_spec
from source.robots.config import descriptors_from_robot_config, optional_tuple
from source.sensors.base import TactileSensorBase, TactileSiteRef


@dataclass(frozen=True)
class TaxelResult:
    index: int
    patch: str
    site: str
    status: str
    contacted: bool
    instant: float
    hold_median: float
    hold_min: float
    peak_other: float
    peak_other_index: int
    crosstalk_ratio: float
    release: float
    contact_step: int
    x: float
    y: float
    z: float


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Automatically press and validate every site-based tactile taxel."
    )
    parser.add_argument("--backend", default=None)
    parser.add_argument("--patch", default="", help="Test only one backend-provided patch.")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument(
        "--max-taxels", type=int, default=0, help="Maximum taxels to test; 0 means all."
    )
    parser.add_argument("--probe-radius", type=float, default=0.0004)
    parser.add_argument("--clearance", type=float, default=0.001)
    parser.add_argument("--scan-step", type=float, default=0.0002)
    parser.add_argument("--max-scan-depth", type=float, default=0.006)
    parser.add_argument("--penetration", type=float, default=0.0001)
    parser.add_argument("--instant-steps", type=int, default=2)
    parser.add_argument("--hold-steps", type=int, default=20)
    parser.add_argument("--release-steps", type=int, default=5)
    parser.add_argument("--min-response", type=float, default=1e-6)
    parser.add_argument("--min-hold-ratio", type=float, default=0.5)
    parser.add_argument("--max-crosstalk-ratio", type=float, default=0.25)
    parser.add_argument("--max-release", type=float, default=1e-6)
    parser.add_argument("--csv", type=Path, default=None, help="Optional raw metrics CSV.")
    parser.add_argument(
        "--plot",
        type=Path,
        default=None,
        help="Optionally save the 3-D status point cloud to this image.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not open the interactive point-cloud window.",
    )
    parser.add_argument("--progress-interval", type=int, default=25)
    add_robot_config_args(parser, include_tactile_toggle=False)
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    positive = ("probe_radius", "clearance", "scan_step", "max_scan_depth")
    for name in positive:
        if float(getattr(args, name)) <= 0.0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive.")
    if args.penetration < 0.0:
        raise ValueError("--penetration must be non-negative.")
    for name in ("instant_steps", "hold_steps", "release_steps"):
        if int(getattr(args, name)) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be at least 1.")
    if args.start_index < 0 or args.max_taxels < 0:
        raise ValueError("--start-index and --max-taxels must be non-negative.")


def _build(args: argparse.Namespace):
    config = load_demo_robot_config(args)
    arm, hand, base = descriptors_from_robot_config(config)
    if hand.tactile_sensor_factory is None:
        raise ValueError(f"End effector {hand.name!r} has no tactile backend.")
    sensor = hand.tactile_sensor_factory(
        args.backend or str(config.get("tactile_backend", "simple_box")),
        **dict(config.get("tactile_options") or {}),
    )
    refs = tuple(sensor.visualization_sites())
    if not refs:
        raise ValueError(f"Backend {type(sensor).__name__!r} exposes no site-based taxels.")
    spec = build_robot_spec(
        arm_descriptor=arm,
        hand_descriptor=hand,
        base_descriptor=base,
        rot_xyz_deg=optional_tuple(config, "hand_attach_rot_xyz_deg"),
        attach_point_name=config.get("attach_point_name"),
        base_mount_site_name=config.get("base_mount_site_name"),
        hand_prefix=config.get("hand_prefix"),
        tactile_sensor=sensor,
        add_tactile_sensors=True,
    )
    _add_probe_to_spec(
        spec,
        radius=args.probe_radius,
        initial_pos=np.asarray([0.0, 0.0, -10.0]),
        gravity_comp=True,
    )
    model = spec.compile()
    data = mujoco.MjData(model)
    sensor.bind(model, data)
    sensor.reset(model, data, rng=np.random.default_rng(0), options=None)
    mujoco.mj_forward(model, data)
    return model, data, sensor, refs


def _joint_addresses(model: mujoco.MjModel) -> tuple[int, int]:
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, PROBE_JOINT_NAME)
    return int(model.jnt_qposadr[joint_id]), int(model.jnt_dofadr[joint_id])


def _place_probe(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    position: np.ndarray,
    addresses: tuple[int, int],
) -> None:
    qpos_adr, qvel_adr = addresses
    data.qpos[qpos_adr : qpos_adr + 3] = position
    data.qpos[qpos_adr + 3 : qpos_adr + 7] = [1.0, 0.0, 0.0, 0.0]
    data.qvel[qvel_adr : qvel_adr + 6] = 0.0
    mujoco.mj_forward(model, data)
    mujoco.mj_step(model, data)


def _touches_body(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    probe_geom_id: int,
    body_id: int,
) -> bool:
    for index in range(data.ncon):
        contact = data.contact[index]
        if contact.geom1 == probe_geom_id:
            other = int(contact.geom2)
        elif contact.geom2 == probe_geom_id:
            other = int(contact.geom1)
        else:
            continue
        if int(model.geom_bodyid[other]) == body_id:
            return True
    return False


def _signals(sensor: TactileSensorBase, model, data) -> np.ndarray:
    return np.asarray(sensor.diagnostic_values(model, data), dtype=np.float64).reshape(-1)


def _sample_metrics(values: np.ndarray, target_index: int) -> tuple[float, float, int]:
    target = float(values[target_index])
    if values.size <= 1:
        return target, 0.0, -1
    other_values = values.copy()
    other_values[target_index] = -np.inf
    peak_index = int(np.argmax(other_values))
    return target, float(max(other_values[peak_index], 0.0)), peak_index


def _test_taxel(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    sensor: TactileSensorBase,
    ref: TactileSiteRef,
    args: argparse.Namespace,
    addresses: tuple[int, int],
    probe_geom_id: int,
) -> TaxelResult:
    full_name = getattr(sensor, "name_prefix", "") + ref.name
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, full_name)
    if site_id < 0:
        raise ValueError(f"Missing tactile site {full_name!r}.")
    position = np.asarray(data.site_xpos[site_id], dtype=np.float64).copy()
    normal = np.asarray(data.site_xmat[site_id], dtype=np.float64).reshape(3, 3)[:, 2]
    half_depth = float(model.site_size[site_id, 2])
    body_id = int(model.site_bodyid[site_id])
    start_distance = half_depth + args.probe_radius + args.clearance
    outside = position + start_distance * normal
    _place_probe(model, data, outside, addresses)

    contacted = False
    contact_step = -1
    contact_position = outside
    scan_steps = int(np.ceil(args.max_scan_depth / args.scan_step))
    for step in range(1, scan_steps + 1):
        candidate = outside - step * args.scan_step * normal
        _place_probe(model, data, candidate, addresses)
        if _touches_body(model, data, probe_geom_id, body_id):
            contacted = True
            contact_step = step
            contact_position = candidate
            break

    if not contacted:
        return TaxelResult(
            ref.flat_index,
            ref.patch,
            ref.name,
            "inactive",
            False,
            0.0,
            0.0,
            0.0,
            0.0,
            -1,
            0.0,
            0.0,
            -1,
            float(position[0]),
            float(position[1]),
            float(position[2]),
        )

    pressed = contact_position - args.penetration * normal
    instant_targets: list[float] = []
    peak_other = 0.0
    peak_other_index = -1
    for _ in range(args.instant_steps):
        _place_probe(model, data, pressed, addresses)
        target, other, other_index = _sample_metrics(_signals(sensor, model, data), ref.flat_index)
        instant_targets.append(target)
        if other > peak_other:
            peak_other = other
            peak_other_index = other_index
    instant = max(instant_targets)

    hold_targets = []
    for _ in range(args.hold_steps):
        _place_probe(model, data, pressed, addresses)
        target, other, other_index = _sample_metrics(_signals(sensor, model, data), ref.flat_index)
        hold_targets.append(target)
        if other > peak_other:
            peak_other = other
            peak_other_index = other_index

    release = 0.0
    for _ in range(args.release_steps):
        _place_probe(model, data, outside, addresses)
        release = max(release, float(_signals(sensor, model, data)[ref.flat_index]))

    hold_median = float(np.median(hold_targets))
    hold_min = float(np.min(hold_targets))
    denominator = max(instant, hold_median, args.min_response)
    crosstalk_ratio = peak_other / denominator
    if instant < args.min_response:
        status = "no_response"
    elif hold_median < args.min_response or hold_min < args.min_hold_ratio * instant:
        status = "unstable_hold"
    elif crosstalk_ratio > args.max_crosstalk_ratio:
        status = "crosstalk"
    elif release > args.max_release:
        status = "release_residual"
    else:
        status = "pass"
    return TaxelResult(
        ref.flat_index,
        ref.patch,
        ref.name,
        status,
        True,
        instant,
        hold_median,
        hold_min,
        peak_other,
        peak_other_index,
        crosstalk_ratio,
        release,
        contact_step,
        float(position[0]),
        float(position[1]),
        float(position[2]),
    )


def _write_csv(path: Path, results: list[TaxelResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(TaxelResult.__dataclass_fields__))
        writer.writeheader()
        writer.writerows(asdict(result) for result in results)


def _plot_results(path: Path | None, results: list[TaxelResult], *, show: bool) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    colors = {
        "pass": "#2ecc71",
        "inactive": "#8c939a",
        "no_response": "#e53935",
        "unstable_hold": "#fb8c00",
        "crosstalk": "#8e24aa",
        "release_residual": "#1e88e5",
    }
    positions = np.asarray([(result.x, result.y, result.z) for result in results])
    instant = np.asarray([result.instant for result in results], dtype=np.float64)
    positive = instant[instant > 0.0]
    scale = float(np.percentile(positive, 90)) if positive.size else 1.0
    sizes = 10.0 + 36.0 * np.sqrt(np.clip(instant / max(scale, 1e-12), 0.0, 1.0))

    figure = plt.figure(figsize=(11.5, 8.0))
    axis = figure.add_subplot(111, projection="3d")
    for status in colors:
        mask = np.asarray([result.status == status for result in results])
        if not np.any(mask):
            continue
        axis.scatter(
            positions[mask, 0],
            positions[mask, 1],
            positions[mask, 2],
            s=sizes[mask],
            c=colors[status],
            alpha=0.88,
            edgecolors="black",
            linewidths=0.15,
            depthshade=False,
        )
    counts = {status: sum(result.status == status for result in results) for status in colors}
    tested = len(results) - counts["inactive"]
    pass_rate = 100.0 * counts["pass"] / max(tested, 1)
    summary = "   ".join(f"{status}={count}" for status, count in counts.items() if count)
    axis.set_title(
        f"Tactile taxel validation point cloud\n{summary}   active pass rate={pass_rate:.2f}%"
    )
    axis.set_xlabel("world X (m)")
    axis.set_ylabel("world Y (m)")
    axis.set_zlabel("world Z (m)")
    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markerfacecolor=color,
            markeredgecolor="black",
            label=status,
        )
        for status, color in colors.items()
        if counts[status]
    ]
    axis.legend(handles=handles, loc="upper left")
    axis.view_init(elev=24, azim=-58)
    figure.tight_layout()
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(path, dpi=220, bbox_inches="tight")
        print(f"Point cloud: {path.resolve()}")
    if show:
        plt.show()
    plt.close(figure)


def run(args: argparse.Namespace) -> int:
    _validate_args(args)
    model, data, sensor, all_refs = _build(args)
    refs = [ref for ref in all_refs if not args.patch or ref.patch == args.patch]
    if not refs:
        raise ValueError(
            f"No taxels matched patch {args.patch!r}; available: "
            f"{sorted({ref.patch for ref in all_refs})}."
        )
    refs = refs[args.start_index :]
    if args.max_taxels:
        refs = refs[: args.max_taxels]
    addresses = _joint_addresses(model)
    probe_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, PROBE_GEOM_NAME)
    results: list[TaxelResult] = []
    started = time.perf_counter()
    for count, ref in enumerate(refs, start=1):
        result = _test_taxel(model, data, sensor, ref, args, addresses, probe_geom_id)
        results.append(result)
        if count == 1 or count % max(1, args.progress_interval) == 0 or count == len(refs):
            elapsed = time.perf_counter() - started
            rate = count / max(elapsed, 1e-9)
            print(
                f"[{count}/{len(refs)}] {ref.patch}/{ref.name}: {result.status}; "
                f"{rate:.2f} taxels/s"
            )
    if args.csv is not None:
        _write_csv(args.csv, results)
    _plot_results(args.plot, results, show=not args.no_show)
    counts = {
        status: sum(result.status == status for result in results)
        for status in sorted({r.status for r in results})
    }
    print(f"Results: {counts}")
    if args.csv is not None:
        print(f"CSV: {args.csv.resolve()}")
    failures = len(results) - counts.get("pass", 0) - counts.get("inactive", 0)
    return 1 if failures else 0


def main() -> None:
    raise SystemExit(run(_parse_args()))


if __name__ == "__main__":
    main()
