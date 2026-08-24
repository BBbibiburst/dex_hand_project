"""Visualize every stage of a generated grasp execution episode.

The report explains the optimized candidate in object coordinates. The MuJoCo
viewer replays the recorded qpos/qvel exactly and draws target contact points
in the object's *current* frame, so markers remain attached if the object moves.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import time

import mujoco
import numpy as np

from source.envs.manipulation import make_lift_env
from source.grasping.catalog import load_object_geometry
from source.grasping.contracts import DemonstrationEpisode
from source.grasping.executor import _contact_digit
from source.viz.overlays import (
    clear_markers,
    draw_label,
    draw_line_marker,
    draw_pose_frame,
    draw_sphere_marker,
)


DEFAULT_STAGE_NAMES = {
    0: "settle",
    1: "transit",
    2: "pregrasp",
    3: "approach",
    4: "close",
    5: "hold",
    6: "lift",
    7: "verify",
}
CONTACT_COLORS = (
    (1.0, 0.2, 0.2, 1.0),
    (0.2, 0.8, 1.0, 1.0),
    (0.2, 1.0, 0.35, 1.0),
    (1.0, 0.75, 0.15, 1.0),
    (0.8, 0.25, 1.0, 1.0),
)
DIGIT_NAMES = ("little", "ring", "middle", "index", "thumb")


def _actual_skin_contacts(env) -> list[tuple[int, np.ndarray]]:
    """Return current object--DexHand skin contacts as (digit, world point)."""
    binding = env.task._require_bindings().objects["object"]
    object_geoms = set(int(value) for value in binding.geom_ids)
    contacts: list[tuple[int, np.ndarray]] = []
    for index in range(env.data.ncon):
        contact = env.data.contact[index]
        if int(contact.geom1) in object_geoms:
            hand_geom = int(contact.geom2)
        elif int(contact.geom2) in object_geoms:
            hand_geom = int(contact.geom1)
        else:
            continue
        digit = _contact_digit(env, hand_geom)
        if digit >= 0:
            contacts.append((digit, np.asarray(contact.pos, dtype=np.float64).copy()))
    return contacts


def _stage_names(episode: DemonstrationEpisode) -> dict[int, str]:
    stored = episode.metadata.get("stage_codes", {})
    names = {int(value): str(name) for name, value in stored.items()}
    return names or DEFAULT_STAGE_NAMES


def _quat_matrix(quaternion_wxyz: np.ndarray) -> np.ndarray:
    matrix = np.empty(9, dtype=np.float64)
    mujoco.mju_quat2Mat(matrix, np.asarray(quaternion_wxyz, dtype=np.float64))
    return matrix.reshape(3, 3)


def contact_points_world(
    episode: DemonstrationEpisode,
    frame: int,
) -> tuple[np.ndarray, np.ndarray]:
    rotation = _quat_matrix(episode.arrays["object_quaternion_wxyz"][frame])
    position = np.asarray(episode.arrays["object_position"][frame], dtype=np.float64)
    points = np.asarray(episode.candidate.contact_points, dtype=np.float64)
    normals = np.asarray(episode.candidate.contact_normals, dtype=np.float64)
    return points @ rotation.T + position, normals @ rotation.T


def save_report(episode: DemonstrationEpisode, output: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    geometry = load_object_geometry(
        episode.object_id,
        surface_points=2048,
        seed=episode.seed,
    )
    stages = np.asarray(episode.arrays["stage"], dtype=np.int64)
    names = _stage_names(episode)
    object_z = np.asarray(episode.arrays["object_position"], dtype=np.float64)[:, 2]
    baseline = float(np.median(object_z[stages == 0])) if np.any(stages == 0) else float(object_z[0])
    lift_mm = 1000.0 * (object_z - baseline)
    candidate = episode.candidate

    figure = plt.figure(figsize=(15, 13), constrained_layout=True)
    grid = figure.add_gridspec(3, 2)
    ax_object = figure.add_subplot(grid[0, 0], projection="3d")
    vertices = geometry.vertices
    stride = max(1, len(vertices) // 3000)
    ax_object.scatter(
        vertices[::stride, 0] * 1000.0,
        vertices[::stride, 1] * 1000.0,
        vertices[::stride, 2] * 1000.0,
        s=2,
        c="#b8bec7",
        alpha=0.35,
        label="convex collision surface",
    )
    for index, (point, normal) in enumerate(
        zip(candidate.contact_points, candidate.contact_normals, strict=True)
    ):
        color = CONTACT_COLORS[index % len(CONTACT_COLORS)]
        point_mm = 1000.0 * point
        normal_mm = 14.0 * normal
        ax_object.scatter(*point_mm, s=70, color=color, depthshade=False)
        ax_object.quiver(*point_mm, *normal_mm, color=color, linewidth=2)
        ax_object.text(*point_mm, f" C{index}")
    ax_object.set_title(f"{candidate.backend} target contacts in object frame")
    ax_object.set_xlabel("x [mm]")
    ax_object.set_ylabel("y [mm]")
    ax_object.set_zlabel("z [mm]")
    ax_object.set_box_aspect(np.maximum(np.ptp(vertices, axis=0), 1e-6))

    ax_height = figure.add_subplot(grid[0, 1])
    ax_height.plot(lift_mm, color="#1464b4", linewidth=2)
    for code in np.unique(stages):
        indices = np.flatnonzero(stages == code)
        ax_height.axvspan(indices[0], indices[-1], alpha=0.10, label=names.get(int(code), str(code)))
    ax_height.axhline(65.0, color="#d62728", linestyle="--", label="65 mm success threshold")
    ax_height.set_title(f"Object lift (maximum {float(lift_mm.max()):.1f} mm)")
    ax_height.set_xlabel("recorded frame")
    ax_height.set_ylabel("height above settled pose [mm]")
    ax_height.grid(alpha=0.25)
    ax_height.legend(ncol=3, fontsize=8)

    ax_contacts = figure.add_subplot(grid[1, :])
    recorded_contacts = np.asarray(
        episode.arrays.get("robot_object_digit_contact_count", np.zeros((len(stages), 5)))
    )
    for digit, label in enumerate(DIGIT_NAMES):
        ax_contacts.step(
            np.arange(len(stages)),
            recorded_contacts[:, digit] > 0,
            where="post",
            label=label,
            color=CONTACT_COLORS[digit],
            linewidth=2,
        )
    opposed = (recorded_contacts[:, 4] > 0) & (recorded_contacts[:, :4].max(axis=1) > 0)
    ax_contacts.fill_between(
        np.arange(len(stages)), 0, 1, where=opposed, color="#42b883", alpha=0.15,
        label="thumb + opposing finger",
    )
    ax_contacts.set_ylim(-0.05, 1.15)
    ax_contacts.set_yticks((0, 1), ("no contact", "skin contact"))
    ax_contacts.set_xlabel("recorded frame")
    ax_contacts.set_title("Actual MuJoCo object contacts (skin only)")
    ax_contacts.grid(alpha=0.2)
    ax_contacts.legend(ncol=6, fontsize=8)

    ax_hand = figure.add_subplot(grid[2, 0])
    labels = ("index", "middle", "ring", "little", "thumb rotate", "thumb close")
    colors = [CONTACT_COLORS[index % len(CONTACT_COLORS)] for index in range(6)]
    ax_hand.barh(labels, candidate.actuator_fractions, color=colors)
    ax_hand.set_xlim(0.0, 1.0)
    ax_hand.set_xlabel("normalized actuator fraction")
    ax_hand.set_title(f"{candidate.backend} closed-hand target")
    for index, value in enumerate(candidate.actuator_fractions):
        ax_hand.text(float(value) + 0.01, index, f"{float(value):.2f}", va="center")

    ax_text = figure.add_subplot(grid[2, 1])
    ax_text.axis("off")
    metrics = "\n".join(f"  {key}: {value:.5g}" for key, value in candidate.metrics.items())
    distances = ", ".join(f"{1000.0 * value:.2f}" for value in candidate.contact_distances)
    ax_text.text(
        0.0,
        1.0,
        "\n".join(
            (
                f"object: {episode.object_id}",
                f"candidate seed: {candidate.seed_index}",
                f"episode success: {episode.success}",
                f"terminal stage: {episode.terminal_stage}",
                f"failure: {episode.failure_reason}",
                f"target distances [mm]: {distances}",
                "metrics:",
                metrics,
            )
        ),
        va="top",
        family="monospace",
        fontsize=10,
    )
    figure.suptitle(f"{candidate.backend} candidate and complete execution", fontsize=16)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)
    return output


def play_episode(
    episode: DemonstrationEpisode,
    *,
    speed: float,
    loop: bool,
    marker_radius: float,
) -> None:
    from mujoco import viewer

    if speed <= 0.0:
        raise ValueError("viewer speed must be positive")
    env = make_lift_env(
        task_config={"object_id": episode.object_id, "terminate_on_success": False},
        control_mode="ik",
        enable_tactile_sensors=False,
        render_mode=None,
        episode_length=max(len(episode.arrays["qpos"]) + 10, 500),
    )
    try:
        env.reset(seed=episode.seed)
        if episode.arrays["qpos"].shape[1:] != env.data.qpos.shape:
            raise ValueError("Recorded qpos does not match the current MuJoCo model")
        handle = viewer.launch_passive(env.model, env.data)
        try:
            control_dt = float(env.config.control_dt)
            stage_names = _stage_names(episode)
            while handle.is_running():
                previous_stage = None
                for frame in range(len(episode.arrays["qpos"])):
                    if not handle.is_running():
                        break
                    started = time.monotonic()
                    env.data.qpos[:] = episode.arrays["qpos"][frame]
                    env.data.qvel[:] = episode.arrays["qvel"][frame]
                    if episode.arrays["ctrl"].shape[1:] == env.data.ctrl.shape:
                        env.data.ctrl[:] = episode.arrays["ctrl"][frame]
                    mujoco.mj_forward(env.model, env.data)
                    points, normals = contact_points_world(episode, frame)
                    clear_markers(handle)
                    for index, (point, normal) in enumerate(zip(points, normals, strict=True)):
                        color = CONTACT_COLORS[index % len(CONTACT_COLORS)]
                        draw_sphere_marker(handle, point, radius=marker_radius, rgba=color)
                        draw_line_marker(
                            handle,
                            point,
                            point + 0.025 * normal,
                            width=0.002,
                            rgba=color,
                        )
                    actual_contacts = _actual_skin_contacts(env)
                    actual_digits = set()
                    for digit, point in actual_contacts:
                        actual_digits.add(digit)
                        draw_sphere_marker(
                            handle,
                            point,
                            radius=0.65 * marker_radius,
                            rgba=(1.0, 1.0, 1.0, 1.0),
                        )
                    stage = int(episode.arrays["stage"][frame])
                    stage_name = stage_names.get(stage, str(stage))
                    object_position = episode.arrays["object_position"][frame]
                    draw_label(
                        handle,
                        object_position + np.asarray([0.0, 0.0, 0.13]),
                        f"{frame:04d}  {stage_name}  actual skin: "
                        + (", ".join(DIGIT_NAMES[digit] for digit in sorted(actual_digits)) or "none"),
                    )
                    if "grasp_ee_position" in episode.metadata:
                        draw_pose_frame(
                            handle,
                            np.asarray(episode.metadata["grasp_ee_position"]),
                            np.asarray(episode.metadata["grasp_ee_quaternion_wxyz"]),
                            axis_length=0.045,
                        )
                    handle.sync()
                    if stage != previous_stage:
                        print(f"[stage] frame={frame} {stage_name}", flush=True)
                        previous_stage = stage
                    time.sleep(max(0.0, control_dt / speed - (time.monotonic() - started)))
                if not loop:
                    break
        finally:
            handle.close()
    finally:
        env.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--play", action="store_true")
    parser.add_argument("--viewer-speed", type=float, default=0.4)
    parser.add_argument("--marker-radius", type=float, default=0.006)
    parser.add_argument("--loop", action="store_true")
    args = parser.parse_args()
    episode = DemonstrationEpisode.load(args.manifest)
    report = args.report or args.manifest.parent / "pipeline_report.png"
    print(f"[report] {save_report(episode, report)}")
    if args.play:
        play_episode(
            episode,
            speed=args.viewer_speed,
            loop=args.loop,
            marker_radius=args.marker_radius,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
