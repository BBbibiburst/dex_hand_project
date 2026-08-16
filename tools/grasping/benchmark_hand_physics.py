"""Report structural and adaptive-closure diagnostics for the Dex Hand MJCF."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from source.assets import DEX_HAND_XML_PATH

ACTUATOR_NAMES = (
    "act_push_0_j",
    "act_push_1_j",
    "act_push_2_j",
    "act_push_3_j",
    "thumb_rotate_act_push_j",
    "thumb_grasp_act_push_j",
)


def _id(model: mujoco.MjModel, object_type: mujoco.mjtObj, name: str) -> int:
    object_id = mujoco.mj_name2id(model, object_type, name)
    if object_id < 0:
        raise RuntimeError(f"MJCF object {name!r} is missing.")
    return object_id


def _joint_qpos(model: mujoco.MjModel, data: mujoco.MjData, name: str) -> float:
    joint_id = _id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    return float(data.qpos[model.jnt_qposadr[joint_id]])


def _settle(
    xml_path: Path,
    controls: np.ndarray,
    steps: int,
    *,
    blocked_finger: int | None = None,
    proximal_limit: float = 0.25,
) -> tuple[mujoco.MjModel, mujoco.MjData]:
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    if blocked_finger is not None:
        joint_id = _id(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            f"finger_first_{blocked_finger}_j",
        )
        model.jnt_range[joint_id, 1] = proximal_limit
    data = mujoco.MjData(model)
    actuator_ids = np.asarray(
        [_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name) for name in ACTUATOR_NAMES]
    )
    data.ctrl[actuator_ids] = controls
    for _ in range(steps):
        mujoco.mj_step(model, data)
    return model, data


def benchmark(xml_path: Path, *, steps: int) -> dict[str, object]:
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    collision_geoms = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
        for geom_id in range(model.ngeom)
        if model.geom_contype[geom_id] or model.geom_conaffinity[geom_id]
    ]
    controls = np.asarray([0.01, 0.01, 0.01, 0.01, 0.004, 0.01])
    closed_model, closed_data = _settle(xml_path, controls, steps)

    adaptation: list[dict[str, float | int]] = []
    for finger in range(4):
        free_proximal = _joint_qpos(
            closed_model,
            closed_data,
            f"finger_first_{finger}_j",
        )
        free_distal = _joint_qpos(
            closed_model,
            closed_data,
            f"finger_second_{finger}_j",
        )
        blocked_model, blocked_data = _settle(
            xml_path,
            controls,
            steps,
            blocked_finger=finger,
        )
        blocked_proximal = _joint_qpos(
            blocked_model,
            blocked_data,
            f"finger_first_{finger}_j",
        )
        blocked_distal = _joint_qpos(
            blocked_model,
            blocked_data,
            f"finger_second_{finger}_j",
        )
        adaptation.append(
            {
                "finger": finger,
                "free_proximal_rad": free_proximal,
                "free_distal_rad": free_distal,
                "free_proximal_deg": float(np.degrees(free_proximal)),
                "free_distal_deg": float(np.degrees(free_distal)),
                "blocked_proximal_rad": blocked_proximal,
                "blocked_distal_rad": blocked_distal,
                "distal_gain_rad": blocked_distal - free_distal,
            }
        )

    actuator_ids = np.asarray(
        [_id(closed_model, mujoco.mjtObj.mjOBJ_ACTUATOR, name) for name in ACTUATOR_NAMES]
    )
    closed_lengths = closed_data.actuator_length[actuator_ids].copy()
    closed_forces = closed_data.actuator_force[actuator_ids].copy()
    settled_max_abs_qvel = float(np.max(np.abs(closed_data.qvel)))
    closed_contact_count = int(closed_data.ncon)
    closed_minimum_contact_distance = min(
        (closed_data.contact[index].dist for index in range(closed_data.ncon)),
        default=0.0,
    )

    closed_data.ctrl[actuator_ids] = 0.0
    maximum_release_contacts = 0
    for _ in range(steps):
        mujoco.mj_step(closed_model, closed_data)
        maximum_release_contacts = max(maximum_release_contacts, closed_data.ncon)

    return {
        "xml": str(xml_path),
        "nq": model.nq,
        "nv": model.nv,
        "nu": model.nu,
        "ntendon": model.ntendon,
        "neq": model.neq,
        "all_actuators_are_tendons": bool(
            np.all(model.actuator_trntype == mujoco.mjtTrn.mjTRN_TENDON)
        ),
        "collision_geoms": collision_geoms,
        "closed_actuator_length_m": closed_lengths.tolist(),
        "closed_actuator_force_n": closed_forces.tolist(),
        "closed_contact_count": closed_contact_count,
        "closed_minimum_contact_distance_m": float(closed_minimum_contact_distance),
        "settled_max_abs_qvel": settled_max_abs_qvel,
        "released_actuator_length_m": closed_data.actuator_length[actuator_ids].tolist(),
        "released_contact_count": int(closed_data.ncon),
        "release_peak_contact_count": int(maximum_release_contacts),
        "released_max_abs_qvel": float(np.max(np.abs(closed_data.qvel))),
        "adaptation": adaptation,
    }


def _print_report(report: dict[str, object]) -> None:
    print(
        "structure: "
        f"nq={report['nq']} nv={report['nv']} nu={report['nu']} "
        f"ntendon={report['ntendon']} neq={report['neq']}"
    )
    print(f"all actuators use tendon transmission: {report['all_actuators_are_tendons']}")
    print(f"active collision geoms ({len(report['collision_geoms'])}):")
    print("  " + ", ".join(report["collision_geoms"]))
    print("full-close actuator force [N]:")
    print("  " + np.array2string(np.asarray(report["closed_actuator_force_n"]), precision=3))
    print("free fist and proximal-block adaptation:")
    for row in report["adaptation"]:
        print(
            f"  finger {row['finger']}: "
            f"free=({row['free_proximal_deg']:.1f} deg, {row['free_distal_deg']:.1f} deg) "
            f"blocked=({row['blocked_proximal_rad']:.3f}, {row['blocked_distal_rad']:.3f}) "
            f"distal_gain={row['distal_gain_rad']:.3f}"
        )
    print(
        "max-close -> zero release: "
        f"contacts={report['released_contact_count']} "
        f"max_length={max(abs(value) for value in report['released_actuator_length_m']):.2e} m "
        f"max_qvel={report['released_max_abs_qvel']:.2e}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xml", type=Path, default=DEX_HAND_XML_PATH)
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    args = parser.parse_args()
    if args.steps <= 0:
        parser.error("--steps must be positive")

    report = benchmark(args.xml.resolve(), steps=args.steps)
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        _print_report(report)
    structurally_valid = (
        report["nq"] == 12
        and report["nu"] == 6
        and report["ntendon"] == 6
        and report["neq"] == 0
        and report["all_actuators_are_tendons"]
        and all(1.25 < row["free_proximal_rad"] < 1.50 for row in report["adaptation"])
        and all(1.40 < row["free_distal_rad"] < 1.65 for row in report["adaptation"])
        and all(row["distal_gain_rad"] > 0.2 for row in report["adaptation"])
        and report["released_contact_count"] == 0
        and max(abs(value) for value in report["released_actuator_length_m"]) < 5e-5
    )
    return 0 if structurally_valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
