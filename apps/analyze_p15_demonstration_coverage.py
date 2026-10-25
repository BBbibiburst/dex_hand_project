"""Audit P15 grasp-mode diversity and coverage of P14 BC divergence states."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

import matplotlib.pyplot as plt
import mujoco
import numpy as np

from apps.trace_diffusion_rollout import _make_env


APPROACH_START = 137
REGIONS = {
    "arm_1_2": np.arange(1, 3),
    "wrist_5_20": np.arange(5, 21),
    "hand_37_40": np.arange(37, 41),
}
STATE_SCALES = np.r_[np.full(3, 0.005), np.full(7, 0.05), np.full(6, 0.001)]
ACTION_SCALES = np.r_[np.full(7, 0.05), np.full(6, 0.001)]


def _first_contacts(counts: np.ndarray) -> list[int | None]:
    result = []
    for digit in range(counts.shape[1]):
        indices = np.flatnonzero(counts[:, digit] > 0)
        result.append(None if not len(indices) else int(indices[0]))
    return result


def _load_episode(manifest: Path) -> tuple[dict, dict[str, np.ndarray]]:
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    with np.load(manifest.parent / payload["arrays"], allow_pickle=False) as archive:
        arrays = {name: np.asarray(archive[name]) for name in archive.files}
    return payload, arrays


def _task_state(env, arrays: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    wrist, hand = [], []
    hand_ids = env.controller.hand_controller.actuator_ids
    for qpos in arrays["qpos"]:
        env.data.qpos[:] = qpos
        env.data.qvel[:] = 0.0
        mujoco.mj_forward(env.model, env.data)
        wrist.append(env.data.site_xpos[env.controller.arm_controller.site_id].copy())
        hand.append(env.data.actuator_length[hand_ids].copy())
    return {"wrist": np.asarray(wrist), "hand_actual": np.asarray(hand)}


def _normalized_distance(query: np.ndarray, candidates: np.ndarray, scales: np.ndarray):
    values = np.sqrt(np.mean(((candidates - query) / scales) ** 2, axis=1))
    index = int(np.argmin(values))
    return float(values[index]), index


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p15-root", type=Path, required=True)
    parser.add_argument("--p14-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    modes: dict[str, list[dict]] = defaultdict(list)
    failures = []
    shared_env = None
    for summary_path in sorted((args.p15_root / "candidates").glob("*/summary.json")):
        mode = summary_path.parent.name
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        for root_id, row in enumerate(summary["results"]):
            record = {"mode": mode, "root_id": root_id, **row}
            if not row["strict_success"]:
                failures.append(record)
                continue
            manifest = Path(row["manifest"])
            payload, arrays = _load_episode(manifest)
            if shared_env is None:
                shared_env = _make_env(payload, len(arrays["qpos"]) + 20)
            state = _task_state(shared_env, arrays)
            start = int(np.flatnonzero(arrays["stage"] == 3)[0])
            digit = arrays["robot_object_digit_contact_count"][start:]
            modes[mode].append(
                {
                    **record,
                    "payload": payload,
                    "arrays": arrays,
                    "wrist": state["wrist"],
                    "hand_actual": state["hand_actual"],
                    "approach_start": start,
                    "first_contact_per_digit": _first_contacts(digit),
                    "digit_contact_fraction": np.mean(digit > 0, axis=0),
                }
            )

    if shared_env is not None:
        shared_env.close()

    mode_rows = []
    for mode, rows in sorted(modes.items()):
        payload = rows[0]["payload"]
        candidate = payload["candidate"]
        first = np.asarray([row["first_contact_per_digit"] for row in rows], dtype=float)
        fractions = np.asarray([row["digit_contact_fraction"] for row in rows])
        mode_rows.append(
            {
                "mode": mode,
                "final_verified": len(rows),
                "independent_roots": len({row["root_id"] for row in rows}),
                "source_manifest": rows[0]["source_manifest"],
                "candidate_seed_index": candidate["seed_index"],
                "candidate_translation": candidate["hand_translation"],
                "candidate_rotation_matrix": candidate["hand_rotation_matrix"],
                "median_first_contact_per_digit": np.median(first, axis=0).tolist(),
                "mean_digit_contact_fraction": np.mean(fractions, axis=0).tolist(),
                "mean_final_lift_m": float(
                    np.mean([row["strict_metrics"]["final_lift"] for row in rows])
                ),
            }
        )

    # Pairwise mode differences use task-space candidate geometry and observed contacts.
    pairwise = []
    for first_index, first in enumerate(mode_rows):
        for second in mode_rows[first_index + 1 :]:
            first_rotation = np.asarray(first["candidate_rotation_matrix"])
            second_rotation = np.asarray(second["candidate_rotation_matrix"])
            cosine = np.clip((np.trace(first_rotation.T @ second_rotation) - 1.0) / 2.0, -1, 1)
            contact_delta = np.asarray(second["median_first_contact_per_digit"]) - np.asarray(
                first["median_first_contact_per_digit"]
            )
            position_delta = float(
                np.linalg.norm(
                    np.asarray(second["candidate_translation"])
                    - np.asarray(first["candidate_translation"])
                )
            )
            orientation_delta = float(np.arccos(cosine))
            distinct = bool(
                (position_delta >= 0.015 or orientation_delta >= np.deg2rad(10.0))
                and np.max(np.abs(contact_delta)) >= 5.0
            )
            pairwise.append(
                {
                    "first": first["mode"],
                    "second": second["mode"],
                    "candidate_position_delta_m": position_delta,
                    "candidate_orientation_delta_rad": orientation_delta,
                    "median_first_contact_delta_steps": contact_delta.tolist(),
                    "distinct_grasp_mode": distinct,
                    "rule": "pose delta >=15mm or >=10deg, and one digit contact differs >=5 steps",
                }
            )

    # Compare P14 BC states against the original expert and all successful P15 modes.
    coverage_rows = []
    for root_id in range(5):
        with np.load(
            args.p14_root / "divergence" / f"root_{root_id:02d}" / "alignment.npz"
        ) as archive:
            old = {name: np.asarray(archive[name]) for name in archive.files}
        bc_state = np.c_[old["bc_wrist"], old["bc_arm_qpos"], old["bc_hand_actual"]]
        old_state = np.c_[old["expert_wrist"], old["expert_arm_qpos"], old["expert_hand_actual"]]
        bc_action = np.c_[old["bc_arm_target"], old["bc_hand_target"]]
        old_action = np.c_[old["expert_arm_target"], old["expert_hand_target"]]
        new_pool = []
        for mode, rows in modes.items():
            matches = [row for row in rows if row["root_id"] == root_id]
            for row in matches:
                start = row["approach_start"]
                stop = start + len(bc_state)
                new_pool.append(
                    {
                        "mode": mode,
                        "state": np.c_[
                            row["wrist"][start:stop],
                            row["arrays"]["qpos"][start:stop, :7],
                            row["hand_actual"][start:stop],
                        ],
                        "action": row["arrays"]["ctrl"][start:stop],
                    }
                )
        for region, indices in REGIONS.items():
            old_distances, new_distances = [], []
            old_action_delta, new_action_delta = [], []
            nearest_modes = []
            for query_index in indices:
                old_distance, old_index_local = _normalized_distance(
                    bc_state[query_index], old_state[indices], STATE_SCALES
                )
                candidates = np.concatenate([row["state"][indices] for row in new_pool])
                actions = np.concatenate([row["action"][indices] for row in new_pool])
                labels = [
                    row["mode"]
                    for row in new_pool
                    for _ in range(len(indices))
                ]
                new_distance, new_index = _normalized_distance(
                    bc_state[query_index], candidates, STATE_SCALES
                )
                old_target = old_action[indices[old_index_local]]
                old_action_distance = float(
                    np.sqrt(np.mean(((old_target - bc_action[query_index]) / ACTION_SCALES) ** 2))
                )
                new_action_distance = float(
                    np.sqrt(
                        np.mean(((actions[new_index] - bc_action[query_index]) / ACTION_SCALES) ** 2)
                    )
                )
                old_distances.append(old_distance)
                new_distances.append(new_distance)
                old_action_delta.append(old_action_distance)
                new_action_delta.append(new_action_distance)
                nearest_modes.append(labels[new_index])
            old_mean = float(np.mean(old_distances))
            new_mean = float(np.mean(new_distances))
            action_limit = max(float(np.mean(old_action_delta)) * 1.5, 1.0)
            meaningful = bool(
                new_mean <= 0.9 * old_mean
                and float(np.mean(new_action_delta)) <= action_limit
            )
            coverage_rows.append(
                {
                    "root_id": root_id,
                    "region": region,
                    "old_state_distance": old_mean,
                    "new_state_distance": new_mean,
                    "new_over_old": new_mean / max(old_mean, 1e-12),
                    "old_action_delta": float(np.mean(old_action_delta)),
                    "new_action_delta": float(np.mean(new_action_delta)),
                    "nearest_modes": sorted(set(nearest_modes)),
                    "meaningful_improvement": meaningful,
                }
            )

    roots_improved = sorted(
        {
            row["root_id"]
            for row in coverage_rows
            if row["meaningful_improvement"]
        }
    )
    split = {"train": [0, 1, 2, 3, 4], "validation": [5], "test": [6, 7]}
    split_sets = [set(values) for values in split.values()]
    leakage_free = all(
        split_sets[first].isdisjoint(split_sets[second])
        for first in range(len(split_sets))
        for second in range(first + 1, len(split_sets))
    )
    aggregate = {
        "schema_version": 1,
        "candidate_budget": 24,
        "executed_candidates": sum(len(row["results"]) for row in [
            json.loads(path.read_text())
            for path in sorted((args.p15_root / "candidates").glob("*/summary.json"))
        ]),
        "final_verified": sum(len(rows) for rows in modes.values()),
        "failure_count": len(failures),
        "failure_categories": {
            category: sum(row["failure_category"] == category for row in failures)
            for category in sorted({row["failure_category"] for row in failures})
        },
        "modes": mode_rows,
        "pairwise_mode_analysis": pairwise,
        "distinct_successful_mode_pairs": sum(
            row["distinct_grasp_mode"] for row in pairwise
        ),
        "coverage": coverage_rows,
        "roots_with_meaningful_improvement": roots_improved,
        "coverage_gate_3_of_5": len(roots_improved) >= 3,
        "proposed_root_split": split,
        "root_split_leakage_free": leakage_free,
        "split_warning": (
            "P14 diagnostic roots 0-4 overlap proposed training roots; they cannot remain an "
            "independent post-training test set. Only roots 6-7 are held out, below the roadmap's "
            "five-root evaluation target."
        ),
    }
    (args.output / "summary.json").write_text(
        json.dumps(aggregate, indent=2) + "\n", encoding="utf-8"
    )
    (args.output / "successful_manifests.json").write_text(
        json.dumps(
            {
                mode: [row["manifest"] for row in rows]
                for mode, rows in sorted(modes.items())
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    digits = ["index", "middle", "ring", "little", "thumb"]
    figure, axis = plt.subplots(figsize=(9, 4.8))
    width = 0.8 / max(len(mode_rows), 1)
    x = np.arange(5)
    for index, row in enumerate(mode_rows):
        axis.bar(
            x + (index - (len(mode_rows) - 1) / 2) * width,
            row["median_first_contact_per_digit"],
            width,
            label=row["mode"],
        )
    axis.set_xticks(x, digits)
    axis.set_ylabel("First contact step from Approach start")
    axis.set_title("P15 grasp-mode contact timing")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(args.output / "grasp_mode_contact_timing.png", dpi=180)
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.4))
    for mode, rows in sorted(modes.items()):
        representative = rows[0]
        start = representative["approach_start"]
        action = representative["arrays"]["action"][start:]
        axes[0].plot(action[:, 0], action[:, 1], label=mode)
        axes[1].plot(np.mean(action[:, 7:], axis=1), label=mode)
    axes[0].set(
        xlabel="EE target x (m)",
        ylabel="EE target y (m)",
        title="Approach-to-Verify target path",
    )
    axes[1].set(
        xlabel="Step from Approach start",
        ylabel="Mean hand6 target (m)",
        title="Hand closure timing",
    )
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend()
    figure.tight_layout()
    figure.savefig(args.output / "grasp_mode_action_trajectories.png", dpi=180)
    plt.close(figure)

    matrix = np.asarray(
        [
            [
                next(
                    row["new_over_old"]
                    for row in coverage_rows
                    if row["root_id"] == root and row["region"] == region
                )
                for region in REGIONS
            ]
            for root in range(5)
        ]
    )
    figure, axis = plt.subplots(figsize=(7.2, 4.5))
    image = axis.imshow(matrix, cmap="RdYlGn_r", vmin=0.5, vmax=1.5, aspect="auto")
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axis.text(column, row, f"{matrix[row, column]:.2f}", ha="center", va="center")
    axis.set_xticks(range(len(REGIONS)), list(REGIONS))
    axis.set_yticks(range(5), [f"root {index}" for index in range(5)])
    axis.set_title("Nearest state distance: new demonstrations / old expert")
    figure.colorbar(image, ax=axis, label="ratio (<1 is closer)")
    figure.tight_layout()
    figure.savefig(args.output / "p14_coverage_matrix.png", dpi=180)
    plt.close(figure)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
