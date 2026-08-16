"""Pure candidate search engine."""

from __future__ import annotations

import time

import numpy as np
from scipy.spatial.transform import Rotation

from source.grasping.search.common import progress
from source.grasping.search.hand_geometry import fraction_candidates, surface_for
from source.grasping.search.planning import local_pose_candidates
from source.grasping.search.scoring import evaluate
from source.grasping.search.types import Candidate, Cloud, Device


def _retain(bucket: list[Candidate], candidate: Candidate, keep: int) -> None:
    bucket.append(candidate)
    bucket.sort(key=lambda item: (not item.valid, item.score))
    del bucket[keep:]


def search(
    cloud: Cloud,
    device: Device,
    *,
    joint_candidates: int,
    anchor_count: int,
    rolls_per_anchor: int,
    coarse_keep: int,
    top_k: int,
    support_margin: float,
    seed: int,
) -> list[Candidate]:
    all_fractions = fraction_candidates(device, max(3, joint_candidates // 16))
    coarse_stride = max(1, len(all_fractions) // 8)
    coarse_fractions = all_fractions[::coarse_stride]
    if all_fractions[-1] is not coarse_fractions[-1]:
        coarse_fractions.append(all_fractions[-1])
    coarse_rolls = max(2, rolls_per_anchor // 2)
    poses = local_pose_candidates(
        cloud,
        anchor_count=anchor_count,
        rolls_per_anchor=coarse_rolls,
        support_margin=support_margin,
        seed=seed,
    )
    coarse_depths = (-0.018, -0.006, 0.006)
    estimated = len(coarse_fractions) * len(poses) * len(coarse_depths)
    progress(
        f"[coarse] hand_shapes={len(coarse_fractions)} poses={len(poses)} "
        f"depths={len(coarse_depths)} evaluations={estimated}"
    )
    coarse: list[Candidate] = []
    progress_step = max(1, estimated // 10)
    evaluated = 0
    for fraction_index, fraction in enumerate(coarse_fractions):
        progress(f"[coarse] building hand shape {fraction_index + 1}/{len(coarse_fractions)}")
        shape_started = time.perf_counter()
        surface = surface_for(device, fraction, seed=seed + fraction_index)
        progress(
            f"[coarse] hand shape {fraction_index + 1}/{len(coarse_fractions)} ready "
            f"({time.perf_counter() - shape_started:.1f}s, points={len(surface.points)})"
        )
        for anchor_index, roll_index, rotation, grasp_center in poses:
            base_translation = grasp_center - surface.midpoint @ rotation.T
            for depth in coarse_depths:
                candidate = evaluate(
                    cloud,
                    device,
                    surface,
                    rotation,
                    base_translation + rotation[:, 0] * depth,
                    roll_index=roll_index,
                    anchor_index=anchor_index,
                    full_checks=False,
                )
                _retain(coarse, candidate, max(1, coarse_keep))
                evaluated += 1
                if evaluated % progress_step == 0 or evaluated == estimated:
                    best = coarse[0]
                    progress(
                        f"[coarse] {evaluated}/{estimated} best={best.score:.4f} valid={best.valid}"
                    )

    progress(f"[fine] refining {len(coarse)} coarse seeds")
    fine: list[Candidate] = []
    angle_offsets = np.deg2rad((-6.0, 0.0, 6.0))
    depth_offsets = (-0.004, 0.0, 0.004)
    lateral_offsets = (-0.003, 0.0, 0.003)
    fine_total = len(coarse) * len(angle_offsets) * len(depth_offsets) * len(lateral_offsets)
    fine_step = max(1, fine_total // 10)
    evaluated = 0
    for seed_index, coarse_candidate in enumerate(coarse):
        for angle in angle_offsets:
            local_delta = Rotation.from_rotvec(np.array([angle, 0.0, 0.0])).as_matrix()
            rotation = coarse_candidate.rotation @ local_delta
            for depth in depth_offsets:
                for lateral in lateral_offsets:
                    translation = (
                        coarse_candidate.translation
                        + rotation[:, 0] * depth
                        + rotation[:, 1] * lateral
                    )
                    candidate = evaluate(
                        cloud,
                        device,
                        coarse_candidate.surface,
                        rotation,
                        translation,
                        roll_index=coarse_candidate.roll_index,
                        anchor_index=coarse_candidate.anchor_index,
                        full_checks=True,
                    )
                    _retain(fine, candidate, max(1, top_k, coarse_keep))
                    evaluated += 1
                    if evaluated % fine_step == 0 or evaluated == fine_total:
                        best = fine[0]
                        progress(
                            f"[fine] {evaluated}/{fine_total} "
                            f"best={best.score:.4f} valid={best.valid}"
                        )

    # Always preserve at least one result for visualization and debugging.
    selected = fine or coarse
    if not selected:
        raise RuntimeError("No candidate was evaluated.")
    selected.sort(key=lambda item: (not item.valid, item.score))
    valid_count = sum(item.valid for item in selected)
    progress(
        f"[search] saved={len(selected[: max(1, top_k)])} "
        f"valid={valid_count} fallback={valid_count == 0}"
    )
    return selected[: max(1, top_k)]
