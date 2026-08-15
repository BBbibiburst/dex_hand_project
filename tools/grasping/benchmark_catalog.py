"""Search and physics-validate grasps for catalogue objects."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

from source.grasping.constants import DEFAULT_GRIP_PRELOAD
from source.grasping.dexevolve import mjwarp_available
from source.workflows.grasp_benchmark import GraspBenchmarkConfig, run_grasp_benchmark

FULL_PIPELINE_PRESET = {
    "dataset": "all",
    "generator": "heuristic",
    "graspqp_iterations": 40,
    "points": 1024,
    "joint_candidates": 64,
    "surface_anchors": 12,
    "rolls_per_anchor": 4,
    "coarse_keep": 12,
    "top_k": 4,
    "search_attempts": 5,
    "target_lift_candidates": 3,
    "maximum_saved_candidates": 24,
    "maximum_robot_candidates_per_attempt": 6,
    "task_conditioned_search": True,
    "task_scene_attempts": 12,
    "task_rotations_per_distance": 4,
    "task_pull_step": 0.05,
    "task_maximum_pull": 0.10,
    "maximum_object_seconds": 2700.0,
    "seconds": 3.0,
    "evolve": True,
    "evolution_population": 32,
    "evolution_offspring": 16,
    "evolution_generations": 20,
    "evolution_seconds": 1.5,
    "evolution_backend": "cpu",
    "validate_robot_lift": True,
    # A full pipeline result is intended for successful demonstration
    # collection, so a trajectory-only grasp is not the terminal goal.
    "retry_incomplete": True,
}

PILOT_OBJECT_IDS = [
    "ycb:003_cracker_box",
    "ycb:011_banana",
    "ycb:019_pitcher_base",
    "ycb:024_bowl",
    "ycb:025_mug",
    "ycb:026_sponge",
    "ycb:035_power_drill",
    "ycb:061_foam_brick",
]

GIB = 1024**3
MAXIMUM_PROCESS_PARALLELISM = 8
MEMORY_PER_WORKER_BYTES = 2 * GIB
RESERVED_MEMORY_BYTES = GIB


def _available_cpu_count() -> int:
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except AttributeError:
        return max(1, os.cpu_count() or 1)


def _available_memory_bytes() -> int | None:
    """Return conservative available memory, respecting cgroup v2 when present."""
    candidates: list[int] = []
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("MemAvailable:"):
                candidates.append(int(line.split()[1]) * 1024)
                break
    except (OSError, ValueError, IndexError):
        pass
    try:
        limit_text = Path("/sys/fs/cgroup/memory.max").read_text(encoding="utf-8").strip()
        if limit_text != "max":
            limit = int(limit_text)
            used = int(Path("/sys/fs/cgroup/memory.current").read_text(encoding="utf-8"))
            candidates.append(max(0, limit - used))
    except (OSError, ValueError):
        pass
    return min(candidates) if candidates else None


def _recommended_parallelism(
    *, cpu_count: int | None = None, available_memory_bytes: int | None = None
) -> int:
    """Choose a safe worker count from CPU affinity and available memory."""
    cpus = _available_cpu_count() if cpu_count is None else max(1, cpu_count)
    cpu_budget = max(1, cpus - 1) if cpus > 2 else 1
    memory = _available_memory_bytes() if available_memory_bytes is None else available_memory_bytes
    memory_budget = MAXIMUM_PROCESS_PARALLELISM
    if memory is not None:
        memory_budget = max(1, (memory - RESERVED_MEMORY_BYTES) // MEMORY_PER_WORKER_BYTES)
    return min(MAXIMUM_PROCESS_PARALLELISM, cpu_budget, memory_budget)


def _apply_full_pipeline_preset(values: dict, explicitly_set: set[str]) -> None:
    """Apply safe defaults without discarding explicit server overrides."""
    for name, value in FULL_PIPELINE_PRESET.items():
        if name not in explicitly_set:
            values[name] = value
    if "evolution_jobs" not in explicitly_set:
        values["evolution_jobs"] = 1
    if "jobs" not in explicitly_set:
        values["jobs"] = max(
            1,
            _recommended_parallelism() // int(values["evolution_jobs"]),
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--full-pipeline",
        action="store_true",
        help="Use the standard heuristic -> DexEvolve -> MuJoCo full-catalog preset.",
    )
    parser.add_argument(
        "--pilot",
        action="store_true",
        help="Run the full pipeline on eight representative shapes with early diagnosis.",
    )
    parser.add_argument("--pilot-min-results", type=int, default=4)
    parser.add_argument("--pilot-min-lift-rate", type=float, default=0.25)
    parser.add_argument("--pilot-max-repeated-failure", type=int, default=3)
    parser.add_argument("--dataset", choices=("all", "ycb", "egad"), default="all")
    parser.add_argument("--object-id", action="append", dest="object_ids")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--points", type=int, default=2048)
    parser.add_argument("--joint-candidates", type=int, default=128)
    parser.add_argument("--surface-anchors", type=int, default=24)
    parser.add_argument("--rolls-per-anchor", type=int, default=8)
    parser.add_argument("--coarse-keep", type=int, default=24)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--support-margin", type=float, default=0.008)
    parser.add_argument("--generator", choices=("heuristic", "graspqp"), default="heuristic")
    parser.add_argument("--graspqp-iterations", type=int, default=120)
    parser.add_argument(
        "--evolve",
        action="store_true",
        help="Run DexEvolve-style MuJoCo refinement after GraspQP generation.",
    )
    parser.add_argument("--evolution-population", type=int, default=32)
    parser.add_argument("--evolution-offspring", type=int, default=16)
    parser.add_argument("--evolution-generations", type=int, default=20)
    parser.add_argument("--evolution-jobs", type=int, default=4)
    parser.add_argument("--evolution-seconds", type=float, default=1.5)
    parser.add_argument(
        "--evolution-backend",
        choices=("auto", "cpu", "mjwarp"),
        default="cpu",
        help="Direct-hold rollout backend; use mjwarp explicitly after benchmarking the GPU.",
    )
    parser.add_argument("--mjwarp-device", default="cuda:0")
    parser.add_argument("--mjwarp-batch-size", type=int, default=32)
    parser.add_argument("--mjwarp-nconmax", type=int, default=128)
    parser.add_argument("--mjwarp-njmax", type=int, default=512)
    parser.add_argument("--evolution-dir", type=Path)
    parser.add_argument("--search-attempts", type=int, default=3)
    parser.add_argument("--target-lift-candidates", type=int, default=1)
    parser.add_argument("--maximum-saved-candidates", type=int, default=24)
    parser.add_argument("--maximum-robot-candidates-per-attempt", type=int, default=6)
    parser.add_argument(
        "--task-conditioned-search",
        action="store_true",
        help="Search complete Lift tasks and resample object yaw/distance after infeasibility.",
    )
    parser.add_argument("--task-scene-attempts", type=int, default=12)
    parser.add_argument("--task-rotations-per-distance", type=int, default=4)
    parser.add_argument("--task-pull-step", type=float, default=0.05)
    parser.add_argument("--task-maximum-pull", type=float, default=0.15)
    parser.add_argument(
        "--maximum-object-seconds",
        type=float,
        default=2700.0,
        help="Stop starting new candidate/attempt work after this per-object wall-clock budget.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--target-size", type=float, default=0.09)
    parser.add_argument("--end-effector", choices=("dex_hand", "pika_gripper"), default="dex_hand")
    parser.add_argument("--seconds", type=float, default=3.0)
    parser.add_argument("--settle-seconds", type=float, default=0.8)
    parser.add_argument("--grip-preload", type=float, default=DEFAULT_GRIP_PRELOAD)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--reuse", action="store_true")
    parser.add_argument(
        "--validate-robot-lift",
        action="store_true",
        help="Record collision-free full-robot Lift validation separately from trajectory stability.",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--retry-incomplete",
        action="store_true",
        help="With --resume, recompute rows lacking trajectory or requested robot-Lift success.",
    )
    parser.add_argument("--config-dir", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    values = vars(args)
    full_pipeline = values.pop("full_pipeline")
    pilot = bool(values["pilot"])
    if full_pipeline or pilot:
        explicitly_set = {
            argument[2:].split("=", 1)[0].replace("-", "_")
            for argument in sys.argv[1:]
            if argument.startswith("--")
        }
        _apply_full_pipeline_preset(values, explicitly_set)
        if pilot:
            if "object_id" not in explicitly_set:
                values["object_ids"] = PILOT_OBJECT_IDS.copy()
            values["limit"] = None
            values["jobs"] = 1
            values["evolution_jobs"] = 1
            if "target_lift_candidates" not in explicitly_set:
                values["target_lift_candidates"] = 1
            if "maximum_object_seconds" not in explicitly_set:
                values["maximum_object_seconds"] = 1200.0
        gpu_evolution = values["evolution_backend"] == "mjwarp" or (
            values["evolution_backend"] == "auto" and mjwarp_available()
        )
        if gpu_evolution:
            if "jobs" not in explicitly_set:
                values["jobs"] = 1
            if "evolution_jobs" not in explicitly_set:
                values["evolution_jobs"] = 1
        if "jobs" not in explicitly_set or "evolution_jobs" not in explicitly_set:
            memory = _available_memory_bytes()
            memory_label = "unknown" if memory is None else f"{memory / GIB:.1f}GiB"
            print(
                f"parallelism=auto available_cpus={_available_cpu_count()} "
                f"available_memory={memory_label} selected_jobs={values['jobs']} "
                f"selected_evolution_jobs={values['evolution_jobs']}",
                f"evolution_backend={values['evolution_backend']}",
                flush=True,
            )
        if values["output"] is None:
            values["output"] = Path(
                "configs/grasps/dex_hand/pilot_benchmark.json"
                if pilot
                else "configs/grasps/dex_hand/full_pipeline_benchmark.json"
            )
        if values["config_dir"] is None:
            values["config_dir"] = Path("configs/grasps/dex_hand/heuristic_seeds")
        if values["evolution_dir"] is None:
            values["evolution_dir"] = Path("configs/grasps/dex_hand/dexevolve")
    raise SystemExit(run_grasp_benchmark(GraspBenchmarkConfig(**values)))


if __name__ == "__main__":
    main()
