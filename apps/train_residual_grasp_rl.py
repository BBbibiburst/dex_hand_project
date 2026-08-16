"""One-command UltraDexGrasp -> MJWarp residual PPO grasp training.

Normal use starts from an object id, not a hand-picked episode path::

    python -m apps.train_residual_grasp_rl --object-id ycb:003_cracker_box

The command reuses an existing Ultra episode when possible, otherwise generates
one automatically, trains residual PPO, and replays the best trajectory in
classic MuJoCo.  ``--reference`` remains available only as a low-level/debug
entry point.  ``--dataset`` runs the same pipeline over the project catalogue.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Iterable


def _slug(object_id: str) -> str:
    return object_id.replace(":", "_").replace("/", "_")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--object-id",
        action="append",
        dest="object_ids",
        help=(
            "Object catalogue id, for example ycb:003_cracker_box. Repeat the "
            "flag to train several objects."
        ),
    )
    source.add_argument(
        "--dataset",
        choices=("all", "ycb", "egad"),
        help="Train every object in the selected project catalogue subset.",
    )
    source.add_argument(
        "--reference",
        type=Path,
        help=(
            "Debug override: an Ultra manifest.json or episode/output directory. "
            "Normal production runs should use --object-id or --dataset."
        ),
    )
    parser.add_argument("--limit", type=int, help="Limit --dataset to its first N objects.")
    parser.add_argument("--output", type=Path, default=Path("outputs/grasp_rl"))

    # Ultra reference generation / discovery.
    parser.add_argument(
        "--ultra-output",
        type=Path,
        default=Path("outputs/ultradexgrasp"),
        help="Root used for automatically generated Ultra episodes.",
    )
    parser.add_argument(
        "--ultra-search-root",
        action="append",
        type=Path,
        default=[],
        help=(
            "Additional root to search for existing Ultra manifests. Can be repeated. "
            "outputs/ultradexgrasp and outputs/ultradexgrasp_catalog are searched automatically."
        ),
    )
    parser.add_argument("--ultra-seeds", type=int, default=3)
    parser.add_argument("--ultra-seed-start", type=int, default=0)
    parser.add_argument("--ultra-device")
    parser.add_argument("--ultra-seed-count", type=int)
    parser.add_argument("--ultra-optimization-steps", type=int)
    parser.add_argument("--ultra-max-execution-candidates", type=int, default=8)
    parser.add_argument(
        "--regenerate-ultra",
        action="store_true",
        help="Ignore reusable Ultra episodes and generate new ones.",
    )

    # MJWarp residual environment / PPO.
    parser.add_argument("--num-envs", type=int, default=1024)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--action-mode", choices=("hand", "arm_hand"), default="hand")
    parser.add_argument("--start-stage", default="approach")
    parser.add_argument("--hand-residual-fraction", type=float, default=0.12)
    parser.add_argument("--arm-residual-radians", type=float, default=0.04)
    parser.add_argument("--nconmax", type=int, default=192)
    parser.add_argument("--njmax", type=int, default=768)
    parser.add_argument("--success-lift-height", type=float, default=0.055)
    parser.add_argument("--success-hold-steps", type=int, default=8)
    parser.add_argument("--updates", type=int, default=1000)
    parser.add_argument("--rollout-steps", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=0, help="Base PPO seed.")
    parser.add_argument("--save-every", type=int, default=25)
    parser.add_argument(
        "--resume",
        type=Path,
        help="Explicit checkpoint for single --object-id/--reference runs.",
    )
    parser.add_argument(
        "--no-auto-resume",
        action="store_true",
        help="Do not automatically continue from an object's latest checkpoint.",
    )
    parser.add_argument(
        "--retrain-complete",
        action="store_true",
        help="Train again even when a verified best trajectory already exists.",
    )
    parser.add_argument("--no-replay", action="store_true")
    parser.add_argument("--render-replay", action="store_true")
    parser.add_argument(
        "--visualize-attempt",
        action="store_true",
        help=(
            "After a one-object run, replay the highest-lift rollout in the classic "
            "MuJoCo viewer even when it failed."
        ),
    )
    parser.add_argument("--fail-fast", action="store_true")
    return parser


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def _object_ids(args: argparse.Namespace) -> list[str]:
    if args.reference is not None:
        return []
    if args.object_ids:
        # Preserve user order while rejecting accidental duplicates.
        return list(dict.fromkeys(args.object_ids))
    from source.envs.manipulation.object_catalog import object_ids

    dataset = None if args.dataset == "all" else args.dataset
    selected = list(object_ids(dataset))
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("--limit must be positive.")
        selected = selected[: args.limit]
    if not selected:
        raise ValueError("The selected object set is empty.")
    return selected


def _ultra_roots(args: argparse.Namespace) -> tuple[Path, ...]:
    roots = [
        args.ultra_output,
        Path("outputs/ultradexgrasp_catalog"),
        *args.ultra_search_root,
    ]
    unique: list[Path] = []
    seen: set[str] = set()
    for root in roots:
        key = str(root.resolve()) if root.exists() else str(root.absolute())
        if key not in seen:
            unique.append(root)
            seen.add(key)
    return tuple(unique)


def _episode_is_usable(manifest: Path, object_id: str, start_stage: str) -> tuple[bool, bool]:
    """Return (usable, source_episode_success). Corrupt/unrelated manifests are ignored."""
    from source.rl.residual.reference import EpisodeRecord, STAGE_CODES

    try:
        episode = EpisodeRecord.load(manifest)
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False, False
    if episode.object_id != object_id or start_stage not in STAGE_CODES:
        return False, False
    stages = {int(value) for value in episode.arrays["stage"].reshape(-1)}
    required = {STAGE_CODES[start_stage], STAGE_CODES["lift"], STAGE_CODES["verify"]}
    if not required.issubset(stages):
        return False, False
    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        payload = {}
    return True, bool(payload.get("success", False))


def _index_existing_references(
    roots: Iterable[Path],
    *,
    start_stage: str,
) -> dict[str, list[tuple[bool, Path]]]:
    """Scan each Ultra root once; batch mode then performs O(1) object lookup."""
    result: dict[str, list[tuple[bool, Path]]] = {}
    visited: set[str] = set()
    for root in roots:
        if not root.exists():
            continue
        for manifest in root.rglob("manifest.json"):
            key = str(manifest.resolve())
            if key in visited:
                continue
            visited.add(key)
            try:
                payload = json.loads(manifest.read_text(encoding="utf-8"))
                object_id = str(payload["object_id"])
            except (OSError, KeyError, TypeError, json.JSONDecodeError):
                continue
            usable, success = _episode_is_usable(manifest, object_id, start_stage)
            if usable:
                result.setdefault(object_id, []).append((success, manifest))
    for entries in result.values():
        # Prefer an already successful Ultra episode, then deterministic paths.
        entries.sort(key=lambda item: (not item[0], str(item[1])))
    return result


def _generate_ultra_reference(
    args: argparse.Namespace,
    object_id: str,
) -> Path | None:
    from source.rl.residual.reference import resolve_reference_manifest
    from tools.ultradexgrasp.generate import main as generate_ultra

    for offset in range(args.ultra_seeds):
        ultra_seed = args.ultra_seed_start + offset
        output = args.ultra_output / _slug(object_id) / f"seed_{ultra_seed:04d}"
        if not args.regenerate_ultra:
            try:
                manifest = resolve_reference_manifest(output)
                usable, _ = _episode_is_usable(manifest, object_id, args.start_stage)
                if usable:
                    print(f"[ultra:reuse] object={object_id} reference={manifest}", flush=True)
                    return manifest
            except (FileNotFoundError, OSError, ValueError):
                pass
            # generate.py intentionally refuses to overwrite a completed episode.
            # If the completed episode is unusable for the requested start stage,
            # move to the next seed instead of crashing on FileExistsError.
            if (output / "manifest.json").is_file():
                print(
                    f"[ultra:skip] object={object_id} seed={ultra_seed} "
                    "existing episode is not a usable RL reference",
                    flush=True,
                )
                continue

        argv = [
            "--object-id",
            object_id,
            "--output",
            str(output),
            "--seed",
            str(ultra_seed),
            "--max-execution-candidates",
            str(args.ultra_max_execution_candidates),
        ]
        if args.ultra_device is not None:
            argv += ["--device", args.ultra_device]
        if args.ultra_seed_count is not None:
            argv += ["--seed-count", str(args.ultra_seed_count)]
        if args.ultra_optimization_steps is not None:
            argv += ["--optimization-steps", str(args.ultra_optimization_steps)]
        if args.regenerate_ultra and (output / "manifest.json").exists():
            argv.append("--overwrite")

        print(
            f"[ultra:generate] object={object_id} seed={ultra_seed} output={output}",
            flush=True,
        )
        rc = generate_ultra(argv)
        # A nonzero Ultra return code can still leave a failed attempt that is a
        # perfectly useful RL reference, so resolve the episode before deciding.
        try:
            manifest = resolve_reference_manifest(output)
            usable, _ = _episode_is_usable(manifest, object_id, args.start_stage)
            if usable:
                print(
                    f"[ultra:reference] object={object_id} rc={rc} reference={manifest}",
                    flush=True,
                )
                return manifest
        except (FileNotFoundError, OSError, ValueError):
            pass
        print(f"[ultra:retry] object={object_id} seed={ultra_seed} rc={rc}", flush=True)
    return None


def _latest_checkpoint(output: Path) -> Path | None:
    final = output / "checkpoint_final.pt"
    if final.is_file():
        return final
    checkpoints = sorted((output / "checkpoints").glob("update_*.pt"))
    return checkpoints[-1] if checkpoints else None


def _train_reference(
    args: argparse.Namespace,
    *,
    reference: Path,
    output: Path,
    seed: int,
    resume: Path | None,
) -> int:
    from source.rl.residual.env import MjWarpResidualLiftEnv, ResidualLiftConfig
    from source.rl.common.ppo import PPOConfig, PPOTrainer

    if args.updates <= 0 or args.save_every <= 0:
        raise ValueError("--updates and --save-every must be positive.")
    env_config = ResidualLiftConfig(
        num_envs=args.num_envs,
        device=args.device,
        action_mode=args.action_mode,
        start_stage=args.start_stage,
        hand_residual_fraction=args.hand_residual_fraction,
        arm_residual_radians=args.arm_residual_radians,
        nconmax=args.nconmax,
        njmax=args.njmax,
        success_lift_height=args.success_lift_height,
        success_hold_steps=args.success_hold_steps,
    )
    ppo_config = PPOConfig(
        rollout_steps=args.rollout_steps,
        learning_rate=args.learning_rate,
    )
    output.mkdir(parents=True, exist_ok=True)
    config_path = output / "config.json"
    config_payload = {
        "reference": str(reference),
        "seed": seed,
        "environment": asdict(env_config),
        "ppo": asdict(ppo_config),
    }
    if resume is not None and config_path.is_file():
        try:
            stored_config = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            stored_config = {}
        compatible = (
            stored_config.get("reference") == config_payload["reference"]
            and stored_config.get("environment") == config_payload["environment"]
            and stored_config.get("ppo") == config_payload["ppo"]
        )
        if not compatible:
            print(
                f"[resume:skip] checkpoint={resume} configuration/reference changed",
                flush=True,
            )
            resume = None
    _write_json(config_path, config_payload)

    env = MjWarpResidualLiftEnv(reference, env_config)
    trainer = PPOTrainer(env, ppo_config, seed=seed)
    if resume is not None:
        trainer.load(resume)
        print(f"resumed={resume} update={trainer.update_index}", flush=True)
    saved_best_version = 0
    saved_attempt_version = 0

    def callback(active: PPOTrainer, metrics: dict) -> None:
        nonlocal saved_best_version, saved_attempt_version
        print(
            f"object={env.reference.object_id} update={metrics['update']:05d} "
            f"steps={metrics['total_steps']} reward={metrics['mean_reward']:.3f} "
            f"success={metrics['episode_success_rate']:.1%} "
            f"lift={metrics['mean_lift']:.3f}/{metrics['max_lift']:.3f}m "
            f"digits={metrics['mean_contact_digits']:.2f} "
            f"thumb={metrics['thumb_contact_rate']:.1%} "
            f"opp={metrics['opposition_rate']:.1%} "
            f"stable={metrics['stable_rate']:.2%} hold={metrics['max_hold_steps']:.0f} "
            f"attempt_lift={metrics['best_attempt_lift']:.3f}m "
            f"best={metrics['best_success_return']:.2f} kl={metrics['kl']:.4f}",
            flush=True,
        )
        if active.update_index % args.save_every == 0:
            active.save(output / "checkpoints" / f"update_{active.update_index:05d}.pt")
        if (
            env.best_attempt_trajectory is not None
            and env.best_attempt_version > saved_attempt_version
        ):
            manifest = env.best_attempt_trajectory.save(output / "best_attempt")
            saved_attempt_version = env.best_attempt_version
            print(
                f"best_attempt={manifest} lift={env.best_attempt_lift:.3f}m "
                f"return={env.best_attempt_return:.2f}",
                flush=True,
            )
        if env.best_trajectory is not None and env.best_version > saved_best_version:
            manifest = env.best_trajectory.save(output / "best_trajectory")
            saved_best_version = env.best_version
            print(f"best_trajectory={manifest}", flush=True)

    try:
        print(
            f"reference={env.reference.source_manifest} object={env.reference.object_id} "
            f"horizon={env.reference.horizon} envs={env.num_envs} "
            f"obs={env.obs_dim} action={env.action_dim} mode={env_config.action_mode}",
            flush=True,
        )
        trainer.train(args.updates, callback=callback)
        checkpoint = trainer.save(output / "checkpoint_final.pt")
        _write_json(output / "metrics.json", env.training_metrics())
        print(f"checkpoint={checkpoint}", flush=True)
        if env.best_attempt_trajectory is not None:
            manifest = env.best_attempt_trajectory.save(output / "best_attempt")
            print(
                f"diagnostic_attempt={manifest} lift={env.best_attempt_lift:.3f}m",
                flush=True,
            )
        if env.best_trajectory is None:
            print("No stable RL trajectory found yet.", flush=True)
            return 2
        env.best_trajectory.save(output / "best_trajectory")
        return 0
    finally:
        env.close()


def _replay_best(args: argparse.Namespace, output: Path) -> tuple[bool, dict]:
    if args.no_replay:
        return True, {"skipped": True}
    from source.rl.residual.replay import replay_residual_trajectory

    result = replay_residual_trajectory(
        output / "best_trajectory",
        render_mode="human" if args.render_replay else None,
    )
    payload = asdict(result)
    _write_json(output / "replay.json", payload)
    print(
        f"[replay] success={result.success} fraction={result.success_fraction:.1%} "
        f"lift={result.object_lift:.3f}m frames={result.frames}",
        flush=True,
    )
    return bool(result.success), payload


def _visualize_attempt(args: argparse.Namespace, output: Path) -> None:
    if not args.visualize_attempt:
        return
    manifest = output / "best_attempt" / "manifest.json"
    if not manifest.is_file():
        print(f"[visualize] no diagnostic attempt at {manifest}", flush=True)
        return
    from source.rl.residual.replay import replay_residual_trajectory

    print(f"[visualize] replaying {manifest} in classic MuJoCo", flush=True)
    result = replay_residual_trajectory(manifest, render_mode="human")
    print(
        f"[visualize] success={result.success} fraction={result.success_fraction:.1%} "
        f"final_lift={result.object_lift:.3f}m frames={result.frames}",
        flush=True,
    )


def _completed_and_verified(output: Path) -> bool:
    manifest = output / "best_trajectory" / "manifest.json"
    replay = output / "replay.json"
    if not manifest.is_file() or not replay.is_file():
        return False
    try:
        return bool(json.loads(replay.read_text(encoding="utf-8")).get("success", False))
    except (OSError, json.JSONDecodeError):
        return False


def _run_explicit_reference(args: argparse.Namespace) -> int:
    output = args.output
    resume = args.resume
    if resume is None and not args.no_auto_resume:
        resume = _latest_checkpoint(output)
    rc = _train_reference(
        args,
        reference=args.reference,
        output=output,
        seed=args.seed,
        resume=resume,
    )
    _visualize_attempt(args, output)
    if rc != 0:
        return rc
    replay_ok, _ = _replay_best(args, output)
    return 0 if replay_ok else 3


def run(args: argparse.Namespace) -> int:
    if args.ultra_seeds <= 0:
        raise ValueError("--ultra-seeds must be positive.")
    if args.ultra_max_execution_candidates <= 0:
        raise ValueError("--ultra-max-execution-candidates must be positive.")
    if args.resume is not None and args.dataset is not None:
        raise ValueError("--resume cannot be shared across a dataset run; use auto-resume instead.")
    if args.reference is not None:
        return _run_explicit_reference(args)

    selected = _object_ids(args)
    if args.visualize_attempt and len(selected) != 1:
        raise ValueError("--visualize-attempt requires exactly one object/reference run.")
    roots = _ultra_roots(args)
    reference_index = (
        {} if args.regenerate_ultra else _index_existing_references(roots, start_stage=args.start_stage)
    )
    multi = len(selected) > 1
    summary: list[dict] = []
    failed = 0

    for index, object_id in enumerate(selected):
        output = args.output / _slug(object_id)
        print(
            f"\n=== [{index + 1}/{len(selected)}] object={object_id} output={output} ===",
            flush=True,
        )
        if _completed_and_verified(output) and not args.retrain_complete:
            row = {"object_id": object_id, "status": "already_verified", "output": str(output)}
            summary.append(row)
            print("[skip] verified RL trajectory already exists", flush=True)
            continue

        reference: Path | None = None
        entries = reference_index.get(object_id, [])
        if entries:
            reference = entries[0][1]
            print(f"[ultra:reuse] object={object_id} reference={reference}", flush=True)
        if reference is None:
            reference = _generate_ultra_reference(args, object_id)
        if reference is None:
            row = {"object_id": object_id, "status": "no_ultra_reference", "output": str(output)}
            summary.append(row)
            failed += 1
            print(f"[failed] no usable Ultra reference for {object_id}", flush=True)
            if args.fail_fast:
                break
            continue

        resume = args.resume if not multi else None
        if resume is None and not args.no_auto_resume:
            resume = _latest_checkpoint(output)
        try:
            rc = _train_reference(
                args,
                reference=reference,
                output=output,
                seed=args.seed + index,
                resume=resume,
            )
            _visualize_attempt(args, output)
            replay_payload: dict = {}
            if rc == 0:
                replay_ok, replay_payload = _replay_best(args, output)
                status = "verified" if replay_ok else "replay_failed"
                if not replay_ok:
                    rc = 3
            else:
                status = "rl_no_success"
            row = {
                "object_id": object_id,
                "status": status,
                "reference": str(reference),
                "output": str(output),
                "return_code": rc,
                "replay": replay_payload,
            }
        except Exception as exc:
            row = {
                "object_id": object_id,
                "status": "exception",
                "reference": str(reference),
                "output": str(output),
                "error": f"{type(exc).__name__}: {exc}",
            }
            rc = 4
            print(f"[exception] {row['error']}", flush=True)
            if args.fail_fast:
                summary.append(row)
                failed += 1
                break
        summary.append(row)
        if rc != 0:
            failed += 1

    summary_path = args.output / "summary.json"
    _write_json(
        summary_path,
        {
            "schema_version": 1,
            "objects": len(summary),
            "failed": failed,
            "results": summary,
        },
    )
    print(f"\nsummary={summary_path} failed={failed}/{len(summary)}", flush=True)
    return 0 if failed == 0 else 2


def main(argv: list[str] | None = None) -> int:
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
