"""Train a hybrid categorical-wrist + continuous-hand grasp editor on one object."""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from source.rl.grasp_edit_templates import (
    build_grasp_edit_templates,
    discover_ultra_attempts,
)
from source.rl.mjwarp_grasp_edit_env import GraspEditConfig, MjWarpGraspEditEnv
from source.rl.grasp_edit_hybrid_ppo import HybridPPOTrainer
from source.rl.ppo import PPOConfig


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--object-id", required=True)
    parser.add_argument(
        "--output-root", type=Path, default=Path("outputs/grasp_edit_rl_v10")
    )
    parser.add_argument(
        "--template-root", type=Path, default=Path("outputs/grasp_edit_lattice_v9")
    )
    parser.add_argument("--ultra-root", type=Path, action="append", dest="ultra_roots")
    parser.add_argument("--ultra-seed-count", type=int, default=100)
    parser.add_argument("--ultra-generate-seeds", type=int, default=3)
    parser.add_argument("--ultra-max-execution-candidates", type=int, default=8)
    parser.add_argument("--no-auto-ultra", action="store_true")
    parser.add_argument(
        "--failed-only",
        action="store_true",
        help="Compile/train only wrist templates that fail before RL editing.",
    )
    parser.add_argument("--base-candidates", type=int, default=3)
    parser.add_argument("--wrist-translation-step", type=float, default=0.01)
    parser.add_argument("--wrist-rotation-step-deg", type=float, default=15.0)
    parser.add_argument("--lattice-max-templates", type=int, default=12)
    parser.add_argument("--lattice-max-executions", type=int, default=32)
    parser.add_argument("--overwrite-templates", action="store_true")
    parser.add_argument("--num-envs", type=int, default=256)
    parser.add_argument("--updates", type=int, default=20)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--wrist-translation-scale",
        type=float,
        default=0.02,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--wrist-rotation-scale-deg",
        type=float,
        default=45.0,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--hand-edit-fraction", type=float, default=0.35)
    parser.add_argument("--success-lift-height", type=float, default=0.055)
    parser.add_argument("--success-tail-steps", type=int, default=8)
    parser.add_argument("--nconmax", type=int, default=192)
    parser.add_argument("--njmax", type=int, default=768)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--initial-std", type=float, default=0.70)
    parser.add_argument("--update-epochs", type=int, default=6)
    parser.add_argument("--minibatches", type=int, default=4)
    parser.add_argument("--resume", type=Path)
    parser.add_argument(
        "--save-every",
        type=int,
        default=5,
        help="Checkpoint interval in updates; 0 disables periodic saves.",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=1,
        help="Print one compact training row every N updates.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show per-template lattice compilation details and template table.",
    )
    parser.add_argument("--visualize", action="store_true")
    return parser


def _object_slug(object_id: str) -> str:
    return object_id.replace(":", "_").replace("/", "_")


def _ensure_ultra_priors(args: argparse.Namespace, ultra_roots: tuple[Path, ...]) -> None:
    existing = discover_ultra_attempts(
        args.object_id,
        roots=ultra_roots,
        maximum=max(1, args.base_candidates),
    )
    if existing:
        print(
            f"[ultra:reuse] object={args.object_id} full_attempts={len(existing)}",
            flush=True,
        )
        return
    if args.no_auto_ultra:
        raise FileNotFoundError(
            f"No full UltraDexGrasp attempts found for {args.object_id} and "
            "--no-auto-ultra was requested."
        )
    if args.ultra_seed_count <= 0 or args.ultra_generate_seeds <= 0:
        raise ValueError("Ultra seed counts must be positive.")
    if args.ultra_max_execution_candidates <= 0:
        raise ValueError("--ultra-max-execution-candidates must be positive.")

    from tools.ultradexgrasp.generate import main as generate_ultra

    primary_root = ultra_roots[0]
    for offset in range(args.ultra_generate_seeds):
        rng_seed = args.seed + offset
        output = primary_root / _object_slug(args.object_id) / f"seed_{rng_seed:04d}"
        print(
            f"[ultra:auto] object={args.object_id} rng_seed={rng_seed} "
            f"grasp_seeds={args.ultra_seed_count} output={output}",
            flush=True,
        )
        generate_args = [
            "--object-id", args.object_id,
            "--seed", str(rng_seed),
            "--seed-count", str(args.ultra_seed_count),
            "--max-execution-candidates", str(args.ultra_max_execution_candidates),
            "--output", str(output),
        ]
        # Regeneration may target a partially created directory from a previous
        # unsuccessful run.  The CLI's overwrite flag is explicit and local to
        # this auto-generation output.
        if output.exists():
            generate_args.append("--overwrite")
        rc = generate_ultra(generate_args)
        existing = discover_ultra_attempts(
            args.object_id,
            roots=ultra_roots,
            maximum=max(1, args.base_candidates),
        )
        if existing:
            print(f"[ultra:auto-ready] rc={rc} full_attempts={len(existing)}", flush=True)
            return
        print(f"[ultra:auto-miss] rc={rc} rng_seed={rng_seed}", flush=True)

    raise FileNotFoundError(
        f"Automatic Ultra generation produced no full attempts for {args.object_id} "
        f"after {args.ultra_generate_seeds} RNG seed(s)."
    )


def _trajectory_summary(trajectory) -> str:
    if trajectory is None:
        return ""
    meta = trajectory.metadata
    template_id = meta.get("template_id", "?")
    label = meta.get("template_label", "?")
    translation = meta.get("template_translation_offset", [0.0, 0.0, 0.0])
    rotation = meta.get("template_rotation_offset_degrees", [0.0, 0.0, 0.0])
    max_lift = meta.get("mjwarp_max_lift", 0.0)
    final_lift = meta.get("mjwarp_final_lift", 0.0)
    return (
        f"template=t{template_id} label={label} "
        f"dxyz={[round(float(x), 3) for x in translation]} "
        f"rpy={[round(float(x), 1) for x in rotation]} "
        f"pre_rl_success={meta.get('template_pre_rl_success', '?')} "
        f"lift={float(max_lift):.3f}m final={float(final_lift):.3f}m"
    )


def run(args: argparse.Namespace) -> int:
    if args.save_every < 0:
        raise ValueError("--save-every must be >= 0.")
    if args.log_every <= 0:
        raise ValueError("--log-every must be positive.")
    ultra_roots = (
        tuple(args.ultra_roots)
        if args.ultra_roots
        else (Path("outputs/ultradexgrasp"), Path("outputs/ultradexgrasp_catalog"))
    )
    _ensure_ultra_priors(args, ultra_roots)
    templates = build_grasp_edit_templates(
        args.object_id,
        output_root=args.template_root,
        ultra_roots=ultra_roots,
        base_candidates=args.base_candidates,
        translation_step=args.wrist_translation_step,
        rotation_step_degrees=args.wrist_rotation_step_deg,
        maximum_templates=args.lattice_max_templates,
        maximum_executions=args.lattice_max_executions,
        seed=args.seed,
        overwrite=args.overwrite_templates,
        failed_only=args.failed_only,
        verbose=args.verbose,
    )

    output = args.output_root / _object_slug(args.object_id)
    if args.failed_only:
        output = output / "failed_only"
    output.mkdir(parents=True, exist_ok=True)

    env_config = GraspEditConfig(
        num_envs=args.num_envs,
        device=args.device,
        wrist_translation_scale=args.wrist_translation_scale,
        wrist_rotation_scale_degrees=args.wrist_rotation_scale_deg,
        hand_edit_fraction=args.hand_edit_fraction,
        success_lift_height=args.success_lift_height,
        success_tail_steps=args.success_tail_steps,
        nconmax=args.nconmax,
        njmax=args.njmax,
    )
    ppo_config = PPOConfig(
        rollout_steps=1,
        update_epochs=args.update_epochs,
        minibatches=args.minibatches,
        learning_rate=args.learning_rate,
        initial_std=args.initial_std,
        entropy_coef=0.01,
    )
    env = MjWarpGraspEditEnv(args.object_id, templates, env_config)
    trainer = HybridPPOTrainer(env, ppo_config, seed=args.seed)

    if args.resume is not None:
        trainer.load(args.resume)
        print(f"resumed={args.resume} update={trainer.update_index}", flush=True)

    _write_json(
        output / "config.json",
        {
            "grasp_edit_version": 10,
            "policy": {
                "template": "categorical",
                "hand": "tanh_gaussian_6d",
                "encoded_action_dim": env.action_dim,
            },
            "object_id": args.object_id,
            "failed_only": bool(args.failed_only),
            "ultra_seed_count": int(args.ultra_seed_count),
            "lattice": {
                "translation_step": args.wrist_translation_step,
                "rotation_step_degrees": args.wrist_rotation_step_deg,
                "maximum_templates": args.lattice_max_templates,
                "maximum_executions": args.lattice_max_executions,
            },
            "environment": asdict(env_config),
            "ppo": asdict(ppo_config),
            "templates": env.template_summary(),
        },
    )

    mode = "failed-only" if args.failed_only else "all"
    print(
        f"[train] object={args.object_id} mode={mode} templates={env.template_count} "
        f"envs={env.num_envs} horizon={env.horizon} "
        f"action=Categorical({env.template_count})+Gaussian(6) updates={args.updates}",
        flush=True,
    )
    if args.verbose:
        for row in env.template_summary():
            print(
                f"  t{row['id']:02d} {row['label']} "
                f"dxyz={[round(float(x), 3) for x in row['translation_offset']]} "
                f"rpy={[round(float(x), 1) for x in row['rotation_offset_degrees']]} "
                f"pre_rl_success={row['success_before_edit']} "
                f"ik={row['precheck_position_error']:.3f}m/"
                f"{row['precheck_orientation_error']:.3f}rad",
                flush=True,
            )

    persisted_attempt_version = 0
    persisted_best_version = 0

    def persist_best() -> str:
        nonlocal persisted_attempt_version, persisted_best_version
        # Once a successful trajectory exists, best_attempt is only diagnostic
        # duplication.  Persist the successful artifact; otherwise persist the
        # best physical attempt seen so far.
        if env.best_trajectory is not None:
            if env.best_version > persisted_best_version:
                env.best_trajectory.save(output / "best_trajectory")
                persisted_best_version = env.best_version
                return "best"
            return ""
        if (
            env.best_attempt_trajectory is not None
            and env.best_attempt_version > persisted_attempt_version
        ):
            env.best_attempt_trajectory.save(output / "best_attempt")
            persisted_attempt_version = env.best_attempt_version
            return "attempt"
        return ""

    def callback(active: HybridPPOTrainer, metrics: dict) -> None:
        rates = [
            metrics.get(f"template_{index}_rate", 0.0)
            for index in range(env.template_count)
        ]
        top_template = max(range(len(rates)), key=rates.__getitem__)
        should_log = (
            active.update_index == 1
            or active.update_index % args.log_every == 0
            or active.update_index == args.updates
        )
        if should_log:
            print(
                f"u {active.update_index:03d}/{args.updates:03d} | "
                f"succ {metrics['episode_success_rate']:6.1%} | "
                f"lift {1000.0 * metrics['mean_max_lift']:4.0f}/"
                f"{1000.0 * metrics['best_attempt_lift']:4.0f}mm | "
                f"final {1000.0 * metrics['mean_final_lift']:4.0f}/"
                f"{1000.0 * metrics['best_attempt_final_lift']:4.0f}mm | "
                f"top t{top_template}:{rates[top_template]:5.1%} | "
                f"kl {metrics['kl']:.3f}",
                flush=True,
            )

        if args.save_every and active.update_index % args.save_every == 0:
            active.save(output / "checkpoints" / f"update_{active.update_index:05d}.pt")
            artifact = persist_best()
            suffix = f" + {artifact}" if artifact else ""
            print(f"[save] u={active.update_index} checkpoint{suffix}", flush=True)

    try:
        trainer.train(args.updates, callback=callback)
        trainer.save(output / "checkpoint_final.pt")
        _write_json(output / "metrics.json", env.training_metrics())
        persist_best()
        if env.best_trajectory is not None:
            print(f"[final] success {_trajectory_summary(env.best_trajectory)}", flush=True)
        elif env.best_attempt_trajectory is not None:
            print(f"[final] no-success {_trajectory_summary(env.best_attempt_trajectory)}", flush=True)
        else:
            print("[final] no trajectory captured", flush=True)
    finally:
        env.close()

    trajectory = (
        output / "best_trajectory"
        if (output / "best_trajectory" / "manifest.json").is_file()
        else output / "best_attempt"
    )
    if args.visualize and (trajectory / "manifest.json").is_file():
        from source.rl.replay import replay_residual_trajectory

        print(f"[visualize] {trajectory}", flush=True)
        result = replay_residual_trajectory(trajectory, render_mode="human")
        print(
            f"[visualize] success={result.success} "
            f"success_fraction={result.success_fraction:.1%} "
            f"final_lift={result.object_lift:.3f}m frames={result.frames}",
            flush=True,
        )

    return 0 if (output / "best_trajectory" / "manifest.json").is_file() else 2


def main(argv: list[str] | None = None) -> int:
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
