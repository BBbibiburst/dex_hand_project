"""Train primitive-conditioned single-step grasp editing on one object.

This is an experimental sibling of :mod:`apps.train_grasp_edit_rl`.  It leaves
that baseline untouched and expands the categorical action from ``wrist
 template`` to ``wrist template x grasp primitive``.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from apps.train_grasp_edit_rl import _ensure_ultra_priors, _object_slug, _write_json
from source.rl.common.ppo import PPOConfig
from source.rl.grasp_edit.ppo import HybridPPOTrainer
from source.rl.grasp_edit.primitive_env import (
    PrimitiveGraspEditConfig,
    PrimitiveMjWarpGraspEditEnv,
)
from source.rl.grasp_edit.primitives import available_grasp_primitives
from source.rl.grasp_edit.templates import build_grasp_edit_templates


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--object-id", required=True)
    parser.add_argument(
        "--output-root", type=Path, default=Path("outputs/grasp_primitive_rl")
    )
    parser.add_argument(
        "--template-root", type=Path, default=Path("outputs/grasp_edit_lattice")
    )
    parser.add_argument("--ultra-root", type=Path, action="append", dest="ultra_roots")
    parser.add_argument("--ultra-seed-count", type=int, default=100)
    parser.add_argument("--ultra-generate-seeds", type=int, default=3)
    parser.add_argument("--ultra-max-execution-candidates", type=int, default=8)
    parser.add_argument("--no-auto-ultra", action="store_true")
    parser.add_argument("--failed-only", action="store_true")
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
    parser.add_argument("--hand-edit-fraction", type=float, default=0.35)
    parser.add_argument(
        "--grasp-primitives",
        default="wrap,pinch,support,hook",
        help=(
            "Comma-separated styles or 'all'. Available: "
            + ",".join(available_grasp_primitives())
        ),
    )
    parser.add_argument("--primitive-bias-scale", type=float, default=1.0)
    parser.add_argument("--success-lift-height", type=float, default=0.055)
    parser.add_argument("--success-tail-steps", type=int, default=8)
    parser.add_argument("--nconmax", type=int, default=192)
    parser.add_argument("--njmax", type=int, default=768)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--initial-std", type=float, default=0.70)
    parser.add_argument("--update-epochs", type=int, default=6)
    parser.add_argument("--minibatches", type=int, default=4)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--visualize", action="store_true")
    return parser


def _primitive_names(value: str) -> tuple[str, ...]:
    value = value.strip()
    if value == "all":
        return available_grasp_primitives()
    names = tuple(part.strip() for part in value.split(",") if part.strip())
    if not names:
        raise ValueError("--grasp-primitives cannot be empty.")
    unknown = sorted(set(names) - set(available_grasp_primitives()))
    if unknown:
        raise ValueError(
            f"Unknown grasp primitives {unknown}; available={available_grasp_primitives()}"
        )
    return tuple(dict.fromkeys(names))


def _trajectory_summary(trajectory) -> str:
    if trajectory is None:
        return ""
    meta = trajectory.metadata
    return (
        f"choice=c{meta.get('choice_id', '?')} "
        f"template=t{meta.get('template_id', '?')} "
        f"primitive={meta.get('grasp_primitive', '?')} "
        f"lift={float(meta.get('mjwarp_max_lift', 0.0)):.3f}m "
        f"final={float(meta.get('mjwarp_final_lift', 0.0)):.3f}m"
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

    env_config = PrimitiveGraspEditConfig(
        num_envs=args.num_envs,
        device=args.device,
        hand_edit_fraction=args.hand_edit_fraction,
        grasp_primitives=_primitive_names(args.grasp_primitives),
        primitive_bias_scale=args.primitive_bias_scale,
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
    env = PrimitiveMjWarpGraspEditEnv(args.object_id, templates, env_config)
    trainer = HybridPPOTrainer(env, ppo_config, seed=args.seed)

    if args.resume is not None:
        trainer.load(args.resume)
        print(f"resumed={args.resume} update={trainer.update_index}", flush=True)

    _write_json(
        output / "config.json",
        {
            "experiment": "primitive_conditioned_grasp_edit",
            "policy": {
                "choice": "categorical(wrist_template x grasp_primitive)",
                "hand": "tanh_gaussian_6d",
                "encoded_action_dim": env.action_dim,
            },
            "object_id": args.object_id,
            "failed_only": bool(args.failed_only),
            "environment": asdict(env_config),
            "ppo": asdict(ppo_config),
            "choices": env.template_summary(),
        },
    )

    print(
        f"[train] object={args.object_id} wrist_templates={env.base_template_count} "
        f"primitives={env.primitive_names} choices={env.template_count} "
        f"envs={env.num_envs} horizon={env.horizon} "
        f"action=Categorical({env.template_count})+Gaussian(6) updates={args.updates}",
        flush=True,
    )
    if args.verbose:
        for row in env.template_summary():
            print(
                f"  c{row['id']:02d} t{row['template_id']:02d}/{row['primitive']} "
                f"{row['label']} dxyz={[round(float(x), 3) for x in row['translation_offset']]} "
                f"rpy={[round(float(x), 1) for x in row['rotation_offset_degrees']]} "
                f"pre_rl_success={row['success_before_edit']}",
                flush=True,
            )

    persisted_attempt_version = 0
    persisted_best_version = 0

    def persist_best() -> str:
        nonlocal persisted_attempt_version, persisted_best_version
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
        top_choice = max(range(len(rates)), key=rates.__getitem__)
        primitive_rates = ",".join(
            f"{name}:{metrics.get(f'primitive_{name}_rate', 0.0):.0%}"
            for name in env.primitive_names
        )
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
                f"top c{top_choice}:{rates[top_choice]:5.1%} | styles {primitive_rates} | "
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
        from source.rl.residual.replay import replay_residual_trajectory

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
