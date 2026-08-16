"""Train grasp-aware residual PPO around one reference using a BC hand prior."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from source.rl.common.ppo import PPOConfig, PPOTrainer
from source.rl.imitation.guided_env import BCGuidedResidualLiftEnv, GuidedResidualConfig


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--bc-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-envs", type=int, default=512)
    parser.add_argument("--updates", type=int, default=80)
    parser.add_argument("--rollout-steps", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--initial-std", type=float, default=0.20)
    parser.add_argument("--entropy-coef", type=float, default=0.002)
    parser.add_argument("--update-epochs", type=int, default=4)
    parser.add_argument("--minibatches", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--hand-residual-fraction", type=float, default=0.12)
    parser.add_argument("--arm-residual-radians", type=float, default=0.04)
    parser.add_argument("--success-lift-height", type=float, default=0.055)
    parser.add_argument("--success-hold-steps", type=int, default=12)
    parser.add_argument("--maximum-object-speed", type=float, default=0.10)
    parser.add_argument("--nconmax", type=int, default=192)
    parser.add_argument("--njmax", type=int, default=768)
    parser.add_argument("--bc-approach-blend", type=float, default=0.15)
    parser.add_argument("--bc-close-blend", type=float, default=0.85)
    parser.add_argument("--bc-hold-blend", type=float, default=0.90)
    parser.add_argument("--bc-lift-blend", type=float, default=0.75)
    parser.add_argument("--bc-verify-blend", type=float, default=0.75)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--no-auto-resume", action="store_true")
    parser.add_argument("--no-replay", action="store_true")
    return parser


def _latest_checkpoint(output: Path) -> Path | None:
    final = output / "checkpoint_final.pt"
    if final.is_file():
        return final
    candidates = sorted((output / "checkpoints").glob("update_*.pt"))
    return candidates[-1] if candidates else None


def run(args: argparse.Namespace) -> int:
    if min(args.num_envs, args.updates, args.rollout_steps, args.save_every) <= 0:
        raise ValueError("num-envs, updates, rollout-steps and save-every must be positive.")
    args.output.mkdir(parents=True, exist_ok=True)

    env_config = GuidedResidualConfig(
        num_envs=args.num_envs,
        device=args.device,
        action_mode="arm_hand",
        start_stage="approach",
        hand_residual_fraction=args.hand_residual_fraction,
        arm_residual_radians=args.arm_residual_radians,
        nconmax=args.nconmax,
        njmax=args.njmax,
        success_lift_height=args.success_lift_height,
        success_hold_steps=args.success_hold_steps,
        maximum_object_speed=args.maximum_object_speed,
        bc_approach_blend=args.bc_approach_blend,
        bc_close_blend=args.bc_close_blend,
        bc_hold_blend=args.bc_hold_blend,
        bc_lift_blend=args.bc_lift_blend,
        bc_verify_blend=args.bc_verify_blend,
    )
    ppo_config = PPOConfig(
        rollout_steps=args.rollout_steps,
        update_epochs=args.update_epochs,
        minibatches=args.minibatches,
        learning_rate=args.learning_rate,
        initial_std=args.initial_std,
        entropy_coef=args.entropy_coef,
    )
    config_payload = {
        "schema_version": 1,
        "reference": str(args.reference),
        "bc_checkpoint": str(args.bc_checkpoint),
        "seed": args.seed,
        "environment": asdict(env_config),
        "ppo": asdict(ppo_config),
    }
    _atomic_json(args.output / "config.json", config_payload)

    env = BCGuidedResidualLiftEnv(args.reference, args.bc_checkpoint, env_config)
    trainer = PPOTrainer(env, ppo_config, seed=args.seed)
    resume = args.resume
    if resume is None and not args.no_auto_resume:
        resume = _latest_checkpoint(args.output)
    if resume is not None and resume.is_file():
        try:
            trainer.load(resume)
            print(f"[resume] checkpoint={resume} update={trainer.update_index}", flush=True)
        except ValueError as exc:
            print(f"[resume:skip] {exc}", flush=True)

    saved_best_version = 0
    saved_attempt_version = 0

    def persist() -> None:
        nonlocal saved_best_version, saved_attempt_version
        if env.best_attempt_trajectory is not None and env.best_attempt_version > saved_attempt_version:
            env.best_attempt_trajectory.save(args.output / "best_attempt")
            saved_attempt_version = env.best_attempt_version
        if env.best_trajectory is not None and env.best_version > saved_best_version:
            env.best_trajectory.save(args.output / "best_trajectory")
            saved_best_version = env.best_version

    def callback(active: PPOTrainer, metrics: dict) -> None:
        print(
            f"object={env.reference.object_id} update={metrics['update']:05d} "
            f"steps={metrics['total_steps']} reward={metrics['mean_reward']:.3f} "
            f"success={metrics['episode_success_rate']:.1%} "
            f"lift={metrics['mean_lift']:.3f}/{metrics['max_lift']:.3f}m "
            f"digits={metrics['mean_contact_digits']:.2f} "
            f"thumb={metrics['thumb_contact_rate']:.1%} "
            f"opp={metrics['opposition_rate']:.1%} stable={metrics['stable_rate']:.2%} "
            f"hold={metrics['max_hold_steps']:.0f} bc={metrics.get('bc_blend', 0.0):.2f} "
            f"kl={metrics['kl']:.4f}",
            flush=True,
        )
        persist()
        if active.update_index % args.save_every == 0:
            active.save(args.output / "checkpoints" / f"update_{active.update_index:05d}.pt")

    result = {
        "schema_version": 1,
        "object_id": env.reference.object_id,
        "reference": str(args.reference),
        "bc_checkpoint": str(args.bc_checkpoint),
        "status": "RL_NO_SUCCESS",
        "mjwarp_success": False,
        "replay": {},
    }
    try:
        print(
            f"[train] reference={env.reference.source_manifest} object={env.reference.object_id} "
            f"horizon={env.reference.horizon} envs={env.num_envs} obs={env.obs_dim} "
            f"action={env.action_dim} bc_hand_prior=6d updates={args.updates}",
            flush=True,
        )
        trainer.train(args.updates, callback=callback)
        trainer.save(args.output / "checkpoint_final.pt")
        persist()
        _atomic_json(args.output / "metrics.json", env.training_metrics())
        result["mjwarp_success"] = env.best_trajectory is not None
    finally:
        env.close()

    best = args.output / "best_trajectory" / "manifest.json"
    if best.is_file() and not args.no_replay:
        from source.rl.residual.replay import replay_residual_trajectory

        replay = replay_residual_trajectory(best, render_mode=None)
        replay_payload = asdict(replay)
        result["replay"] = replay_payload
        result["status"] = "VERIFIED_SUCCESS" if replay.success else "REPLAY_FAILED"
        print(
            f"[replay] success={replay.success} fraction={replay.success_fraction:.1%} "
            f"lift={replay.object_lift:.3f}m frames={replay.frames}",
            flush=True,
        )
    elif best.is_file():
        result["status"] = "MJWARP_SUCCESS_UNVERIFIED"
    else:
        result["status"] = "RL_NO_SUCCESS"

    _atomic_json(args.output / "result.json", result)
    print(f"[done] status={result['status']} result={args.output / 'result.json'}", flush=True)
    return 0 if result["status"] == "VERIFIED_SUCCESS" else 2


def main(argv: list[str] | None = None) -> int:
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
