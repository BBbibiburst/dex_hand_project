"""BC-only rollout validation on held-out expert objects."""
from __future__ import annotations

from pathlib import Path

import torch

from source.rl.imitation.guided_env import BCGuidedResidualLiftEnv, GuidedResidualConfig


def evaluate_bc_checkpoint(
    manifests,
    *,
    checkpoint: str | Path,
    device: str = "cuda:0",
    maximum_objects: int = 4,
    nconmax: int = 192,
    njmax: int = 768,
) -> dict:
    selected = list(manifests)[: max(1, int(maximum_objects))]
    rows = []
    for manifest in selected:
        cfg = GuidedResidualConfig(
            num_envs=1,
            device=device,
            action_mode="arm_hand",
            nconmax=nconmax,
            njmax=njmax,
            arm_approach_gate=0.0,
            arm_close_gate=0.0,
            arm_hold_gate=0.0,
            arm_lift_gate=0.0,
            arm_verify_gate=0.0,
            locked_arm_lift_gate=0.0,
            hand_approach_gate=0.0,
            hand_close_gate=0.0,
            hand_hold_gate=0.0,
            hand_lift_gate=0.0,
            hand_verify_gate=0.0,
            locked_hand_lift_gate=0.0,
        )
        env = BCGuidedResidualLiftEnv(manifest, checkpoint, cfg)
        try:
            zero = torch.zeros((1, env.action_dim), device=env.torch_device)
            env.reset()
            for _ in range(env.reference.horizon):
                env.step(zero)
            metrics = env.training_metrics()
            success = env.best_trajectory is not None
            rows.append(
                {
                    "object_id": env.reference.object_id,
                    "manifest": str(manifest),
                    "success": bool(success),
                    "opposition_rate": float(metrics.get("opposition_rate", 0.0)),
                    "mean_contact_digits": float(metrics.get("mean_contact_digits", 0.0)),
                    "max_lift": float(metrics.get("max_lift", 0.0)),
                }
            )
        finally:
            env.close()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    rate = sum(bool(row["success"]) for row in rows) / max(len(rows), 1)
    return {"objects": len(rows), "success_rate": float(rate), "results": rows}
