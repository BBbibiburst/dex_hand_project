"""Measure cold and steady-state Diffusion Policy inference latency by stage."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from source.imitation.diffusion_policy import DiffusionPolicy


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--repetitions", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--image-width", type=int, default=640)
    parser.add_argument("--image-height", type=int, default=480)
    parser.add_argument("--inference-steps", type=int, action="append")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _synchronize(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize()


def _measure(function, *, repetitions: int, device: str) -> list[float]:
    values = []
    for _ in range(repetitions):
        _synchronize(device)
        started = time.perf_counter()
        function()
        _synchronize(device)
        values.append(1000.0 * (time.perf_counter() - started))
    return values


def _stats(values: list[float]) -> dict[str, float]:
    return {
        "mean_ms": float(np.mean(values)),
        "p50_ms": float(np.percentile(values, 50)),
        "p95_ms": float(np.percentile(values, 95)),
    }


def run(args: argparse.Namespace) -> dict:
    if args.repetitions <= 0 or args.warmup < 0:
        raise ValueError("repetitions must be positive and warmup must be non-negative.")
    torch.manual_seed(args.seed)
    checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    policy = DiffusionPolicy.from_checkpoint(checkpoint, args.device).eval()
    image_numpy = np.zeros((args.image_height, args.image_width, 3), dtype=np.uint8)
    tactile_numpy = np.zeros(policy.config.tactile_dim, dtype=np.float32)
    state_numpy = np.zeros(policy.config.state_dim, dtype=np.float32)

    def preprocess():
        return (
            torch.from_numpy(image_numpy).permute(2, 0, 1).float().div(255).unsqueeze(0),
            torch.from_numpy(tactile_numpy).unsqueeze(0),
            torch.from_numpy(state_numpy).unsqueeze(0),
        )

    def optimized_image_to_device():
        return (
            torch.from_numpy(image_numpy)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .to(device=args.device, dtype=torch.float32)
            .div_(255.0)
        )

    cpu_inputs = preprocess()

    def transfer():
        return tuple(value.to(args.device) for value in cpu_inputs)

    inputs = transfer()
    image, tactile, state = inputs
    normalized_state = (state - policy.state_mean) / policy.state_std
    normalized_tactile = (
        (tactile - policy.tactile_mean) / policy.tactile_std
        if policy.tactile is not None
        else tactile
    )

    def encode_condition():
        return policy.encode(image, tactile, state)

    condition = encode_condition()

    def ddpm_denoising():
        action = torch.zeros(
            1,
            policy.config.horizon,
            policy.config.action_dim,
            device=args.device,
        )
        for step in reversed(range(policy.config.diffusion_steps)):
            step_tensor = torch.full((1,), step, device=args.device, dtype=torch.long)
            noise = policy._predict_noise(action, step_tensor, condition)
            alpha = policy.alphas[step]
            alpha_bar = policy.alpha_bars[step]
            beta = policy.betas[step]
            action = (action - (1.0 - alpha) / torch.sqrt(1.0 - alpha_bar) * noise) / torch.sqrt(
                alpha
            )
            if step:
                action = action + beta.sqrt() * torch.zeros_like(action)
        return action

    with torch.inference_mode():
        cold_started = time.perf_counter()
        policy.sample(*inputs)
        _synchronize(args.device)
        cold_ms = 1000.0 * (time.perf_counter() - cold_started)
        for _ in range(args.warmup):
            policy.sample(*inputs)
        stages = {
            "image_preprocess_cpu": _stats(
                _measure(preprocess, repetitions=args.repetitions, device="cpu")
            ),
            "cpu_to_device": _stats(
                _measure(transfer, repetitions=args.repetitions, device=args.device)
            ),
            "optimized_image_preprocess_and_transfer": _stats(
                _measure(
                    optimized_image_to_device,
                    repetitions=args.repetitions,
                    device=args.device,
                )
            ),
            "visual_encoder": _stats(
                _measure(
                    lambda: policy.vision(image),
                    repetitions=args.repetitions,
                    device=args.device,
                )
            ),
            "state_encoder": _stats(
                _measure(
                    lambda: policy.state(normalized_state),
                    repetitions=args.repetitions,
                    device=args.device,
                )
            ),
            "observation_fusion": _stats(
                _measure(
                    encode_condition,
                    repetitions=args.repetitions,
                    device=args.device,
                )
            ),
            f"ddpm_denoising_{policy.config.diffusion_steps}_steps": _stats(
                _measure(
                    ddpm_denoising,
                    repetitions=args.repetitions,
                    device=args.device,
                )
            ),
        }
        if policy.tactile is not None:
            stages["tactile_encoder"] = _stats(
                _measure(
                    lambda: policy.tactile(normalized_tactile),
                    repetitions=args.repetitions,
                    device=args.device,
                )
            )
        step_counts = args.inference_steps or [policy.config.diffusion_steps]
        sampling = {}
        for steps in step_counts:
            selected = None if steps == policy.config.diffusion_steps else steps
            values = _measure(
                lambda selected=selected: policy.sample(*inputs, inference_steps=selected),
                repetitions=args.repetitions,
                device=args.device,
            )
            sampling[str(steps)] = _stats(values)

    payload = {
        "checkpoint": str(args.checkpoint),
        "device": args.device,
        "seed": args.seed,
        "repetitions": args.repetitions,
        "warmup": args.warmup,
        "cold_full_policy_ms": cold_ms,
        "stages": stages,
        "full_policy_by_inference_steps": sampling,
        "timing_note": "CUDA timings include an explicit synchronize before and after each sample.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return payload


def main(argv: list[str] | None = None) -> int:
    run(build_parser().parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
