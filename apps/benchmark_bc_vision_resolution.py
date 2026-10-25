"""Benchmark an actual BC checkpoint at multiple native image resolutions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch

from source.imitation.action_chunk_bc import ActionChunkBC


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--frames", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--repeats", type=int, default=40)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--training-repeats", type=int, default=5)
    return parser


def _sync(device):
    if device.startswith("cuda"):
        torch.cuda.synchronize()


def _image(path, device):
    array = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
    return torch.from_numpy(array.copy()).permute(2, 0, 1).float().div(255).unsqueeze(0).to(device)


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    model = ActionChunkBC.from_checkpoint(checkpoint, args.device).eval()
    state = model.state_mean.detach().unsqueeze(0)
    rows = []
    for width, height in ((160, 120), (320, 240), (640, 480), (1920, 1080)):
        image = _image(args.frames / f"close_{width}x{height}.png", args.device)
        with torch.inference_mode():
            for _ in range(args.warmup):
                model.sample(image, state)
            _sync(args.device)
            times = []
            if args.device.startswith("cuda"):
                torch.cuda.reset_peak_memory_stats()
                baseline = torch.cuda.memory_allocated()
            else:
                baseline = 0
            for _ in range(args.repeats):
                started = time.perf_counter()
                model.sample(image, state)
                _sync(args.device)
                times.append(1000 * (time.perf_counter() - started))
            inference_peak = (
                torch.cuda.max_memory_allocated() - baseline
                if args.device.startswith("cuda")
                else None
            )

        train_image = image.detach().requires_grad_(False)
        target = model.sample(image, state).detach()
        for _ in range(2):
            model.zero_grad(set_to_none=True)
            model.loss(train_image, state, target).backward()
        _sync(args.device)
        model.zero_grad(set_to_none=True)
        if args.device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats()
            train_baseline = torch.cuda.memory_allocated()
        else:
            train_baseline = 0
        train_times = []
        for _ in range(args.training_repeats):
            model.zero_grad(set_to_none=True)
            started = time.perf_counter()
            model.loss(train_image, state, target).backward()
            _sync(args.device)
            train_times.append(1000 * (time.perf_counter() - started))
        train_peak = (
            torch.cuda.max_memory_allocated() - train_baseline
            if args.device.startswith("cuda")
            else None
        )
        model.zero_grad(set_to_none=True)
        rows.append(
            {
                "resolution_wh": [width, height],
                "pixels": width * height,
                "inference_mean_ms": float(np.mean(times)),
                "inference_p50_ms": float(np.percentile(times, 50)),
                "inference_p95_ms": float(np.percentile(times, 95)),
                "inference_peak_increment_mib": (
                    None if inference_peak is None else inference_peak / 2**20
                ),
                "single_sample_forward_backward_p50_ms": float(
                    np.percentile(train_times, 50)
                ),
                "single_sample_forward_backward_p95_ms": float(
                    np.percentile(train_times, 95)
                ),
                "single_sample_train_peak_increment_mib": (
                    None if train_peak is None else train_peak / 2**20
                ),
            }
        )
    parameters = sum(parameter.numel() for parameter in model.parameters())
    summary = {
        "checkpoint": str(args.checkpoint),
        "device": args.device,
        "device_name": (
            torch.cuda.get_device_name(args.device) if args.device.startswith("cuda") else "CPU"
        ),
        "parameters": parameters,
        "encoder_accepts_variable_resolution": True,
        "adaptive_pooling_warning": (
            "The current encoder accepts arbitrary shapes but globally pools the final feature map; "
            "higher pixel count does not guarantee preservation of local contact detail."
        ),
        "rows": rows,
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
