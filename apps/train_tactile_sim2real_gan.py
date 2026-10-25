"""Train a small unpaired patch-statistic Sim-to-Real tactile GAN."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch

from apps.run_tactile_domain_baseline import _regional_features, feature_names
from apps.run_tactile_raw_domain_audit import STAGE_CODES, STAGES, _real_stage_frames


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _windows(frames: np.ndarray, size: int):
    return [frames[start : start + size] for start in range(0, len(frames) - size + 1, size)]


def build_dataset(real_root: Path, sim_root: Path, split_path: Path, window: int):
    split = json.loads(split_path.read_text(encoding="utf-8"))
    real_rows, sim_rows = [], []
    for session, subset in split["real_sessions"].items():
        trials = sorted((real_root / session).glob("trial_*"))[:2]
        for trial in trials:
            if not (trial / "tactile_processed.npz").is_file():
                continue
            for stage in STAGES:
                frames, _ = _real_stage_frames(trial, stage, "processed")
                if frames is None:
                    continue
                for index, values in enumerate(_windows(frames, window)):
                    real_rows.append(
                        {
                            "features": _regional_features(values, "real"),
                            "split": subset,
                            "group": session,
                            "stage": stage,
                            "unit": f"{session}/{trial.name}/{stage}/{index}",
                        }
                    )
    for manifest in sorted(sim_root.glob("mode_*/ycb_004_sugar_box/p*/candidate_00/manifest.json")):
        metadata = json.loads(manifest.read_text(encoding="utf-8"))
        root = metadata["root_id"]
        subset = split["sim_roots"][root]
        with np.load(manifest.parent / "tactile_dual.npz") as payload:
            stages = np.asarray(payload["stage"])
            tactile = np.asarray(payload["processed_tactile"], dtype=np.float32)
            digit = np.asarray(payload["robot_object_digit_contact_count"]) > 0
            for stage in STAGES:
                indices = np.flatnonzero(stages == STAGE_CODES[stage])
                for index, selected in enumerate(_windows(indices, window)):
                    contact = digit[selected]
                    sim_rows.append(
                        {
                            "features": _regional_features(tactile[selected], "sim"),
                            "labels": [
                                int(np.mean(np.any(contact, axis=1)) >= 0.5),
                                int(np.mean(np.sum(contact, axis=1) >= 2) >= 0.5),
                                int(np.mean(contact[:, 4] & np.any(contact[:, :4], axis=1)) >= 0.5),
                            ],
                            "split": subset,
                            "group": root,
                            "stage": stage,
                            "unit": f"{metadata['mode']}/{root}/{stage}/{index}",
                        }
                    )
    return real_rows, sim_rows, split


class DomainNormalizer:
    def fit(self, values: np.ndarray):
        self.median = np.median(values, axis=0).astype(np.float32)
        q25, q75 = np.quantile(values, (0.25, 0.75), axis=0)
        self.scale = np.where(q75 - q25 > 1e-6, q75 - q25, 1.0).astype(np.float32)
        return self

    def transform(self, values: np.ndarray):
        return np.clip((values - self.median) / self.scale, -20.0, 20.0).astype(np.float32)

    def inverse_tensor(self, values: torch.Tensor):
        median = torch.as_tensor(self.median, device=values.device)
        scale = torch.as_tensor(self.scale, device=values.device)
        return values * scale + median

    def state(self):
        return {"median": self.median.tolist(), "scale": self.scale.tolist()}


class Generator(torch.nn.Module):
    def __init__(self, dimension=127, hidden=128):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(dimension, hidden),
            torch.nn.LeakyReLU(0.1),
            torch.nn.Linear(hidden, hidden),
            torch.nn.LeakyReLU(0.1),
            torch.nn.Linear(hidden, dimension),
        )

    def forward(self, value):
        return value + self.net(value)


class Discriminator(torch.nn.Module):
    def __init__(self, dimension=127, hidden=128):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(dimension, hidden),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Linear(hidden, 64),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Linear(64, 1),
        )

    def forward(self, value):
        return self.net(value).squeeze(-1)


def _mmd_mean_variance(first: torch.Tensor, second: torch.Tensor):
    mean = torch.mean((first.mean(0) - second.mean(0)) ** 2)
    variance = torch.mean((first.var(0, unbiased=False) - second.var(0, unbiased=False)) ** 2)
    return mean + 0.1 * variance


def train_seed(
    real_rows,
    sim_rows,
    output: Path,
    seed: int,
    epochs: int,
    batch_size: int,
    patience_limit: int,
    content_weight: float,
):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    real = np.stack([row["features"] for row in real_rows])
    sim = np.stack([row["features"] for row in sim_rows])
    real_split = np.asarray([row["split"] for row in real_rows])
    sim_split = np.asarray([row["split"] for row in sim_rows])
    real_norm = DomainNormalizer().fit(real[real_split == "train"])
    sim_norm = DomainNormalizer().fit(sim[sim_split == "train"])
    real_z, sim_z = real_norm.transform(real), sim_norm.transform(sim)
    train_real = torch.from_numpy(real_z[real_split == "train"])
    train_sim = torch.from_numpy(sim_z[sim_split == "train"])
    val_real = torch.from_numpy(real_z[real_split == "validation"])
    val_sim = torch.from_numpy(sim_z[sim_split == "validation"])
    generator, discriminator = Generator(), Discriminator()
    opt_g = torch.optim.Adam(generator.parameters(), lr=2e-4, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=2e-4, betas=(0.5, 0.999))
    loss_fn = torch.nn.BCEWithLogitsLoss()
    ratio_indices = torch.as_tensor(
        [region * 11 for region in range(11)] + list(range(121, 127)), dtype=torch.long
    )
    curves, best, best_state, patience = [], float("inf"), None, patience_limit
    rng = np.random.default_rng(seed)
    steps = max(1, int(np.ceil(max(len(train_real), len(train_sim)) / batch_size)))
    for epoch in range(1, epochs + 1):
        d_losses, g_losses = [], []
        for _ in range(steps):
            real_batch = train_real[rng.integers(0, len(train_real), size=batch_size)]
            sim_batch = train_sim[rng.integers(0, len(train_sim), size=batch_size)]
            with torch.no_grad():
                fake = generator(sim_batch)
            opt_d.zero_grad()
            d_loss = loss_fn(discriminator(real_batch), torch.ones(batch_size)) + loss_fn(
                discriminator(fake), torch.zeros(batch_size)
            )
            d_loss.backward()
            opt_d.step()
            opt_g.zero_grad()
            fake = generator(sim_batch)
            generated_raw = real_norm.inverse_tensor(fake)
            sim_raw = sim_norm.inverse_tensor(sim_batch)
            content = torch.mean((generated_raw[:, ratio_indices] - sim_raw[:, ratio_indices]) ** 2)
            g_loss = loss_fn(discriminator(fake), torch.ones(batch_size)) + content_weight * content
            g_loss.backward()
            torch.nn.utils.clip_grad_norm_(generator.parameters(), 5.0)
            opt_g.step()
            d_losses.append(float(d_loss.detach()))
            g_losses.append(float(g_loss.detach()))
        generator.eval()
        with torch.no_grad():
            generated_val = generator(val_sim)
            validation = float(_mmd_mean_variance(generated_val, val_real))
            content_val = float(
                torch.mean(
                    (
                        real_norm.inverse_tensor(generated_val)[:, ratio_indices]
                        - sim_norm.inverse_tensor(val_sim)[:, ratio_indices]
                    )
                    ** 2
                )
            )
            score = validation + content_weight * content_val
        generator.train()
        curves.append(
            {
                "epoch": epoch,
                "d_loss": float(np.mean(d_losses)),
                "g_loss": float(np.mean(g_losses)),
                "validation_distribution": validation,
                "validation_content": content_val,
                "selection_score": score,
            }
        )
        if np.isfinite(score) and score < best - 1e-6:
            best, patience = score, patience_limit
            best_state = {
                name: value.detach().clone() for name, value in generator.state_dict().items()
            }
        else:
            patience -= 1
            if patience == 0:
                break
    if best_state is None:
        raise RuntimeError("GAN produced no finite validation checkpoint.")
    generator.load_state_dict(best_state)
    checkpoint = {
        "schema_version": 1,
        "seed": seed,
        "generator": generator.state_dict(),
        "real_normalizer": real_norm.state(),
        "sim_normalizer": sim_norm.state(),
        "feature_names": feature_names(),
        "hidden": 128,
        "content_weight": content_weight,
        "best_validation_score": best,
        "epochs_completed": len(curves),
    }
    output.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output / "checkpoint.pt")
    _write_json(output / "training_curve.json", curves)
    return {
        "seed": seed,
        "best_validation_score": best,
        "epochs_completed": len(curves),
        "checkpoint": str(output / "checkpoint.pt"),
        "checkpoint_sha256": _sha256(output / "checkpoint.pt"),
    }


def run(args):
    real_rows, sim_rows, split = build_dataset(
        args.real_root, args.sim_root, args.frozen_split, args.window
    )
    args.output.mkdir(parents=True, exist_ok=True)
    dataset_path = args.output / "dataset.npz"
    np.savez_compressed(
        dataset_path,
        real_features=np.stack([row["features"] for row in real_rows]),
        real_split=np.asarray([row["split"] for row in real_rows]),
        real_group=np.asarray([row["group"] for row in real_rows]),
        real_stage=np.asarray([row["stage"] for row in real_rows]),
        sim_features=np.stack([row["features"] for row in sim_rows]),
        sim_labels=np.asarray([row["labels"] for row in sim_rows], dtype=np.int8),
        sim_split=np.asarray([row["split"] for row in sim_rows]),
        sim_group=np.asarray([row["group"] for row in sim_rows]),
        sim_stage=np.asarray([row["stage"] for row in sim_rows]),
    )
    runs = [
        train_seed(
            real_rows,
            sim_rows,
            args.output / f"seed_{seed}",
            seed,
            args.epochs,
            args.batch_size,
            args.patience,
            args.content_weight,
        )
        for seed in args.seeds
    ]
    manifest = {
        "schema_version": 1,
        "direction": "sim_processed_to_real_like_processed",
        "representation": "127D orientation-invariant patch statistics",
        "window": args.window,
        "frozen_split": str(args.frozen_split),
        "frozen_split_sha256": _sha256(args.frozen_split),
        "dataset_sha256": _sha256(dataset_path),
        "real_windows": len(real_rows),
        "sim_windows": len(sim_rows),
        "split_counts_real": dict(
            __import__("collections").Counter(row["split"] for row in real_rows)
        ),
        "split_counts_sim": dict(
            __import__("collections").Counter(row["split"] for row in sim_rows)
        ),
        "epochs_max": args.epochs,
        "patience": args.patience,
        "batch_size": args.batch_size,
        "content_weight": args.content_weight,
        "runs": runs,
        "limitations": [
            "Unpaired training",
            "No verified per-taxel spatial correspondence",
            "Real processed physical meaning is unknown",
            "Generator operates on aggregates, not 712D frames",
        ],
    }
    _write_json(args.output / "training_manifest.json", manifest)
    return 0


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real-root", type=Path, default=Path("outputs/realdata"))
    parser.add_argument(
        "--sim-root",
        type=Path,
        default=Path("outputs/tactile_p18_2_2026-10-25/raw_domain_audit/sim_dual"),
    )
    parser.add_argument(
        "--frozen-split",
        type=Path,
        default=Path("outputs/tactile_p18_2_2026-10-25/diagnostic_baseline/split_manifest.json"),
    )
    parser.add_argument("--output", type=Path, default=Path("outputs/tactile_p19_gan_2026-10-25"))
    parser.add_argument("--window", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--patience", type=int, default=60)
    parser.add_argument("--content-weight", type=float, default=8.0)
    parser.add_argument("--seeds", type=int, nargs="+", default=(0, 1, 2))
    return parser


if __name__ == "__main__":
    raise SystemExit(run(build_parser().parse_args()))
