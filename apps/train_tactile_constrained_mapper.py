"""Train constrained 127D tactile mappers using train/validation data only."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from apps.run_tactile_constrained_mapping import (
    STAGES,
    _candidate_score,
    _domain_evaluation,
    _task_evaluation,
    _write_json,
    contract_violations,
)


class ConstrainedMapper(torch.nn.Module):
    def __init__(self, median: np.ndarray, scale: np.ndarray, hidden: int = 256):
        super().__init__()
        self.register_buffer("median", torch.as_tensor(median, dtype=torch.float32))
        self.register_buffer("scale", torch.as_tensor(scale, dtype=torch.float32))
        self.net = torch.nn.Sequential(
            torch.nn.Linear(130, hidden),
            torch.nn.LeakyReLU(0.1),
            torch.nn.Linear(hidden, hidden),
            torch.nn.LeakyReLU(0.1),
            torch.nn.Linear(hidden, 127),
        )

    @staticmethod
    def enforce(raw: torch.Tensor) -> torch.Tensor:
        regions = []
        for region in range(11):
            start = region * 11
            nonzero = torch.sigmoid(raw[:, start : start + 1])
            positive = torch.nn.functional.softplus(raw[:, start + 1 : start + 3])
            base = torch.nn.functional.softplus(raw[:, start + 3 : start + 4])
            increments = torch.nn.functional.softplus(raw[:, start + 4 : start + 8])
            quantiles = torch.cat((base, base + torch.cumsum(increments, dim=1)), dim=1)
            temporal = torch.nn.functional.softplus(raw[:, start + 8 : start + 10])
            autocorrelation = torch.tanh(raw[:, start + 10 : start + 11])
            regions.append(
                torch.cat((nonzero, positive, quantiles, temporal, autocorrelation), dim=1)
            )
        shares = torch.softmax(raw[:, 121:127], dim=1)
        return torch.cat((*regions, shares), dim=1)

    def forward(self, normalized_sim: torch.Tensor, stage_one_hot: torch.Tensor) -> torch.Tensor:
        raw = self.net(torch.cat((normalized_sim, stage_one_hot), dim=1))
        scaled = raw * self.scale + self.median
        return self.enforce(scaled)


class DomainAdversary(torch.nn.Module):
    def __init__(self, hidden: int = 128):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(127, hidden),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Linear(hidden, 64),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Linear(64, 1),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.net(value).squeeze(-1)


class TaskHead(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(127, 64), torch.nn.ReLU(), torch.nn.Linear(64, 3)
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.net(value)


def _stage_one_hot(stages: np.ndarray) -> np.ndarray:
    return np.stack([(stages == stage).astype(np.float32) for stage in STAGES], axis=1)


def _moments_loss(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    mean = torch.mean((first.mean(0) - second.mean(0)) ** 2)
    first_centered = first - first.mean(0)
    second_centered = second - second.mean(0)
    first_cov = first_centered.T @ first_centered / max(len(first) - 1, 1)
    second_cov = second_centered.T @ second_centered / max(len(second) - 1, 1)
    covariance = torch.mean((first_cov - second_cov) ** 2)
    return mean + 0.05 * covariance


def _normalizer(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    median = np.median(values, axis=0).astype(np.float32)
    q25, q75 = np.quantile(values, (0.25, 0.75), axis=0)
    scale = np.where(q75 - q25 > 1e-5, q75 - q25, 1.0).astype(np.float32)
    return median, scale


def train_candidate(
    sim: np.ndarray,
    real: np.ndarray,
    labels: np.ndarray,
    sim_stage: np.ndarray,
    real_stage: np.ndarray,
    sim_split: np.ndarray,
    real_split: np.ndarray,
    *,
    seed: int,
    adversarial_weight: float,
    task_weight: float,
    epochs: int,
) -> tuple[ConstrainedMapper, list[dict]]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    sim_train = sim[sim_split == "train"]
    real_train = real[real_split == "train"]
    label_train = labels[sim_split == "train"].astype(np.float32)
    sim_stage_train = sim_stage[sim_split == "train"]
    real_stage_train = real_stage[real_split == "train"]
    sim_median, sim_scale = _normalizer(sim_train)
    real_median, real_scale = _normalizer(real_train)
    sim_z = torch.from_numpy(
        np.clip((sim_train - sim_median) / sim_scale, -20, 20).astype(np.float32)
    )
    real_z = torch.from_numpy(
        np.clip((real_train - real_median) / real_scale, -20, 20).astype(np.float32)
    )
    stage_tensor = torch.from_numpy(_stage_one_hot(sim_stage_train))
    label_tensor = torch.from_numpy(label_train)
    mapper = ConstrainedMapper(real_median, real_scale)
    adversary = DomainAdversary()
    task_head = TaskHead()
    optimizer_mapper = torch.optim.AdamW(
        list(mapper.parameters()) + list(task_head.parameters()), lr=5e-4, weight_decay=1e-5
    )
    optimizer_adversary = torch.optim.AdamW(adversary.parameters(), lr=5e-4, weight_decay=1e-5)
    domain_loss = torch.nn.BCEWithLogitsLoss()
    positive = label_tensor.sum(0)
    negative = len(label_tensor) - positive
    positive_weight = torch.clamp(negative / torch.clamp(positive, min=1.0), min=0.2, max=20.0)
    curves = []
    rng = np.random.default_rng(seed)
    for epoch in range(1, epochs + 1):
        generated = mapper(sim_z, stage_tensor)
        generated_z = (generated - mapper.median) / mapper.scale
        optimizer_adversary.zero_grad()
        count = min(len(real_z), len(generated_z))
        real_index = torch.from_numpy(rng.integers(0, len(real_z), count))
        sim_index = torch.from_numpy(rng.integers(0, len(generated_z), count))
        discriminator = domain_loss(adversary(real_z[real_index]), torch.ones(count)) + domain_loss(
            adversary(generated_z.detach()[sim_index]), torch.zeros(count)
        )
        discriminator.backward()
        optimizer_adversary.step()

        optimizer_mapper.zero_grad()
        generated = mapper(sim_z, stage_tensor)
        generated_z = (generated - mapper.median) / mapper.scale
        stage_loss = torch.tensor(0.0)
        for stage in STAGES:
            sim_mask = torch.from_numpy(sim_stage_train == stage)
            real_mask = torch.from_numpy(real_stage_train == stage)
            stage_loss = stage_loss + _moments_loss(generated_z[sim_mask], real_z[real_mask])
        fool = domain_loss(adversary(generated_z), torch.ones(len(generated_z)))
        task = torch.nn.functional.binary_cross_entropy_with_logits(
            task_head(generated_z), label_tensor, pos_weight=positive_weight
        )
        loss = stage_loss + adversarial_weight * fool + task_weight * task
        loss.backward()
        torch.nn.utils.clip_grad_norm_(mapper.parameters(), 5.0)
        optimizer_mapper.step()
        if epoch == 1 or epoch % 25 == 0 or epoch == epochs:
            curves.append(
                {
                    "epoch": epoch,
                    "mapper_loss": float(loss.detach()),
                    "stage_moments": float(stage_loss.detach()),
                    "adversarial": float(fool.detach()),
                    "task": float(task.detach()),
                    "discriminator": float(discriminator.detach()),
                }
            )
    mapper.sim_median = sim_median
    mapper.sim_scale = sim_scale
    return mapper, curves


def _transform(mapper: ConstrainedMapper, sim: np.ndarray, stages: np.ndarray) -> np.ndarray:
    normalized = np.clip((sim - mapper.sim_median) / mapper.sim_scale, -20, 20).astype(np.float32)
    with torch.inference_mode():
        return mapper(
            torch.from_numpy(normalized), torch.from_numpy(_stage_one_hot(stages))
        ).numpy()


def run(args: argparse.Namespace) -> int:
    with np.load(args.dataset) as payload:
        real = np.asarray(payload["real_features"], dtype=np.float32)
        sim = np.asarray(payload["sim_features"], dtype=np.float32)
        labels = np.asarray(payload["sim_labels"], dtype=np.int64)
        real_split = np.asarray(payload["real_split"])
        sim_split = np.asarray(payload["sim_split"])
        real_stage = np.asarray(payload["real_stage"])
        sim_stage = np.asarray(payload["sim_stage"])
    args.output.mkdir(parents=True, exist_ok=True)
    candidates = []
    configurations = (
        ("B", 0.0, 1.0, 0, 300),
        ("C", 0.2, 1.0, 1, 400),
        ("C", 1.0, 1.0, 2, 400),
        ("C", 3.0, 1.0, 3, 500),
        ("C", 1.0, 3.0, 4, 400),
    )
    for route, adversarial_weight, task_weight, seed, epochs in configurations:
        mapper, curve = train_candidate(
            sim,
            real,
            labels,
            sim_stage,
            real_stage,
            sim_split,
            real_split,
            seed=seed,
            adversarial_weight=adversarial_weight,
            task_weight=task_weight,
            epochs=epochs,
        )
        mapped = _transform(mapper, sim, sim_stage)
        domain = _domain_evaluation(real, mapped, real_split, sim_split, "validation", (2201,))
        task = _task_evaluation(sim, mapped, labels, sim_split, "validation")
        score, selection = _candidate_score(domain, task)
        candidate = {
            "route": route,
            "seed": seed,
            "adversarial_weight": adversarial_weight,
            "task_weight": task_weight,
            "epochs": epochs,
            "selection_score": score,
            "selection": selection,
            "contract": contract_violations(mapped),
            "domain_validation": domain,
            "task_validation": task,
            "curve": curve,
        }
        checkpoint = {
            "model": mapper.state_dict(),
            "sim_median": mapper.sim_median,
            "sim_scale": mapper.sim_scale,
            "hidden": 256,
            "candidate": candidate,
        }
        torch.save(checkpoint, args.output / f"candidate_{route}_{seed}.pt")
        candidates.append(candidate)
    candidates.sort(key=lambda item: item["selection_score"])
    _write_json(args.output / "route_bc_candidates.json", candidates)
    best = candidates[0]
    validation_target_met = bool(
        all(item["domain_confusion"] >= 0.90 for item in best["domain_validation"])
        and best["selection"]["mapped_task_floor"] >= 0.95
        and best["contract"]["finite"]
        and all(
            value == 0.0
            for key, value in best["contract"].items()
            if key.endswith("violation_fraction")
        )
    )
    _write_json(
        args.output / "locked_learned_candidate.json",
        {
            "schema_version": 1,
            "selection_uses_test": False,
            "locked_candidate": best,
            "validation_target_met": validation_target_met,
        },
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", type=Path, default=Path("outputs/tactile_p19_gan_2026-10-25/dataset.npz")
    )
    parser.add_argument("--output", type=Path, default=Path("outputs/tactile_p22_2026-10-25"))
    return parser


if __name__ == "__main__":
    raise SystemExit(run(build_parser().parse_args()))
