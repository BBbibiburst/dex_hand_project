"""Stage-conditional empirical transport using train-only real prototypes."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from apps.run_tactile_constrained_mapping import (
    STAGES,
    _domain_evaluation,
    _task_evaluation,
    _write_json,
    contract_violations,
    project_contract,
)
from apps.run_tactile_domain_baseline import _train_probe


class EmpiricalTaskTransport:
    def __init__(self, thumb_weight: float = 0.0, nonzero_weight: float = 0.0):
        self.thumb_weight = thumb_weight
        self.nonzero_weight = nonzero_weight
        self.models = []
        self.sim_median = None
        self.sim_scale = None
        self.score_quantiles = {}
        self.real_prototypes = {}

    def fit(
        self,
        sim: np.ndarray,
        labels: np.ndarray,
        stages: np.ndarray,
        real: np.ndarray,
        real_stages: np.ndarray,
    ):
        self.sim_median = np.median(sim, axis=0).astype(np.float32)
        q25, q75 = np.quantile(sim, (0.25, 0.75), axis=0)
        self.sim_scale = np.where(q75 - q25 > 1e-6, q75 - q25, 1.0).astype(np.float32)
        normalized = np.clip((sim - self.sim_median) / self.sim_scale, -20, 20).astype(np.float32)
        # Use a fixed tail of training groups as an internal validation subset.
        split = max(1, int(round(0.8 * len(sim))))
        for label_index in range(3):
            model = _train_probe(
                normalized[:split],
                labels[:split, label_index],
                normalized[split:],
                labels[split:, label_index],
                kind="logistic",
                seed=2210 + label_index,
            )
            self.models.append(model)
        score = self._score(sim)
        probabilities = np.linspace(0.0, 1.0, 101)
        activation_indices = np.asarray([region * 11 + 1 for region in range(11)])
        for stage in STAGES:
            self.score_quantiles[stage] = np.quantile(score[stages == stage], probabilities)
            stage_real = project_contract(real[real_stages == stage])
            activation = stage_real[:, activation_indices].sum(axis=1)
            nonzero = stage_real[:, np.arange(0, 11 * 11, 11)].sum(axis=1)
            thumb_share = stage_real[:, 125]

            def standardize(values: np.ndarray) -> np.ndarray:
                scale = np.std(values)
                return (values - np.mean(values)) / (scale if scale > 1e-6 else 1.0)

            prototype_score = (
                standardize(activation)
                + self.nonzero_weight * standardize(nonzero)
                + self.thumb_weight * standardize(thumb_share)
            )
            self.real_prototypes[stage] = stage_real[np.argsort(prototype_score)]
        return self

    def _score(self, sim: np.ndarray) -> np.ndarray:
        normalized = np.clip((sim - self.sim_median) / self.sim_scale, -20, 20).astype(np.float32)
        with torch.inference_mode():
            probabilities = np.stack(
                [
                    torch.sigmoid(model(torch.from_numpy(normalized))).numpy()
                    for model in self.models
                ],
                axis=1,
            )
        # Lexicographic task score: contact, multi-finger, thumb opposition.
        return probabilities @ np.asarray([1.0, 2.0, 4.0])

    def transform(self, sim: np.ndarray, stages: np.ndarray) -> np.ndarray:
        score = self._score(sim)
        probability_grid = np.linspace(0.0, 1.0, 101)
        output = np.empty_like(sim, dtype=np.float32)
        for stage in STAGES:
            selected = stages == stage
            probability = np.interp(score[selected], self.score_quantiles[stage], probability_grid)
            prototypes = self.real_prototypes[stage]
            indices = np.clip(
                np.round(probability * (len(prototypes) - 1)).astype(int),
                0,
                len(prototypes) - 1,
            )
            output[selected] = prototypes[indices]
        return output


def _selection_score(domain: list[dict], task: list[dict]) -> tuple[float, dict]:
    domain_distance = max(abs(item["balanced_accuracy"] - 0.5) for item in domain)
    raw = {item["label"]: item["balanced_accuracy"] for item in task if item["source"] == "raw"}
    mapped = {
        item["label"]: item["balanced_accuracy"] for item in task if item["source"] == "mapped"
    }
    absolute_shortfall = max(max(0.0, 0.95 - value) for value in mapped.values())
    relative_shortfall = max(max(0.0, raw[key] - 0.03 - mapped[key]) for key in mapped)
    score = domain_distance + 10.0 * absolute_shortfall + 10.0 * relative_shortfall
    return score, {
        "max_domain_distance": domain_distance,
        "mapped_task_floor": min(mapped.values()),
        "max_absolute_task_shortfall": absolute_shortfall,
        "max_relative_task_shortfall": relative_shortfall,
    }


def _target_met(candidate: dict) -> bool:
    return bool(
        all(item["domain_confusion"] >= 0.90 for item in candidate["domain_validation"])
        and candidate["selection"]["max_absolute_task_shortfall"] == 0.0
        and candidate["selection"]["max_relative_task_shortfall"] == 0.0
        and candidate["contract"]["finite"]
        and all(
            value == 0.0
            for key, value in candidate["contract"].items()
            if key.endswith("violation_fraction")
        )
    )


def run(args: argparse.Namespace) -> int:
    with np.load(args.dataset) as payload:
        real = np.asarray(payload["real_features"], dtype=np.float32)
        sim = np.asarray(payload["sim_features"], dtype=np.float32)
        labels = np.asarray(payload["sim_labels"], dtype=np.int64)
        real_split = np.asarray(payload["real_split"])
        sim_split = np.asarray(payload["sim_split"])
        real_stage = np.asarray(payload["real_stage"])
        sim_stage = np.asarray(payload["sim_stage"])
        real_group = np.asarray(payload["real_group"])
        sim_group = np.asarray(payload["sim_group"])
    args.output.mkdir(parents=True, exist_ok=True)
    candidates = []
    for thumb_weight in (0.0, 0.5, 1.0, 2.0, 4.0, 8.0):
        for nonzero_weight in (0.0, 0.5, 1.0):
            mapper = EmpiricalTaskTransport(thumb_weight, nonzero_weight).fit(
                sim[sim_split == "train"],
                labels[sim_split == "train"],
                sim_stage[sim_split == "train"],
                real[real_split == "train"],
                real_stage[real_split == "train"],
            )
            mapped = mapper.transform(sim, sim_stage)
            domain = _domain_evaluation(
                real, mapped, real_split, sim_split, "validation", (2201,)
            )
            task = _task_evaluation(sim, mapped, labels, sim_split, "validation")
            score, selection = _selection_score(domain, task)
            candidates.append(
                {
                    "route": "empirical_task_transport",
                    "thumb_weight": thumb_weight,
                    "nonzero_weight": nonzero_weight,
                    "selection_score": score,
                    "selection": selection,
                    "contract": contract_violations(mapped),
                    "domain_validation": domain,
                    "task_validation": task,
                    "unique_output_rows": int(len(np.unique(mapped, axis=0))),
                    "uses_test_statistics": False,
                }
            )
    candidates.sort(key=lambda item: item["selection_score"])
    candidate = candidates[0]
    validation_target_met = _target_met(candidate)
    _write_json(args.output / "empirical_transport_candidates.json", candidates)
    _write_json(
        args.output / "empirical_transport_validation.json",
        {
            "schema_version": 1,
            "candidate": candidate,
            "validation_target_met": validation_target_met,
            "limitations": [
                "Output is a train-real empirical prototype, not a 712D tactile frame.",
                "Task score is inferred from Sim features; no ground-truth label is used at transform time.",
            ],
        },
    )
    mapper = EmpiricalTaskTransport(
        candidate["thumb_weight"], candidate["nonzero_weight"]
    ).fit(
        sim[sim_split == "train"],
        labels[sim_split == "train"],
        sim_stage[sim_split == "train"],
        real[real_split == "train"],
        real_stage[real_split == "train"],
    )
    mapped = mapper.transform(sim, sim_stage)
    np.savez_compressed(args.output / "empirical_mapped_features.npz", mapped=mapped)
    if args.phase == "final":
        if not validation_target_met:
            raise RuntimeError("No empirical candidate passed validation; do not evaluate test.")
        final = {
            "candidate": {
                "thumb_weight": candidate["thumb_weight"],
                "nonzero_weight": candidate["nonzero_weight"],
            },
            "domain_test": _domain_evaluation(
                real,
                mapped,
                real_split,
                sim_split,
                "test",
                (11, 12, 13),
                real_group,
                sim_group,
            ),
            "task_test": _task_evaluation(sim, mapped, labels, sim_split, "test"),
            "contract": contract_violations(mapped),
            "unique_output_rows": int(len(np.unique(mapped, axis=0))),
        }
        final["domain_test_by_stage"] = {
            stage: _domain_evaluation(
                real[real_stage == stage],
                mapped[sim_stage == stage],
                real_split[real_stage == stage],
                sim_split[sim_stage == stage],
                "test",
                (11, 12, 13),
                real_group[real_stage == stage],
                sim_group[sim_stage == stage],
            )
            for stage in STAGES
        }
        _write_json(args.output / "empirical_final_test.json", final)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", type=Path, default=Path("outputs/tactile_p19_gan_2026-10-25/dataset.npz")
    )
    parser.add_argument("--output", type=Path, default=Path("outputs/tactile_p22_2026-10-25"))
    parser.add_argument("--phase", choices=("tune", "final"), default="tune")
    return parser


if __name__ == "__main__":
    raise SystemExit(run(build_parser().parse_args()))
