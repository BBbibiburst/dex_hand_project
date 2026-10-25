"""Fit and evaluate constrained stage-conditional 127D tactile mappings."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from apps.run_tactile_domain_baseline import (
    Preprocessor,
    _group_bootstrap,
    _train_probe,
    feature_names,
    metrics,
)


STAGES = ("CLOSE", "HOLD", "LIFT")


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


def project_contract(values: np.ndarray) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64).copy()
    for region in range(11):
        start = region * 11
        result[:, start] = np.clip(result[:, start], 0.0, 1.0)
        result[:, start + 1 : start + 3] = np.maximum(result[:, start + 1 : start + 3], 0.0)
        result[:, start + 3 : start + 8] = np.maximum(
            np.maximum.accumulate(result[:, start + 3 : start + 8], axis=1), 0.0
        )
        result[:, start + 8 : start + 10] = np.maximum(result[:, start + 8 : start + 10], 0.0)
        result[:, start + 10] = np.clip(result[:, start + 10], -1.0, 1.0)
    shares = np.maximum(result[:, 121:127], 0.0)
    denominator = shares.sum(axis=1, keepdims=True)
    result[:, 121:127] = np.divide(
        shares,
        denominator,
        out=np.full_like(shares, 1.0 / 6.0),
        where=denominator > 1e-12,
    )
    return result.astype(np.float32)


def contract_violations(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values)
    nonnegative = []
    bounded = []
    autocorrelation = []
    quantile_order = []
    for region in range(11):
        start = region * 11
        bounded.append(values[:, start])
        nonnegative.append(values[:, start + 1 : start + 10])
        autocorrelation.append(values[:, start + 10])
        quantile_order.append(np.diff(values[:, start + 3 : start + 8], axis=1))
    shares = values[:, 121:127]
    return {
        "nonnegative_violation_fraction": float(np.mean(np.concatenate(nonnegative, axis=1) < 0)),
        "bounded_violation_fraction": float(
            np.mean((np.concatenate(bounded, axis=0) < 0) | (np.concatenate(bounded, axis=0) > 1))
        ),
        "autocorrelation_violation_fraction": float(
            np.mean((np.concatenate(autocorrelation) < -1) | (np.concatenate(autocorrelation) > 1))
        ),
        "quantile_order_violation_fraction": float(
            np.mean(np.concatenate(quantile_order, axis=1) < -1e-7)
        ),
        "share_violation_fraction": float(
            np.mean((shares < 0) | (shares > 1)) + np.mean(np.abs(shares.sum(axis=1) - 1.0) > 1e-6)
        ),
        "finite": bool(np.isfinite(values).all()),
    }


def _sqrt_matrix(matrix: np.ndarray, inverse: bool) -> np.ndarray:
    values, vectors = np.linalg.eigh(matrix)
    values = np.maximum(values, 1e-5)
    power = -0.5 if inverse else 0.5
    return (vectors * (values**power)) @ vectors.T


@dataclass
class StageMap:
    sim_quantiles: np.ndarray
    real_quantiles: np.ndarray
    marginal_train: np.ndarray
    marginal_mean: np.ndarray
    real_mean: np.ndarray
    coral: np.ndarray


class ConditionalTransport:
    def __init__(self, covariance_strength: float = 0.0, shrinkage: float = 0.2):
        self.covariance_strength = covariance_strength
        self.shrinkage = shrinkage
        self.probability = np.linspace(0.0, 1.0, 101)
        self.maps: dict[str, StageMap] = {}

    def _marginal(self, values: np.ndarray, stage_map: StageMap) -> np.ndarray:
        result = np.empty_like(values, dtype=np.float64)
        for feature in range(values.shape[1]):
            probability = np.interp(
                values[:, feature],
                stage_map.sim_quantiles[:, feature],
                self.probability,
            )
            result[:, feature] = np.interp(
                probability,
                self.probability,
                stage_map.real_quantiles[:, feature],
            )
        return result

    def fit(self, sim: np.ndarray, real: np.ndarray, sim_stage: np.ndarray, real_stage: np.ndarray):
        for stage in STAGES:
            sim_values = sim[sim_stage == stage]
            real_values = real[real_stage == stage]
            sim_quantiles = np.quantile(sim_values, self.probability, axis=0)
            real_quantiles = np.quantile(real_values, self.probability, axis=0)
            initial = StageMap(
                sim_quantiles=sim_quantiles,
                real_quantiles=real_quantiles,
                marginal_train=np.empty((0, sim.shape[1])),
                marginal_mean=np.zeros(sim.shape[1]),
                real_mean=np.zeros(sim.shape[1]),
                coral=np.eye(sim.shape[1]),
            )
            marginal = self._marginal(sim_values, initial)
            marginal_mean = marginal.mean(axis=0)
            real_mean = real_values.mean(axis=0)
            sim_centered = marginal - marginal_mean
            real_centered = real_values - real_mean
            sim_cov = np.cov(sim_centered, rowvar=False)
            real_cov = np.cov(real_centered, rowvar=False)
            identity = np.eye(sim.shape[1])
            sim_cov = (1 - self.shrinkage) * sim_cov + self.shrinkage * np.trace(
                sim_cov
            ) / sim.shape[1] * identity
            real_cov = (1 - self.shrinkage) * real_cov + self.shrinkage * np.trace(
                real_cov
            ) / real.shape[1] * identity
            coral = _sqrt_matrix(sim_cov, True) @ _sqrt_matrix(real_cov, False)
            self.maps[stage] = StageMap(
                sim_quantiles, real_quantiles, marginal, marginal_mean, real_mean, coral
            )
        return self

    def transform(self, sim: np.ndarray, stages: np.ndarray) -> np.ndarray:
        result = np.empty_like(sim, dtype=np.float32)
        for stage in STAGES:
            selected = stages == stage
            stage_map = self.maps[stage]
            marginal = self._marginal(sim[selected], stage_map)
            aligned = (marginal - stage_map.marginal_mean) @ stage_map.coral + stage_map.real_mean
            mixed = (1 - self.covariance_strength) * marginal + self.covariance_strength * aligned
            result[selected] = project_contract(mixed)
        return result

    def state(self) -> dict:
        return {
            "kind": "stage_conditional_quantile_coral",
            "covariance_strength": self.covariance_strength,
            "shrinkage": self.shrinkage,
        }


def _knn_scores(
    train: np.ndarray, labels: np.ndarray, query: np.ndarray, neighbors: int = 9
) -> np.ndarray:
    distances = np.sum((query[:, None, :] - train[None, :, :]) ** 2, axis=2)
    indices = np.argpartition(distances, min(neighbors, len(train)) - 1, axis=1)[
        :, : min(neighbors, len(train))
    ]
    return labels[indices].mean(axis=1)


def _domain_evaluation(
    real: np.ndarray,
    mapped: np.ndarray,
    real_split: np.ndarray,
    sim_split: np.ndarray,
    subset: str,
    seeds: tuple[int, ...],
    real_groups: np.ndarray | None = None,
    sim_groups: np.ndarray | None = None,
) -> list[dict]:
    joined = np.concatenate((mapped, real))
    domains = np.concatenate((np.zeros(len(mapped), dtype=int), np.ones(len(real), dtype=int)))
    split = np.concatenate((sim_split, real_split))
    processor = Preprocessor("common_robust")
    processor.fit(joined[split == "train"], domains[split == "train"])
    values = processor.transform(joined, domains)
    selected = split == subset
    groups = None
    if real_groups is not None and sim_groups is not None:
        groups = np.concatenate((sim_groups, real_groups))
    results = []
    for kind in ("logistic", "mlp"):
        for seed in seeds:
            model = _train_probe(
                values[split == "train"],
                domains[split == "train"],
                values[split == "validation"],
                domains[split == "validation"],
                kind=kind,
                seed=seed,
            )
            with torch.inference_mode():
                scores = torch.sigmoid(model(torch.from_numpy(values))).numpy()
            result = metrics(domains[selected], scores[selected])
            if groups is not None:
                result["group_bootstrap_95ci"] = _group_bootstrap(
                    domains[selected], scores[selected], groups[selected], seed=seed + 9000
                )
            result.update({"probe": kind, "seed": seed})
            results.append(result)
    scores = _knn_scores(values[split == "train"], domains[split == "train"], values[selected])
    result = metrics(domains[selected], scores)
    if groups is not None:
        result["group_bootstrap_95ci"] = _group_bootstrap(
            domains[selected], scores, groups[selected], seed=19042
        )
    result.update({"probe": "knn", "seed": None})
    results.append(result)
    return results


def _task_evaluation(
    raw: np.ndarray,
    mapped: np.ndarray,
    labels: np.ndarray,
    split: np.ndarray,
    subset: str,
) -> list[dict]:
    results = []
    for source_name, values in (("raw", raw), ("mapped", mapped)):
        median = np.median(values[split == "train"], axis=0)
        q25, q75 = np.quantile(values[split == "train"], (0.25, 0.75), axis=0)
        scale = np.where(q75 - q25 > 1e-6, q75 - q25, 1.0)
        normalized = np.clip((values - median) / scale, -20, 20).astype(np.float32)
        for label_index, label_name in enumerate(("contact", "multi_finger", "thumb_opposition")):
            model = _train_probe(
                normalized[split == "train"],
                labels[split == "train", label_index],
                normalized[split == "validation"],
                labels[split == "validation", label_index],
                kind="logistic",
                seed=2201,
            )
            with torch.inference_mode():
                scores = torch.sigmoid(model(torch.from_numpy(normalized))).numpy()
            result = metrics(labels[split == subset, label_index], scores[split == subset])
            result.update({"source": source_name, "label": label_name})
            results.append(result)
    return results


def _candidate_score(domain_results: list[dict], task_results: list[dict]) -> tuple[float, dict]:
    domain_distance = max(abs(item["balanced_accuracy"] - 0.5) for item in domain_results)
    task = {item["label"]: item for item in task_results if item["source"] == "mapped"}
    task_floor = min(item["balanced_accuracy"] for item in task.values())
    score = domain_distance + 10.0 * max(0.0, 0.95 - task_floor)
    return score, {"max_domain_distance": domain_distance, "mapped_task_floor": task_floor}


def run(args: argparse.Namespace) -> int:
    with np.load(args.dataset) as payload:
        real = np.asarray(payload["real_features"], dtype=np.float32)
        sim = np.asarray(payload["sim_features"], dtype=np.float32)
        labels = np.asarray(payload["sim_labels"], dtype=np.int64)
        real_split = np.asarray(payload["real_split"])
        sim_split = np.asarray(payload["sim_split"])
        real_stage = np.asarray(payload["real_stage"])
        sim_stage = np.asarray(payload["sim_stage"])
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    candidates = []
    for covariance_strength in (0.0, 0.25, 0.5, 0.75, 1.0):
        for shrinkage in (0.1, 0.5, 0.9):
            mapper = ConditionalTransport(covariance_strength, shrinkage).fit(
                sim[sim_split == "train"],
                real[real_split == "train"],
                sim_stage[sim_split == "train"],
                real_stage[real_split == "train"],
            )
            mapped = mapper.transform(sim, sim_stage)
            domain = _domain_evaluation(real, mapped, real_split, sim_split, "validation", (2201,))
            task = _task_evaluation(sim, mapped, labels, sim_split, "validation")
            score, selection = _candidate_score(domain, task)
            candidates.append(
                {
                    "mapper": mapper.state(),
                    "selection_score": score,
                    "selection": selection,
                    "contract": contract_violations(mapped),
                    "domain_validation": domain,
                    "task_validation": task,
                }
            )
    candidates.sort(key=lambda item: item["selection_score"])
    _write_json(output / "route_a_candidates.json", candidates)
    best = candidates[0]
    locked = {
        "schema_version": 1,
        "dataset_sha256": _sha256(args.dataset),
        "feature_dimension": len(feature_names()),
        "selection_uses_test": False,
        "locked_candidate": best,
        "validation_target_met": bool(
            all(item["domain_confusion"] >= 0.90 for item in best["domain_validation"])
            and best["selection"]["mapped_task_floor"] >= 0.95
            and best["contract"]["finite"]
            and all(
                value == 0.0
                for key, value in best["contract"].items()
                if key.endswith("violation_fraction")
            )
        ),
    }
    _write_json(output / "locked_candidate.json", locked)
    if args.phase == "tune":
        return 0
    if not locked["validation_target_met"]:
        raise RuntimeError("No Route A candidate passed validation; do not evaluate test.")
    state = best["mapper"]
    mapper = ConditionalTransport(state["covariance_strength"], state["shrinkage"]).fit(
        sim[sim_split == "train"],
        real[real_split == "train"],
        sim_stage[sim_split == "train"],
        real_stage[real_split == "train"],
    )
    mapped = mapper.transform(sim, sim_stage)
    final = {
        "mapper": state,
        "domain_test": _domain_evaluation(
            real, mapped, real_split, sim_split, "test", (11, 12, 13)
        ),
        "task_test": _task_evaluation(sim, mapped, labels, sim_split, "test"),
        "contract": contract_violations(mapped),
    }
    _write_json(output / "final_test.json", final)
    np.savez_compressed(
        output / "mapped_features.npz",
        mapped_sim=mapped,
        sim_split=sim_split,
        sim_stage=sim_stage,
    )
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
