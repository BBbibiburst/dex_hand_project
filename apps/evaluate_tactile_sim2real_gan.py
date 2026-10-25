"""Evaluate P19 GAN outputs with independent domain and contact probes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from matplotlib import pyplot as plt

from apps.run_tactile_domain_baseline import Preprocessor, _group_bootstrap, _train_probe, metrics
from apps.train_tactile_sim2real_gan import DomainNormalizer, Generator


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def _normalizer(state):
    result = DomainNormalizer()
    result.median = np.asarray(state["median"], dtype=np.float32)
    result.scale = np.asarray(state["scale"], dtype=np.float32)
    return result


def _gan_transform(checkpoint, sim):
    generator = Generator(hidden=int(checkpoint["hidden"]))
    generator.load_state_dict(checkpoint["generator"])
    generator.eval()
    sim_norm = _normalizer(checkpoint["sim_normalizer"])
    with torch.inference_mode():
        return generator(torch.from_numpy(sim_norm.transform(sim))).numpy()


def _method_data(method, real, sim, real_split, sim_split, checkpoint=None):
    joined = np.concatenate((sim, real))
    domains = np.concatenate((np.zeros(len(sim), dtype=int), np.ones(len(real), dtype=int)))
    train = np.concatenate((sim_split == "train", real_split == "train"))
    if method in {"common_robust", "domain_robust", "domain_quantile"}:
        processor = Preprocessor(method)
        processor.fit(joined[train], domains[train])
        transformed = processor.transform(joined, domains)
        return transformed[len(sim) :], transformed[: len(sim)]
    if method == "gan":
        real_norm = _normalizer(checkpoint["real_normalizer"])
        return real_norm.transform(real), _gan_transform(checkpoint, sim)
    raise ValueError(method)


def _domain_probe(real, sim, real_split, sim_split, real_groups, sim_groups, probe_kind, seed):
    x = np.concatenate((sim, real))
    y = np.concatenate((np.zeros(len(sim), dtype=int), np.ones(len(real), dtype=int)))
    splits = np.concatenate((sim_split, real_split))
    groups = np.concatenate((sim_groups, real_groups))
    model = _train_probe(
        x[splits == "train"],
        y[splits == "train"],
        x[splits == "validation"],
        y[splits == "validation"],
        kind=probe_kind,
        seed=seed,
    )
    with torch.inference_mode():
        scores = torch.sigmoid(model(torch.from_numpy(x))).numpy()
    test = splits == "test"
    result = metrics(y[test], scores[test])
    result["bootstrap_95ci"] = _group_bootstrap(y[test], scores[test], groups[test])
    return result


def _task_probe(values, labels, splits, groups, label_index, seed):
    target = labels[:, label_index].astype(int)
    model = _train_probe(
        values[splits == "train"],
        target[splits == "train"],
        values[splits == "validation"],
        target[splits == "validation"],
        kind="logistic",
        seed=seed,
    )
    with torch.inference_mode():
        scores = torch.sigmoid(model(torch.from_numpy(values))).numpy()
    test = splits == "test"
    result = metrics(target[test], scores[test])
    result["bootstrap_95ci"] = _group_bootstrap(target[test], scores[test], groups[test])
    return result


def run(args):
    with np.load(args.dataset) as data:
        real = np.asarray(data["real_features"], dtype=np.float32)
        sim = np.asarray(data["sim_features"], dtype=np.float32)
        labels = np.asarray(data["sim_labels"])
        real_split = np.asarray(data["real_split"])
        sim_split = np.asarray(data["sim_split"])
        real_groups = np.asarray(data["real_group"])
        sim_groups = np.asarray(data["sim_group"])
        real_stage = np.asarray(data["real_stage"])
        sim_stage = np.asarray(data["sim_stage"])
    checkpoints = {
        path.parent.name: torch.load(path, map_location="cpu", weights_only=False)
        for path in sorted(args.root.glob("seed_*/checkpoint.pt"))
    }
    results, task_results, collapse = [], [], []
    methods = ("common_robust", "domain_robust", "domain_quantile")
    for method in methods:
        mapped_real, mapped_sim = _method_data(method, real, sim, real_split, sim_split)
        for probe in ("logistic", "mlp"):
            for seed in args.probe_seeds:
                item = _domain_probe(
                    mapped_real,
                    mapped_sim,
                    real_split,
                    sim_split,
                    real_groups,
                    sim_groups,
                    probe,
                    seed,
                )
                item.update(
                    {"method": method, "generator_seed": None, "probe": probe, "probe_seed": seed}
                )
                item["by_stage"] = {}
                results.append(item)
        for label_index, label in enumerate(("contact", "multi_finger", "thumb_opposition")):
            for seed in args.probe_seeds:
                item = _task_probe(mapped_sim, labels, sim_split, sim_groups, label_index, seed)
                item.update(
                    {"method": method, "generator_seed": None, "label": label, "probe_seed": seed}
                )
                task_results.append(item)
    for generator_name, checkpoint in checkpoints.items():
        mapped_real, mapped_sim = _method_data("gan", real, sim, real_split, sim_split, checkpoint)
        real_normalizer = _normalizer(checkpoint["real_normalizer"])
        generated_physical = real_normalizer.inverse_tensor(torch.from_numpy(mapped_sim)).numpy()
        ratio_indices = np.asarray([region * 11 for region in range(11)] + list(range(121, 127)))
        generator_seed = int(checkpoint["seed"])
        for probe in ("logistic", "mlp"):
            for seed in args.probe_seeds:
                item = _domain_probe(
                    mapped_real,
                    mapped_sim,
                    real_split,
                    sim_split,
                    real_groups,
                    sim_groups,
                    probe,
                    seed,
                )
                item.update(
                    {
                        "method": "gan",
                        "generator_seed": generator_seed,
                        "probe": probe,
                        "probe_seed": seed,
                    }
                )
                stage_results = {}
                for stage in ("CLOSE", "HOLD", "LIFT"):
                    real_mask, sim_mask = real_stage == stage, sim_stage == stage
                    stage_results[stage] = _domain_probe(
                        mapped_real[real_mask],
                        mapped_sim[sim_mask],
                        real_split[real_mask],
                        sim_split[sim_mask],
                        real_groups[real_mask],
                        sim_groups[sim_mask],
                        probe,
                        seed,
                    )
                item["by_stage"] = stage_results
                results.append(item)
        for label_index, label in enumerate(("contact", "multi_finger", "thumb_opposition")):
            for seed in args.probe_seeds:
                item = _task_probe(mapped_sim, labels, sim_split, sim_groups, label_index, seed)
                item.update(
                    {
                        "method": "gan",
                        "generator_seed": generator_seed,
                        "label": label,
                        "probe_seed": seed,
                    }
                )
                task_results.append(item)
        train = sim_split == "train"
        real_train = real_split == "train"
        collapse.append(
            {
                "generator_seed": generator_seed,
                "finite": bool(np.isfinite(mapped_sim).all()),
                "generated_range": [float(np.min(mapped_sim)), float(np.max(mapped_sim))],
                "mean_feature_std": float(np.mean(np.std(mapped_sim, axis=0))),
                "variance_ratio_to_real": float(
                    np.mean(np.var(mapped_sim[train], axis=0))
                    / max(float(np.mean(np.var(mapped_real[real_train], axis=0))), 1e-12)
                ),
                "near_constant_feature_fraction": float(np.mean(np.std(mapped_sim, axis=0) < 1e-4)),
                "physical_feature_range": [
                    float(np.min(generated_physical)),
                    float(np.max(generated_physical)),
                ],
                "negative_physical_feature_fraction": float(np.mean(generated_physical < 0)),
                "bounded_ratio_invalid_fraction": float(
                    np.mean(
                        (generated_physical[:, ratio_indices] < 0)
                        | (generated_physical[:, ratio_indices] > 1)
                    )
                ),
            }
        )
    _write_json(args.root / "evaluation" / "domain_results.json", results)
    _write_json(args.root / "evaluation" / "task_results.json", task_results)
    _write_json(args.root / "evaluation" / "collapse_checks.json", collapse)
    summary = {
        "schema_version": 1,
        "methods": {},
        "target_domain_confusion": 0.90,
        "contact_onset_status": "not_estimable_from_aggregate_window_generator",
        "real_task_truth_status": "unavailable",
    }
    for method in (*methods, "gan"):
        selected = [item for item in results if item["method"] == method]
        summary["methods"][method] = {
            "balanced_accuracy_mean": float(
                np.mean([item["balanced_accuracy"] for item in selected])
            ),
            "balanced_accuracy_range": [
                float(min(item["balanced_accuracy"] for item in selected)),
                float(max(item["balanced_accuracy"] for item in selected)),
            ],
            "roc_auc_mean": float(np.mean([item["roc_auc"] for item in selected])),
            "domain_confusion_mean": float(
                np.mean([item["domain_confusion"] for item in selected])
            ),
        }
    _write_json(args.root / "evaluation" / "summary.json", summary)
    curves = {
        path.parent.name: json.loads((path.parent / "training_curve.json").read_text())
        for path in sorted(args.root.glob("seed_*/checkpoint.pt"))
    }
    figure, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for name, curve in curves.items():
        axes[0].plot(
            [row["epoch"] for row in curve], [row["selection_score"] for row in curve], label=name
        )
    axes[0].set_title("GAN validation selection score")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    names = list(summary["methods"])
    values = [summary["methods"][name]["balanced_accuracy_mean"] for name in names]
    axes[1].bar(names, values, color="#3b74af")
    axes[1].axhline(0.5, color="black", linestyle="--")
    axes[1].set_ylim(0, 1.05)
    axes[1].set_ylabel("Independent probe BA")
    axes[1].tick_params(axis="x", rotation=20)
    figure.savefig(args.root / "evaluation" / "training_and_domain_results.png", dpi=180)
    plt.close(figure)
    return 0


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("outputs/tactile_p19_gan_2026-10-25"))
    parser.add_argument(
        "--dataset", type=Path, default=Path("outputs/tactile_p19_gan_2026-10-25/dataset.npz")
    )
    parser.add_argument("--probe-seeds", type=int, nargs="+", default=(11, 12, 13))
    return parser


if __name__ == "__main__":
    raise SystemExit(run(build_parser().parse_args()))
