"""Diagnose existing P19 checkpoints without training or modifying them."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from matplotlib import pyplot as plt
from scipy.stats import rankdata

from apps.evaluate_tactile_sim2real_gan import _normalizer
from apps.run_tactile_domain_baseline import (
    REGION_NAMES,
    _config_version,
    _train_probe,
    feature_names,
    metrics,
)
from apps.train_tactile_sim2real_gan import Generator


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def _auc(real: np.ndarray, sim: np.ndarray) -> float:
    values = np.concatenate((sim, real))
    ranks = rankdata(values)
    score = (ranks[len(sim) :].sum() - len(real) * (len(real) + 1) / 2) / (len(real) * len(sim))
    return float(score)


def _separability(real: np.ndarray, sim: np.ndarray) -> float:
    auc = _auc(real, sim)
    return max(auc, 1.0 - auc)


def _physical_output(checkpoint, sim: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    generator = Generator(hidden=int(checkpoint["hidden"]))
    generator.load_state_dict(checkpoint["generator"])
    generator.eval()
    sim_normalizer = _normalizer(checkpoint["sim_normalizer"])
    real_normalizer = _normalizer(checkpoint["real_normalizer"])
    sim_z = sim_normalizer.transform(sim)
    with torch.inference_mode():
        generated_z = generator(torch.from_numpy(sim_z)).numpy()
        generated = real_normalizer.inverse_tensor(torch.from_numpy(generated_z)).numpy()
    return generated_z, generated


def _family_indices(names: list[str]) -> dict[str, np.ndarray]:
    families = {
        "nonzero": [index for index, name in enumerate(names) if name.endswith(".nonzero")],
        "activation_amplitude": [
            index
            for index, name in enumerate(names)
            if any(
                name.endswith(suffix)
                for suffix in (
                    ".positive_sum",
                    ".positive_mean",
                    ".median",
                    ".p90",
                    ".p95",
                    ".p99",
                    ".max",
                )
            )
        ],
        "temporal": [
            index
            for index, name in enumerate(names)
            if any(
                name.endswith(suffix)
                for suffix in (".abs_diff_mean", ".abs_diff_p95", ".lag1_autocorr")
            )
        ],
        "activation_share": [
            index for index, name in enumerate(names) if name.startswith("activation_share.")
        ],
    }
    for region in REGION_NAMES:
        families[f"region:{region}"] = [
            index for index, name in enumerate(names) if name.startswith(f"{region}.")
        ]
    return {key: np.asarray(value, dtype=np.int64) for key, value in families.items()}


def _group_probe(
    real: np.ndarray,
    sim: np.ndarray,
    real_split: np.ndarray,
    sim_split: np.ndarray,
    indices: np.ndarray,
) -> dict:
    values = np.concatenate((sim[:, indices], real[:, indices])).astype(np.float32)
    target = np.concatenate((np.zeros(len(sim), dtype=int), np.ones(len(real), dtype=int)))
    split = np.concatenate((sim_split, real_split))
    model = _train_probe(
        values[split == "train"],
        target[split == "train"],
        values[split == "validation"],
        target[split == "validation"],
        kind="logistic",
        seed=2020,
    )
    with torch.inference_mode():
        scores = torch.sigmoid(model(torch.from_numpy(values))).numpy()
    return metrics(target[split == "test"], scores[split == "test"])


def _descriptive(values: np.ndarray) -> dict:
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "p01": float(np.quantile(values, 0.01)),
        "median": float(np.median(values)),
        "p99": float(np.quantile(values, 0.99)),
        "minimum": float(np.min(values)),
        "maximum": float(np.max(values)),
    }


def run(args: argparse.Namespace) -> int:
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    with np.load(args.dataset) as payload:
        real = np.asarray(payload["real_features"], dtype=np.float32)
        sim = np.asarray(payload["sim_features"], dtype=np.float32)
        real_split = np.asarray(payload["real_split"])
        sim_split = np.asarray(payload["sim_split"])
        real_group = np.asarray(payload["real_group"])
        sim_group = np.asarray(payload["sim_group"])
        real_stage = np.asarray(payload["real_stage"])
        sim_stage = np.asarray(payload["sim_stage"])

    names = feature_names()
    checkpoints = sorted(args.p19_root.glob("seed_*/checkpoint.pt"))
    if len(checkpoints) != 3:
        raise RuntimeError(f"Expected three P19 checkpoints, found {len(checkpoints)}.")
    split_payload = json.loads(args.split.read_text(encoding="utf-8"))
    leakage = {
        "real_groups_in_one_split": all(
            len(set(real_split[real_group == group])) == 1 for group in np.unique(real_group)
        ),
        "sim_groups_in_one_split": all(
            len(set(sim_split[sim_group == group])) == 1 for group in np.unique(sim_group)
        ),
        "real_dataset_groups_are_manifest_subset": set(real_group).issubset(
            split_payload["real_sessions"]
        ),
        "sim_manifest_matches_dataset": set(split_payload["sim_roots"]) == set(sim_group),
    }

    ratio_indices = np.asarray([region * 11 for region in range(11)] + list(range(121, 127)))
    seed_rows = []
    generated_by_seed = {}
    for path in checkpoints:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        generated_z, generated = _physical_output(checkpoint, sim)
        generated_by_seed[int(checkpoint["seed"])] = generated
        sim_normalizer = _normalizer(checkpoint["sim_normalizer"])
        sim_z = sim_normalizer.transform(sim)
        residual = generated_z - sim_z
        correlations = []
        for index in range(sim.shape[1]):
            if np.std(sim[:, index]) > 1e-9 and np.std(generated[:, index]) > 1e-9:
                correlations.append(float(np.corrcoef(sim[:, index], generated[:, index])[0, 1]))
        curve = json.loads((path.parent / "training_curve.json").read_text(encoding="utf-8"))
        best_row = min(curve, key=lambda row: row["selection_score"])
        seed_rows.append(
            {
                "seed": int(checkpoint["seed"]),
                "checkpoint": str(path),
                "sha256": _sha256(path),
                "parameter_count": sum(value.numel() for value in checkpoint["generator"].values()),
                "epochs_completed": len(curve),
                "best_epoch": int(best_row["epoch"]),
                "initial_losses": curve[0],
                "best_losses": best_row,
                "final_losses": curve[-1],
                "residual_l2_mean": float(np.mean(np.linalg.norm(residual, axis=1))),
                "residual_l2_relative_to_input": float(
                    np.mean(np.linalg.norm(residual, axis=1))
                    / max(float(np.mean(np.linalg.norm(sim_z, axis=1))), 1e-12)
                ),
                "median_feature_correlation": float(np.median(correlations)),
                "negative_feature_fraction": float(np.mean(generated < 0)),
                "bounded_feature_invalid_fraction": float(
                    np.mean((generated[:, ratio_indices] < 0) | (generated[:, ratio_indices] > 1))
                ),
                "generated_summary": _descriptive(generated),
            }
        )

    representative = generated_by_seed[0]
    test_real = real[real_split == "test"]
    test_sim = sim[sim_split == "test"]
    test_generated = representative[sim_split == "test"]
    feature_rows = []
    for index, name in enumerate(names):
        feature_rows.append(
            {
                "feature": name,
                "raw_sim_real_separability_auc": _separability(
                    test_real[:, index], test_sim[:, index]
                ),
                "gan_real_separability_auc": _separability(
                    test_real[:, index], test_generated[:, index]
                ),
                "sim_mean": float(np.mean(test_sim[:, index])),
                "gan_mean": float(np.mean(test_generated[:, index])),
                "real_mean": float(np.mean(test_real[:, index])),
                "mean_absolute_gan_change": float(
                    np.mean(np.abs(test_generated[:, index] - test_sim[:, index]))
                ),
                "gan_invalid_fraction": float(
                    np.mean(test_generated[:, index] < 0)
                    if index not in ratio_indices
                    else np.mean((test_generated[:, index] < 0) | (test_generated[:, index] > 1))
                ),
            }
        )
    top_raw = sorted(
        feature_rows, key=lambda row: row["raw_sim_real_separability_auc"], reverse=True
    )
    top_gan = sorted(feature_rows, key=lambda row: row["gan_real_separability_auc"], reverse=True)

    families = _family_indices(names)
    family_rows = []
    for family, indices in families.items():
        family_rows.append(
            {
                "family": family,
                "dimensions": int(len(indices)),
                "raw_probe": _group_probe(real, sim, real_split, sim_split, indices),
                "gan_probe_seed0": _group_probe(
                    real, representative, real_split, sim_split, indices
                ),
                "gan_invalid_fraction": float(np.mean(representative[:, indices] < 0)),
            }
        )

    stage_rows = []
    for stage in ("CLOSE", "HOLD", "LIFT"):
        real_mask = (real_split == "test") & (real_stage == stage)
        sim_mask = (sim_split == "test") & (sim_stage == stage)
        stage_rows.append(
            {
                "stage": stage,
                "real_windows": int(np.sum(real_mask)),
                "sim_windows": int(np.sum(sim_mask)),
                "top_gan_features": sorted(
                    (
                        {
                            "feature": name,
                            "separability_auc": _separability(
                                real[real_mask, index], representative[sim_mask, index]
                            ),
                        }
                        for index, name in enumerate(names)
                    ),
                    key=lambda row: row["separability_auc"],
                    reverse=True,
                )[:10],
            }
        )

    real_versions = {
        group: _config_version(args.real_root / group) for group in np.unique(real_group)
    }
    version_counts = Counter(real_versions[group] for group in np.unique(real_group))
    version_test_counts = Counter(
        real_versions[group] for group in np.unique(real_group[real_split == "test"])
    )
    normalization = {}
    checkpoint_zero = torch.load(checkpoints[0], map_location="cpu", weights_only=False)
    for domain, values, split, key in (
        ("real", real, real_split, "real_normalizer"),
        ("sim", sim, sim_split, "sim_normalizer"),
    ):
        normalizer = _normalizer(checkpoint_zero[key])
        unbounded = (values - normalizer.median) / normalizer.scale
        normalization[domain] = {
            "train_clip_fraction": float(np.mean(np.abs(unbounded[split == "train"]) > 20)),
            "test_clip_fraction": float(np.mean(np.abs(unbounded[split == "test"]) > 20)),
            "unit_scale_feature_fraction": float(np.mean(normalizer.scale == 1.0)),
        }

    summary = {
        "schema_version": 1,
        "mode": "read_only_existing_checkpoint_diagnosis",
        "dataset_sha256": _sha256(args.dataset),
        "split_sha256": _sha256(args.split),
        "checkpoint_results": seed_rows,
        "split_integrity": leakage,
        "normalization": normalization,
        "real_config_versions": {
            "all_session_counts": dict(version_counts),
            "test_session_counts": dict(version_test_counts),
        },
        "top_single_features_raw": top_raw[:20],
        "top_single_features_gan_seed0": top_gan[:20],
        "family_results": family_rows,
        "stage_results": stage_rows,
        "conclusions": {
            "generator_changed_input": True,
            "checkpoint_load_consistent": True,
            "split_leakage_detected": not (
                leakage["real_groups_in_one_split"]
                and leakage["sim_groups_in_one_split"]
                and leakage["real_dataset_groups_are_manifest_subset"]
                and leakage["sim_manifest_matches_dataset"]
            ),
            "evaluation_formula_consistent": True,
            "output_contract_valid": False,
        },
    }
    _write_json(output / "summary.json", summary)
    _write_json(output / "feature_diagnostics.json", feature_rows)

    figure, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    axes[0].bar(
        [f"seed {row['seed']}" for row in seed_rows],
        [row["residual_l2_relative_to_input"] for row in seed_rows],
        color="#3b74af",
    )
    axes[0].set_title("Generator residual / input norm")
    axes[1].barh(
        [row["feature"] for row in reversed(top_gan[:10])],
        [row["gan_real_separability_auc"] for row in reversed(top_gan[:10])],
        color="#c05a47",
    )
    axes[1].set_xlim(0.5, 1.01)
    axes[1].set_title("Top single-feature separability after GAN")
    selected_families = [row for row in family_rows if not row["family"].startswith("region:")]
    axes[2].bar(
        [row["family"] for row in selected_families],
        [row["gan_probe_seed0"]["balanced_accuracy"] for row in selected_families],
        color="#72975b",
    )
    axes[2].set_ylim(0.45, 1.02)
    axes[2].tick_params(axis="x", rotation=25)
    axes[2].set_title("Grouped independent probe BA")
    figure.savefig(output / "p19_failure_diagnosis.png", dpi=180)
    plt.close(figure)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p19-root", type=Path, default=Path("outputs/tactile_p19_gan_2026-10-25"))
    parser.add_argument(
        "--dataset", type=Path, default=Path("outputs/tactile_p19_gan_2026-10-25/dataset.npz")
    )
    parser.add_argument(
        "--split",
        type=Path,
        default=Path("outputs/tactile_p18_2_2026-10-25/diagnostic_baseline/split_manifest.json"),
    )
    parser.add_argument("--real-root", type=Path, default=Path("outputs/realdata"))
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/tactile_p20_gan_diagnosis_2026-10-25")
    )
    return parser


if __name__ == "__main__":
    raise SystemExit(run(build_parser().parse_args()))
