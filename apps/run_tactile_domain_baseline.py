"""Run the P18.2-B orientation-invariant tactile domain diagnostic."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import yaml
from matplotlib import pyplot as plt
from scipy.special import ndtri


STAGES = ("BASELINE", "CLOSE", "HOLD", "LIFT")
SIM_STAGE_CODES = {"BASELINE": 0, "CLOSE": 4, "HOLD": 5, "LIFT": 6}
STAT_NAMES = ("nonzero", "positive_sum", "positive_mean", "median", "p90", "p95", "p99", "max")
REGION_NAMES = tuple(
    name
    for finger in ("index", "middle", "ring", "little", "thumb")
    for name in (f"{finger}_tip", f"{finger}_segments")
) + ("palm",)
REAL_FINGER_PATCHES = ((3, 4, 5), (6, 7, 8), (9, 10, 11), (12, 13, 14), (0, 1, 2))
SIM_FINGER_SLICES = (
    ((64, 120), (0, 64)),
    ((184, 240), (120, 184)),
    ((304, 360), (240, 304)),
    ((424, 480), (360, 424)),
    ((544, 600), (480, 544)),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def _uniform_indices(length: int, maximum: int) -> np.ndarray:
    if length <= maximum:
        return np.arange(length, dtype=np.int64)
    return np.unique(np.linspace(0, length - 1, maximum).round().astype(np.int64))


def _regional_features(frames: np.ndarray, domain: str) -> np.ndarray:
    """Return direction-invariant summary features for one stage sequence."""
    frames = np.asarray(frames, dtype=np.float64)
    if frames.ndim != 2 or frames.shape[1] != 712:
        raise ValueError(f"Expected [time,712] tactile data, got {frames.shape}.")
    regions = []
    if domain == "real":
        offsets = np.cumsum([0, 56, 32, 32, 56, 32, 32, 56, 32, 32, 56, 32, 32, 56, 32, 32, 112])
        patches = [frames[:, offsets[index] : offsets[index + 1]] for index in range(16)]
        for tip, first, second in REAL_FINGER_PATCHES:
            regions.extend((patches[tip], np.concatenate((patches[first], patches[second]), axis=1)))
        regions.append(patches[15])
    elif domain == "sim":
        for tip_slice, segment_slice in SIM_FINGER_SLICES:
            regions.extend((frames[:, slice(*tip_slice)], frames[:, slice(*segment_slice)]))
        regions.append(frames[:, 600:712])
    else:
        raise ValueError(f"Unknown domain {domain!r}.")

    result: list[float] = []
    region_sums: list[float] = []
    for region in regions:
        positive = np.clip(region, 0.0, None)
        flat = positive.reshape(-1)
        positives = flat[flat > 0.0]
        result.extend(
            (
                float(np.mean(flat > 0.0)),
                float(np.mean(np.sum(positive, axis=1))),
                float(np.mean(positives)) if len(positives) else 0.0,
                float(np.median(flat)),
                float(np.quantile(flat, 0.90)),
                float(np.quantile(flat, 0.95)),
                float(np.quantile(flat, 0.99)),
                float(np.max(flat)),
            )
        )
        stage_sum = np.sum(positive, axis=1)
        region_sums.append(float(np.mean(stage_sum)))
        if len(stage_sum) > 1:
            differences = np.abs(np.diff(stage_sum))
            result.extend((float(np.mean(differences)), float(np.quantile(differences, 0.95))))
        else:
            result.extend((0.0, 0.0))
        if len(stage_sum) > 2 and np.std(stage_sum[:-1]) > 1e-12 and np.std(stage_sum[1:]) > 1e-12:
            result.append(float(np.corrcoef(stage_sum[:-1], stage_sum[1:])[0, 1]))
        else:
            result.append(0.0)
    finger_sums = np.asarray(
        [region_sums[2 * index] + region_sums[2 * index + 1] for index in range(5)]
        + [region_sums[-1]],
        dtype=np.float64,
    )
    denominator = max(float(finger_sums.sum()), 1e-12)
    result.extend((finger_sums / denominator).tolist())
    return np.asarray(result, dtype=np.float32)


def feature_names() -> list[str]:
    names = []
    for region in REGION_NAMES:
        names.extend(f"{region}.{name}" for name in STAT_NAMES)
        names.extend((f"{region}.abs_diff_mean", f"{region}.abs_diff_p95", f"{region}.lag1_autocorr"))
    names.extend(f"activation_share.{name}" for name in ("index", "middle", "ring", "little", "thumb", "palm"))
    return names


def _read_real_stage(trial: Path, stage: str, signal: str, max_frames: int) -> np.ndarray | None:
    samples = []
    with (trial / "samples.jsonl").open(encoding="utf-8", errors="ignore") as stream:
        for line in stream:
            item = json.loads(line)
            if str(item.get("stage", "")).upper() == stage and item.get("sync_id") is not None:
                samples.append(int(item["sync_id"]))
    if not samples:
        return None
    selected = [samples[index] for index in _uniform_indices(len(samples), max_frames)]
    path = trial / ("tactile_raw_split.npz" if signal == "raw" else "tactile_processed.npz")
    prefix = "raw" if signal == "raw" else "proc"
    with np.load(path) as payload:
        rows = []
        for sync_id in selected:
            keys = [f"{prefix}_{sync_id:06d}_{patch:02d}" for patch in range(16)]
            if all(key in payload for key in keys):
                rows.append(np.concatenate([np.asarray(payload[key]).reshape(-1) for key in keys]))
    return np.asarray(rows, dtype=np.float32) if rows else None


def _config_version(session: Path) -> str:
    path = session / "session_config.yaml"
    if not path.is_file():
        return "missing"
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    tactile = payload.get("tactile", {})
    contract = {
        "sample_hz": payload.get("sync_group", {}).get("sample_hz"),
        "sensor_class": tactile.get("sensor_class"),
        "sensor_shape": tactile.get("sensor_shape"),
        "mapping_file": tactile.get("mapping_file"),
        "calibrate": tactile.get("calibrate"),
        "auto_zero_calibration": tactile.get("auto_zero_calibration"),
        "zero_on_baseline": tactile.get("zero_on_baseline"),
        "provider_gated_filter": tactile.get("provider_gated_filter"),
        "processed_gated_filter": tactile.get("processed_gated_filter"),
        "gated": tactile.get("gated"),
        "consistency": tactile.get("consistency"),
    }
    encoded = json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()[:12]


def _split_names(names: list[str], train_fraction=0.65, validation_fraction=0.15) -> dict[str, str]:
    ordered = sorted(names, key=lambda value: hashlib.sha256(value.encode()).hexdigest())
    train_end = max(1, int(round(len(ordered) * train_fraction)))
    validation_end = max(train_end + 1, int(round(len(ordered) * (train_fraction + validation_fraction))))
    validation_end = min(validation_end, len(ordered) - 1)
    result = {}
    for index, name in enumerate(ordered):
        result[name] = "train" if index < train_end else "validation" if index < validation_end else "test"
    return result


def build_dataset(real_root: Path, sim_root: Path, *, max_real_sessions: int, trials_per_session: int, max_frames: int):
    complete_by_session = {}
    for session in sorted(path for path in real_root.iterdir() if path.is_dir()):
        trials = [
            trial
            for trial in sorted(session.glob("trial_*"))
            if (trial / "samples.jsonl").is_file()
            and (trial / "tactile_processed.npz").is_file()
            and (trial / "tactile_raw_split.npz").is_file()
        ]
        if trials:
            complete_by_session[session.name] = (session, trials)
    version_groups = defaultdict(list)
    for name, (session, _) in complete_by_session.items():
        version_groups[_config_version(session)].append(name)
    session_names = []
    versions = sorted(version_groups, key=lambda key: (-len(version_groups[key]), key))
    while len(session_names) < min(max_real_sessions, len(complete_by_session)):
        progressed = False
        for version in versions:
            candidates = sorted(version_groups[version], key=lambda value: hashlib.sha256(value.encode()).hexdigest())
            used = sum(complete_by_session[name][0] in [complete_by_session[x][0] for x in session_names] for name in candidates)
            if used < len(candidates):
                candidate = candidates[used]
                if candidate not in session_names:
                    session_names.append(candidate)
                    progressed = True
                    if len(session_names) >= max_real_sessions:
                        break
        if not progressed:
            break
    train_end = max(1, int(round(len(session_names) * 0.65)))
    validation_end = min(len(session_names) - 1, train_end + max(1, int(round(len(session_names) * 0.15))))
    real_splits = {
        name: "train" if index < train_end else "validation" if index < validation_end else "test"
        for index, name in enumerate(session_names)
    }

    sim_episodes = sorted(sim_root.glob("mode_*/ycb_004_sugar_box/p*/candidate_00/episode.npz"))
    sim_roots = sorted({path.parents[1].name for path in sim_episodes})
    sim_splits = _split_names(sim_roots)
    rows = []
    for session_name in session_names:
        session, trials = complete_by_session[session_name]
        version = _config_version(session)
        for trial in trials[:trials_per_session]:
            for stage in STAGES:
                for signal in ("raw", "processed"):
                    frames = _read_real_stage(trial, stage, signal, max_frames)
                    if frames is None or len(frames) < 2:
                        continue
                    rows.append(
                        {"domain": 1, "domain_name": "real", "signal": signal, "stage": stage,
                         "group": session_name, "unit": f"{session_name}/{trial.name}", "split": real_splits[session_name],
                         "config_version": version, "frames": int(len(frames)), "features": _regional_features(frames, "real")}
                    )
    for path in sim_episodes:
        root = path.parents[1].name
        with np.load(path) as payload:
            tactile = np.asarray(payload["tactile"], dtype=np.float32)
            stages = np.asarray(payload["stage"])
        for stage in STAGES:
            frames = tactile[stages == SIM_STAGE_CODES[stage]]
            if len(frames) < 2:
                continue
            frames = frames[_uniform_indices(len(frames), max_frames)]
            rows.append(
                {"domain": 0, "domain_name": "sim", "signal": "processed", "stage": stage,
                 "group": root, "unit": str(path.parent), "split": sim_splits[root], "config_version": "sim_v1",
                 "frames": int(len(frames)), "features": _regional_features(frames, "sim")}
            )
    split_payload = {
        "real_sessions": real_splits,
        "sim_roots": sim_splits,
        "real_config_versions": {name: _config_version(complete_by_session[name][0]) for name in session_names},
        "sim_episode_paths": [str(path) for path in sim_episodes],
    }
    return rows, split_payload


class Preprocessor:
    def __init__(self, kind: str):
        self.kind = kind
        self.parameters: dict[str, dict[str, list[float]]] = {}

    def fit(self, x: np.ndarray, domains: np.ndarray) -> None:
        keys = ("pooled",) if self.kind == "common_robust" else ("0", "1")
        for key in keys:
            values = x if key == "pooled" else x[domains == int(key)]
            if self.kind in {"common_robust", "domain_robust"}:
                median = np.median(values, axis=0)
                q25, q75 = np.quantile(values, (0.25, 0.75), axis=0)
                scale = np.where(q75 - q25 > 1e-8, q75 - q25, 1.0)
                self.parameters[key] = {"median": median.tolist(), "scale": scale.tolist()}
            elif self.kind == "domain_quantile":
                quantiles = np.quantile(values, np.linspace(0.0, 1.0, 101), axis=0).T
                self.parameters[key] = {"quantiles": quantiles.tolist()}
            else:
                raise ValueError(self.kind)

    def transform(self, x: np.ndarray, domains: np.ndarray) -> np.ndarray:
        result = np.empty_like(x, dtype=np.float32)
        for domain in np.unique(domains):
            selected = domains == domain
            key = "pooled" if self.kind == "common_robust" else str(int(domain))
            params = self.parameters[key]
            if self.kind in {"common_robust", "domain_robust"}:
                result[selected] = (x[selected] - np.asarray(params["median"])) / np.asarray(params["scale"])
            else:
                quantiles = np.asarray(params["quantiles"])
                transformed = np.empty_like(x[selected])
                probability = np.linspace(0.0, 1.0, quantiles.shape[1])
                for feature in range(x.shape[1]):
                    transformed[:, feature] = ndtri(np.clip(np.interp(x[selected, feature], quantiles[feature], probability), 1e-3, 1 - 1e-3))
                result[selected] = transformed
        result = np.nan_to_num(result, nan=0.0, posinf=20.0, neginf=-20.0)
        return np.clip(result, -20.0, 20.0).astype(np.float32)


class Probe(torch.nn.Module):
    def __init__(self, dimension: int, kind: str):
        super().__init__()
        self.network = torch.nn.Linear(dimension, 1) if kind == "logistic" else torch.nn.Sequential(
            torch.nn.Linear(dimension, 64), torch.nn.ReLU(), torch.nn.Linear(64, 32), torch.nn.ReLU(), torch.nn.Linear(32, 1)
        )

    def forward(self, value):
        return self.network(value).squeeze(-1)


def _auc(labels: np.ndarray, scores: np.ndarray) -> float:
    positive = scores[labels == 1]
    negative = scores[labels == 0]
    if not len(positive) or not len(negative):
        return float("nan")
    return float(np.mean(positive[:, None] > negative[None, :]) + 0.5 * np.mean(positive[:, None] == negative[None, :]))


def metrics(labels: np.ndarray, scores: np.ndarray) -> dict[str, object]:
    predicted = scores >= 0.5
    recalls = []
    matrix = []
    for actual in (0, 1):
        selected = labels == actual
        recalls.append(float(np.mean(predicted[selected] == actual)) if np.any(selected) else float("nan"))
        matrix.append([int(np.sum(selected & (predicted == 0))), int(np.sum(selected & (predicted == 1)))])
    balanced = float(np.nanmean(recalls))
    return {"balanced_accuracy": balanced, "roc_auc": _auc(labels, scores), "recall_sim": recalls[0],
            "recall_real": recalls[1], "confusion_matrix": matrix,
            "domain_confusion": max(0.0, 1.0 - 2.0 * abs(balanced - 0.5))}


def _train_probe(x_train, y_train, x_validation, y_validation, *, kind: str, seed: int):
    torch.manual_seed(seed)
    model = Probe(x_train.shape[1], kind)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3 if kind == "logistic" else 1e-3, weight_decay=1e-4)
    tx, ty = torch.from_numpy(x_train), torch.from_numpy(y_train.astype(np.float32))
    vx, vy = torch.from_numpy(x_validation), torch.from_numpy(y_validation.astype(np.float32))
    counts = np.bincount(y_train, minlength=2).astype(float)
    weights = torch.from_numpy(np.asarray([len(y_train) / max(2 * counts[int(v)], 1.0) for v in y_train], dtype=np.float32))
    best, best_state, patience = float("inf"), None, 60
    for _ in range(600):
        model.train()
        optimizer.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(model(tx), ty, weight=weights)
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            val = torch.nn.functional.binary_cross_entropy_with_logits(model(vx), vy).item()
        if val < best - 1e-6:
            best, best_state, patience = val, {key: value.detach().clone() for key, value in model.state_dict().items()}, 60
        else:
            patience -= 1
            if patience == 0:
                break
    if best_state is None:
        raise RuntimeError(f"Probe training produced no finite validation loss: kind={kind}, seed={seed}.")
    model.load_state_dict(best_state)
    return model


def _group_bootstrap(labels, scores, groups, repetitions=1000, seed=9042):
    rng = np.random.default_rng(seed)
    by_domain = {domain: sorted(set(groups[labels == domain])) for domain in (0, 1)}
    values = defaultdict(list)
    for _ in range(repetitions):
        indices = []
        for domain in (0, 1):
            selected_groups = rng.choice(by_domain[domain], len(by_domain[domain]), replace=True)
            for group in selected_groups:
                candidates = np.flatnonzero((labels == domain) & (groups == group))
                indices.extend(rng.choice(candidates, len(candidates), replace=True).tolist())
        result = metrics(labels[np.asarray(indices)], scores[np.asarray(indices)])
        for key in ("balanced_accuracy", "roc_auc", "domain_confusion"):
            values[key].append(result[key])
    return {key: [float(np.nanquantile(value, 0.025)), float(np.nanquantile(value, 0.975))] for key, value in values.items()}


def run(args) -> dict:
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    rows, split_payload = build_dataset(args.real_root, args.sim_root, max_real_sessions=args.max_real_sessions,
                                        trials_per_session=args.trials_per_session, max_frames=args.max_frames)
    feature_labels = feature_names()
    if not rows or len(rows[0]["features"]) != len(feature_labels):
        raise RuntimeError("Feature construction failed.")
    _atomic_json(output / "split_manifest.json", split_payload)
    split_hash = _sha256(output / "split_manifest.json")
    inventory = Counter((row["domain_name"], row["signal"], row["stage"], row["split"]) for row in rows)
    _atomic_json(output / "dataset_inventory.json", {
        "units": len(rows), "feature_dimension": len(feature_labels), "feature_names": feature_labels,
        "counts": [{"domain": key[0], "signal": key[1], "stage": key[2], "split": key[3], "units": value} for key, value in sorted(inventory.items())],
        "split_manifest_sha256": split_hash,
    })
    results = []
    task_results = []
    tracks = (("real_processed_vs_sim_processed", "processed"), ("real_raw_vs_sim_processed", "raw"))
    for track_name, real_signal in tracks:
        selected_rows = [row for row in rows if row["domain_name"] == "sim" or row["signal"] == real_signal]
        x = np.stack([row["features"] for row in selected_rows])
        y = np.asarray([row["domain"] for row in selected_rows], dtype=np.int64)
        groups = np.asarray([row["group"] for row in selected_rows])
        splits = np.asarray([row["split"] for row in selected_rows])
        stages = np.asarray([row["stage"] for row in selected_rows])
        for preprocessing in ("common_robust", "domain_robust", "domain_quantile"):
            processor = Preprocessor(preprocessing)
            processor.fit(x[splits == "train"], y[splits == "train"])
            transformed = processor.transform(x, y)
            _atomic_json(output / "preprocessing" / f"{track_name}_{preprocessing}.json", processor.parameters)
            for model_kind in ("logistic", "mlp"):
                for seed in args.seeds:
                    model = _train_probe(transformed[splits == "train"], y[splits == "train"],
                                         transformed[splits == "validation"], y[splits == "validation"], kind=model_kind, seed=seed)
                    with torch.no_grad():
                        scores = torch.sigmoid(model(torch.from_numpy(transformed))).numpy()
                    test = splits == "test"
                    result = metrics(y[test], scores[test])
                    result.update({"track": track_name, "preprocessing": preprocessing, "model": model_kind, "seed": seed,
                                   "test_units": int(test.sum()), "test_groups_sim": len(set(groups[test & (y == 0)])),
                                   "test_groups_real": len(set(groups[test & (y == 1)])),
                                   "bootstrap_95ci": _group_bootstrap(y[test], scores[test], groups[test])})
                    stage_results = {}
                    for stage in STAGES:
                        mask = test & (stages == stage)
                        if np.sum(mask & (y == 0)) and np.sum(mask & (y == 1)):
                            stage_results[stage] = metrics(y[mask], scores[mask])
                    result["by_stage"] = stage_results
                    real_versions = defaultdict(list)
                    for index, row in enumerate(selected_rows):
                        if test[index] and row["domain_name"] == "real":
                            real_versions[row["config_version"]].append(index)
                    result["real_recall_by_config"] = {
                        version: float(np.mean(scores[indices] >= 0.5)) for version, indices in real_versions.items()
                    }
                    if model_kind == "logistic":
                        weights = np.abs(model.network.weight.detach().numpy()[0])
                        result["top_features"] = [
                            {"name": feature_labels[index], "absolute_weight": float(weights[index])}
                            for index in np.argsort(weights)[-15:][::-1]
                        ]
                    results.append(result)
            for domain, domain_name in ((0, "sim"), (1, "real")):
                domain_mask = y == domain
                task_labels = (stages != "BASELINE").astype(np.int64)
                for seed in args.seeds:
                    model = _train_probe(
                        transformed[domain_mask & (splits == "train")],
                        task_labels[domain_mask & (splits == "train")],
                        transformed[domain_mask & (splits == "validation")],
                        task_labels[domain_mask & (splits == "validation")],
                        kind="logistic",
                        seed=seed,
                    )
                    with torch.no_grad():
                        scores = torch.sigmoid(model(torch.from_numpy(transformed))).numpy()
                    test = domain_mask & (splits == "test")
                    task_result = metrics(task_labels[test], scores[test])
                    task_result.update(
                        {
                            "track": track_name,
                            "preprocessing": preprocessing,
                            "domain": domain_name,
                            "model": "logistic",
                            "seed": seed,
                            "negative_label": "BASELINE",
                            "positive_label": "CLOSE_OR_HOLD_OR_LIFT",
                            "test_units": int(test.sum()),
                            "bootstrap_95ci": _group_bootstrap(
                                task_labels[test], scores[test], groups[test]
                            ),
                        }
                    )
                    task_results.append(task_result)
    _atomic_json(output / "results.json", results)
    _atomic_json(output / "task_preservation.json", task_results)
    with (output / "results.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("track", "preprocessing", "model", "seed", "balanced_accuracy", "roc_auc", "recall_sim", "recall_real", "domain_confusion"))
        writer.writeheader()
        for result in results:
            writer.writerow({key: result[key] for key in writer.fieldnames})
    summary = {
        "schema_version": 1, "split_manifest_sha256": split_hash, "tracks": {},
        "limitations": [
            "Simulation episodes contain processed tactile only; real_raw_vs_sim_processed is not a raw-to-raw comparison.",
            "BASELINE maps real BASELINE to simulation settle and is only operationally, not physically, equivalent.",
            "Real contact truth is unavailable; task preservation uses stage labels only.",
            "Only eight independent simulation roots are available, so grouped confidence intervals are expected to be wide.",
            "Domain-specific robust/quantile transforms require known domain identity and are diagnostic, not deployable mappings."
        ],
    }
    for track_name, _ in tracks:
        candidates = [result for result in results if result["track"] == track_name]
        best = min(candidates, key=lambda result: abs(result["balanced_accuracy"] - 0.5))
        summary["tracks"][track_name] = {"closest_to_chance": best}
    _atomic_json(output / "summary.json", summary)

    labels = []
    domain_values = []
    task_real_values = []
    task_sim_values = []
    for preprocessing in ("common_robust", "domain_robust", "domain_quantile"):
        labels.append(preprocessing.replace("_", "\n"))
        selected = [
            result["balanced_accuracy"]
            for result in results
            if result["track"] == "real_processed_vs_sim_processed"
            and result["preprocessing"] == preprocessing
            and result["model"] == "logistic"
        ]
        domain_values.append(float(np.mean(selected)))
        task_real_values.append(
            float(np.mean([result["balanced_accuracy"] for result in task_results
                           if result["track"] == "real_processed_vs_sim_processed"
                           and result["preprocessing"] == preprocessing and result["domain"] == "real"]))
        )
        task_sim_values.append(
            float(np.mean([result["balanced_accuracy"] for result in task_results
                           if result["track"] == "real_processed_vs_sim_processed"
                           and result["preprocessing"] == preprocessing and result["domain"] == "sim"]))
        )
    figure, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    axes[0].bar(labels, domain_values, color="#3b74af")
    axes[0].axhline(0.5, color="black", linestyle="--", linewidth=1)
    axes[0].set_ylim(0.0, 1.05)
    axes[0].set_ylabel("Test balanced accuracy")
    axes[0].set_title("Domain probe (processed track)")
    positions = np.arange(len(labels))
    width = 0.36
    axes[1].bar(positions - width / 2, task_real_values, width, label="Real", color="#d9822b")
    axes[1].bar(positions + width / 2, task_sim_values, width, label="Sim", color="#55a868")
    axes[1].set_xticks(positions, labels)
    axes[1].set_ylim(0.0, 1.05)
    axes[1].set_ylabel("Stage-label balanced accuracy")
    axes[1].set_title("Task-label preservation")
    axes[1].legend()
    figure.savefig(output / "domain_and_task_summary.png", dpi=180)
    plt.close(figure)
    return summary


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real-root", type=Path, default=Path("outputs/realdata"))
    parser.add_argument("--sim-root", type=Path, default=Path("outputs/dp_p15_2026-10-24/candidates"))
    parser.add_argument("--output", type=Path, default=Path("outputs/tactile_p18_2_2026-10-25/diagnostic_baseline"))
    parser.add_argument("--max-real-sessions", type=int, default=24)
    parser.add_argument("--trials-per-session", type=int, default=2)
    parser.add_argument("--max-frames", type=int, default=64)
    parser.add_argument("--seeds", type=int, nargs="+", default=(0, 1, 2))
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
