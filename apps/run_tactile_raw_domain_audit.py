"""Run the P18.2-C strictly grouped raw/processed tactile domain audit."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

from apps.run_tactile_domain_baseline import (
    Preprocessor,
    _config_version,
    _group_bootstrap,
    _regional_features,
    _train_probe,
    feature_names,
    metrics,
)


STAGES = ("CLOSE", "HOLD", "LIFT")
STAGE_CODES = {"CLOSE": 4, "HOLD": 5, "LIFT": 6}


def _value_summary(frames: np.ndarray) -> dict[str, float]:
    values = np.asarray(frames, dtype=np.float64).reshape(-1)
    positive = values[values > 0.0]
    return {
        "minimum": float(np.min(values)),
        "maximum": float(np.max(values)),
        "zero_fraction": float(np.mean(values == 0.0)),
        "p50": float(np.quantile(values, 0.50)),
        "p90": float(np.quantile(values, 0.90)),
        "p95": float(np.quantile(values, 0.95)),
        "p99": float(np.quantile(values, 0.99)),
        "positive_mean": float(np.mean(positive)) if len(positive) else 0.0,
    }


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def _real_stage_frames(trial: Path, stage: str, signal: str, target_hz: float = 20.0):
    rows = []
    with (trial / "samples.jsonl").open(encoding="utf-8", errors="ignore") as stream:
        for line in stream:
            item = json.loads(line)
            if str(item.get("stage", "")).upper() == stage and item.get("sync_id") is not None:
                timestamp = item.get("tactile_ts", item.get("sample_ts"))
                if timestamp is not None:
                    rows.append((float(timestamp), int(item["sync_id"])))
    if len(rows) < 2:
        return None, None
    selected = []
    next_time = rows[0][0]
    for timestamp, sync_id in rows:
        if timestamp + 1e-9 >= next_time:
            selected.append((timestamp, sync_id))
            next_time = timestamp + 1.0 / target_hz
    filename = "tactile_raw_split.npz" if signal == "raw" else "tactile_processed.npz"
    prefix = "raw" if signal == "raw" else "proc"
    frames, times = [], []
    with np.load(trial / filename) as payload:
        for timestamp, sync_id in selected:
            keys = [f"{prefix}_{sync_id:06d}_{patch:02d}" for patch in range(16)]
            if all(key in payload for key in keys):
                frames.append(np.concatenate([np.asarray(payload[key]).reshape(-1) for key in keys]))
                times.append(timestamp)
    if len(frames) < 2:
        return None, None
    return np.asarray(frames, dtype=np.float32), np.asarray(times, dtype=np.float64)


def build_rows(real_root: Path, sim_root: Path, frozen_split: dict):
    rows = []
    interval_stats = defaultdict(list)
    real_splits = frozen_split["real_sessions"]
    for session_name, split in real_splits.items():
        session = real_root / session_name
        version = _config_version(session)
        trials = [trial for trial in sorted(session.glob("trial_*"))
                  if (trial / "samples.jsonl").is_file() and (trial / "tactile_raw_split.npz").is_file()
                  and (trial / "tactile_processed.npz").is_file()][:2]
        for trial in trials:
            for stage in STAGES:
                for signal in ("raw", "processed"):
                    frames, times = _real_stage_frames(trial, stage, signal)
                    if frames is None:
                        continue
                    if signal == "processed":
                        interval_stats[f"real:{version}:{stage}"].extend(np.diff(times).tolist())
                    rows.append({"domain": 1, "domain_name": "real", "signal": signal, "stage": stage,
                                 "group": session_name, "unit": f"{session_name}/{trial.name}", "split": split,
                                 "config_version": version, "frames": len(frames),
                                 "features": _regional_features(frames, "real"),
                                 "value_summary": _value_summary(frames)})
    sim_splits = frozen_split["sim_roots"]
    for manifest in sorted(sim_root.glob("mode_*/ycb_004_sugar_box/p*/candidate_00/manifest.json")):
        metadata = json.loads(manifest.read_text(encoding="utf-8"))
        root = metadata["root_id"]
        with np.load(manifest.parent / "tactile_dual.npz") as payload:
            stages = np.asarray(payload["stage"])
            times = np.asarray(payload["simulation_time"])
            for stage in STAGES:
                selected = stages == STAGE_CODES[stage]
                if np.sum(selected) < 2:
                    continue
                interval_stats[f"sim:{stage}"].extend(np.diff(times[selected]).tolist())
                for signal, key in (("raw", "raw_tactile"), ("processed", "processed_tactile")):
                    frames = np.asarray(payload[key][selected], dtype=np.float32)
                    rows.append({"domain": 0, "domain_name": "sim", "signal": signal, "stage": stage,
                                 "group": root, "unit": str(manifest.parent), "split": sim_splits[root],
                                 "config_version": "sim_v1", "frames": len(frames),
                                 "features": _regional_features(frames, "sim"),
                                 "value_summary": _value_summary(frames)})
    return rows, interval_stats


def _evaluate_track(rows, real_signal, sim_signal, seeds):
    selected = [row for row in rows if (row["domain_name"] == "real" and row["signal"] == real_signal)
                or (row["domain_name"] == "sim" and row["signal"] == sim_signal)]
    x = np.stack([row["features"] for row in selected])
    y = np.asarray([row["domain"] for row in selected], dtype=np.int64)
    groups = np.asarray([row["group"] for row in selected])
    splits = np.asarray([row["split"] for row in selected])
    stages = np.asarray([row["stage"] for row in selected])
    results = []
    for preprocessing in ("common_robust", "domain_robust", "domain_quantile"):
        processor = Preprocessor(preprocessing)
        processor.fit(x[splits == "train"], y[splits == "train"])
        transformed = processor.transform(x, y)
        for model_kind in ("logistic", "mlp"):
            for seed in seeds:
                model = _train_probe(transformed[splits == "train"], y[splits == "train"],
                                     transformed[splits == "validation"], y[splits == "validation"],
                                     kind=model_kind, seed=seed)
                with torch.no_grad():
                    scores = torch.sigmoid(model(torch.from_numpy(transformed))).numpy()
                test = splits == "test"
                result = metrics(y[test], scores[test])
                result.update({"real_signal": real_signal, "sim_signal": sim_signal,
                               "preprocessing": preprocessing, "model": model_kind, "seed": seed,
                               "test_units": int(test.sum()),
                               "bootstrap_95ci": _group_bootstrap(y[test], scores[test], groups[test])})
                result["by_stage"] = {
                    stage: metrics(y[test & (stages == stage)], scores[test & (stages == stage)])
                    for stage in STAGES
                }
                results.append(result)
    return results


def _task_windows(sim_root: Path, sim_splits: dict, window=8):
    result = []
    for manifest in sorted(sim_root.glob("mode_*/ycb_004_sugar_box/p*/candidate_00/manifest.json")):
        metadata = json.loads(manifest.read_text(encoding="utf-8"))
        root = metadata["root_id"]
        with np.load(manifest.parent / "tactile_dual.npz") as payload:
            stages = np.asarray(payload["stage"])
            eligible = stages >= 3
            indices = np.flatnonzero(eligible)
            for start in range(0, max(0, len(indices) - window + 1), window):
                chosen = indices[start : start + window]
                digit = np.asarray(payload["robot_object_digit_contact_count"])[chosen] > 0
                labels = {
                    "contact": int(np.mean(np.sum(digit, axis=1) > 0) >= 0.5),
                    "multi_finger": int(np.mean(np.sum(digit, axis=1) >= 2) >= 0.5),
                    "thumb_opposition": int(np.mean(digit[:, 4] & np.any(digit[:, :4], axis=1)) >= 0.5),
                }
                for signal, key in (("raw", "raw_tactile"), ("processed", "processed_tactile")):
                    frames = np.asarray(payload[key][chosen], dtype=np.float32)
                    result.append({"root": root, "split": sim_splits[root], "signal": signal,
                                   "features": _regional_features(frames, "sim"), "labels": labels})
    return result


def _contact_onsets(sim_root: Path):
    rows = []
    for manifest in sorted(sim_root.glob("mode_*/ycb_004_sugar_box/p*/candidate_00/manifest.json")):
        metadata = json.loads(manifest.read_text(encoding="utf-8"))
        with np.load(manifest.parent / "tactile_dual.npz") as payload:
            truth = np.sum(np.asarray(payload["robot_object_digit_contact_count"]), axis=1) > 0
            true_indices = np.flatnonzero(truth)
            item = {"root": metadata["root_id"], "mode": metadata["mode"],
                    "true_contact_frame": int(true_indices[0]) if len(true_indices) else None}
            for signal, key in (("raw", "raw_tactile"), ("processed", "processed_tactile")):
                indices = np.flatnonzero(np.any(np.asarray(payload[key]) > 0.0, axis=1))
                onset = int(indices[0]) if len(indices) else None
                item[f"{signal}_positive_frame"] = onset
                item[f"{signal}_onset_error_steps"] = (
                    None if onset is None or item["true_contact_frame"] is None
                    else onset - item["true_contact_frame"]
                )
            rows.append(item)
    summary = {}
    for signal in ("raw", "processed"):
        values = [row[f"{signal}_onset_error_steps"] for row in rows
                  if row[f"{signal}_onset_error_steps"] is not None]
        summary[signal] = ({"episodes": len(values), "median_steps": float(np.median(values)),
                            "p05_steps": float(np.quantile(values, 0.05)),
                            "p95_steps": float(np.quantile(values, 0.95))}
                           if values else {"episodes": 0})
    return {"definition": "first any positive tactile taxel minus first digit-object contact frame",
            "per_episode": rows, "summary": summary}


def _task_preservation(windows, seeds):
    output = []
    for signal in ("raw", "processed"):
        selected = [row for row in windows if row["signal"] == signal]
        x = np.stack([row["features"] for row in selected])
        splits = np.asarray([row["split"] for row in selected])
        groups = np.asarray([row["root"] for row in selected])
        conditioner = Preprocessor("common_robust")
        dummy_domain = np.zeros(len(x), dtype=np.int64)
        conditioner.fit(x[splits == "train"], dummy_domain[splits == "train"])
        x = conditioner.transform(x, dummy_domain)
        for label_name in ("contact", "multi_finger", "thumb_opposition"):
            labels = np.asarray([row["labels"][label_name] for row in selected], dtype=np.int64)
            if any(len(np.unique(labels[splits == split])) < 2 for split in ("train", "validation", "test")):
                output.append({"signal": signal, "label": label_name, "status": "not_estimable",
                               "class_counts": dict(Counter(labels.tolist()))})
                continue
            for seed in seeds:
                model = _train_probe(x[splits == "train"], labels[splits == "train"],
                                     x[splits == "validation"], labels[splits == "validation"],
                                     kind="logistic", seed=seed)
                with torch.no_grad():
                    scores = torch.sigmoid(model(torch.from_numpy(x))).numpy()
                test = splits == "test"
                item = metrics(labels[test], scores[test])
                item.update({"signal": signal, "label": label_name, "status": "estimated", "seed": seed,
                             "bootstrap_95ci": _group_bootstrap(labels[test], scores[test], groups[test])})
                output.append(item)
    return output


def _signal_statistics(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["domain_name"], row["signal"], row["config_version"], row["stage"])].append(row)
    return [
        {"domain": key[0], "signal": key[1], "config_version": key[2], "stage": key[3],
         "units": len(values),
         "feature_median": np.median([value["features"] for value in values], axis=0).tolist(),
         "feature_p05": np.quantile([value["features"] for value in values], 0.05, axis=0).tolist(),
         "feature_p95": np.quantile([value["features"] for value in values], 0.95, axis=0).tolist(),
         "value_range": [min(value["value_summary"]["minimum"] for value in values),
                         max(value["value_summary"]["maximum"] for value in values)],
         "unit_summary_medians": {
             name: float(np.median([value["value_summary"][name] for value in values]))
             for name in ("zero_fraction", "p50", "p90", "p95", "p99", "positive_mean")
         }}
        for key, values in sorted(grouped.items())
    ]


def run(args):
    args.output.mkdir(parents=True, exist_ok=True)
    frozen = json.loads(args.frozen_split.read_text(encoding="utf-8"))
    rows, intervals = build_rows(args.real_root, args.sim_root, frozen)
    results = []
    for real_signal, sim_signal in (("processed", "processed"), ("raw", "raw")):
        results.extend(_evaluate_track(rows, real_signal, sim_signal, args.seeds))
    windows = _task_windows(args.sim_root, frozen["sim_roots"])
    task = _task_preservation(windows, args.seeds)
    _write_json(args.output / "domain_results.json", results)
    _write_json(args.output / "task_preservation.json", {
        "probe_results": task,
        "contact_onset": _contact_onsets(args.sim_root),
        "real_contact_truth_available": False,
    })
    _write_json(args.output / "signal_statistics.json", {
        "feature_names": feature_names(), "groups": _signal_statistics(rows),
        "sample_intervals": {key: {"count": len(values), "median": float(np.median(values)),
                                   "p05": float(np.quantile(values, 0.05)),
                                   "p95": float(np.quantile(values, 0.95))}
                             for key, values in sorted(intervals.items()) if values},
    })
    inventory = Counter((row["domain_name"], row["signal"], row["stage"], row["split"]) for row in rows)
    _write_json(args.output / "dataset_inventory.json", {
        "units": len(rows), "feature_dimension": len(feature_names()),
        "counts": [{"domain": key[0], "signal": key[1], "stage": key[2], "split": key[3], "units": value}
                   for key, value in sorted(inventory.items())],
        "frozen_split": str(args.frozen_split),
    })
    summary = {"schema_version": 1, "domain_tracks": {}, "sim_task_windows": len(windows),
               "limitations": ["Strict-replay dual export begins at APPROACH, so BASELINE is excluded.",
                               "Real raw and MuJoCo raw do not share physical units.",
                               "Real data has stage labels but no per-frame object-contact truth.",
                               "Only two simulation roots are held out for test."]}
    for real_signal, sim_signal in (("processed", "processed"), ("raw", "raw")):
        name = f"real_{real_signal}_vs_sim_{sim_signal}"
        selected = [item for item in results if item["real_signal"] == real_signal and item["sim_signal"] == sim_signal]
        summary["domain_tracks"][name] = min(selected, key=lambda item: abs(item["balanced_accuracy"] - 0.5))
    _write_json(args.output / "summary.json", summary)
    return 0


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real-root", type=Path, default=Path("outputs/realdata"))
    parser.add_argument("--sim-root", type=Path, default=Path("outputs/tactile_p18_2_2026-10-25/raw_domain_audit/sim_dual"))
    parser.add_argument("--frozen-split", type=Path, default=Path("outputs/tactile_p18_2_2026-10-25/diagnostic_baseline/split_manifest.json"))
    parser.add_argument("--output", type=Path, default=Path("outputs/tactile_p18_2_2026-10-25/raw_domain_audit/analysis"))
    parser.add_argument("--seeds", type=int, nargs="+", default=(0, 1, 2))
    return parser


if __name__ == "__main__":
    raise SystemExit(run(build_parser().parse_args()))
