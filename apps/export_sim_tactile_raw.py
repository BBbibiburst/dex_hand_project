"""Replay existing episodes and export aligned raw/processed simulation tactile signals."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import mujoco
import numpy as np

from source.envs.manipulation import make_lift_env
from source.grasping.contracts import DemonstrationEpisode
from source.grasping.executor import _palm_object_contact, _robot_object_contact_summary
from source.sensors.tactile.signal_processing import TactileSignalProcessor
from source.verification.strict_replay import load_replay_controls


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
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


def _array_hashes(arrays: dict[str, np.ndarray]) -> dict[str, str]:
    return {
        name: _sha256_bytes(np.ascontiguousarray(value).view(np.uint8).tobytes())
        for name, value in arrays.items()
    }


def export_manifest(manifest: Path, output: Path) -> dict:
    manifest = manifest.resolve()
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    source = DemonstrationEpisode.load(manifest)
    metadata = dict(payload.get("metadata", {}))
    control_dt = float(metadata.get("control_dt", 0.05))
    source_seed = int(metadata.get("source_seed", payload.get("seed", 0)))
    env = make_lift_env(
        task_config={"object_id": source.object_id, "reward_shaping": False, "terminate_on_success": False},
        control_mode="position",
        enable_tactile_sensors=True,
        render_mode=None,
        control_dt=control_dt,
        episode_length=int(payload.get("frames", 300)) + 20,
    )
    try:
        reset_observation, _ = env.reset(seed=source_seed)
        sensor = env.tactile_sensor
        processor_config = asdict(sensor.signal_processor.config)
        validator = TactileSignalProcessor(processor_config)
        validator.reset(sensor.total_taxels)
        reset_raw = sensor.read_raw(env.model, env.data)
        reset_reconstructed = validator.process(reset_raw, sensor.patches)
        reset_error = float(np.max(np.abs(reset_reconstructed - reset_observation["tactile"])))

        controls, initial_qpos, initial_qvel, initial_ctrl = load_replay_controls(manifest, env)
        controls = np.asarray(controls, dtype=np.float32)
        source_stages = np.asarray(source.arrays["stage"])
        if len(controls) == len(source_stages):
            source_offset = 0
        else:
            approach = int(source.metadata["stage_codes"]["approach"])
            matches = np.flatnonzero(source_stages == approach)
            if not len(matches) or len(source_stages) - int(matches[0]) != len(controls):
                raise ValueError(f"Controls/stages differ: {len(controls)} != {len(source_stages)}.")
            source_offset = int(matches[0])
        env.data.qpos[:] = initial_qpos
        env.data.qvel[:] = initial_qvel
        if initial_ctrl is not None and np.asarray(initial_ctrl).shape == env.data.ctrl.shape:
            env.data.ctrl[:] = initial_ctrl
        mujoco.mj_forward(env.model, env.data)

        values = {
            "raw_tactile": [], "processed_tactile": [], "source_processed_tactile": [],
            "simulation_time": [], "frame_index": [], "stage": [], "qpos": [], "qvel": [], "ctrl": [],
            "object_position": [], "object_quaternion_wxyz": [], "robot_object_contact_count": [],
            "robot_object_digit_contact_count": [], "robot_object_digit_normal_force": [],
            "palm_object_contact": [], "task_success": [],
        }
        reconstruction_errors = []
        state_mutation_errors = []
        for index, control in enumerate(controls):
            observation, _, terminated, truncated, info = env.step(control)
            before = np.concatenate((env.data.qpos.copy(), env.data.qvel.copy(), env.data.ctrl.copy(), [env.data.time]))
            raw = sensor.read_raw(env.model, env.data)
            after = np.concatenate((env.data.qpos.copy(), env.data.qvel.copy(), env.data.ctrl.copy(), [env.data.time]))
            reconstructed = validator.process(raw, sensor.patches)
            processed = np.asarray(observation["tactile"], dtype=np.float32)
            reconstruction_errors.append(float(np.max(np.abs(reconstructed - processed))))
            state_mutation_errors.append(float(np.max(np.abs(after - before))))
            _, _, digit_counts, digit_forces = _robot_object_contact_summary(env, "object")
            values["raw_tactile"].append(raw)
            values["processed_tactile"].append(processed)
            source_index = source_offset + index
            values["source_processed_tactile"].append(
                np.asarray(source.arrays["tactile"][source_index], dtype=np.float32)
            )
            values["simulation_time"].append(float(env.data.time))
            values["frame_index"].append(index)
            values["stage"].append(int(source.arrays["stage"][source_index]))
            values["qpos"].append(env.data.qpos.astype(np.float32).copy())
            values["qvel"].append(env.data.qvel.astype(np.float32).copy())
            values["ctrl"].append(env.data.ctrl.astype(np.float32).copy())
            values["object_position"].append(np.asarray(observation["object_pos"], dtype=np.float32))
            values["object_quaternion_wxyz"].append(np.asarray(observation["object_quat"], dtype=np.float32))
            values["robot_object_contact_count"].append(int(np.sum(digit_counts)))
            values["robot_object_digit_contact_count"].append(digit_counts)
            values["robot_object_digit_normal_force"].append(digit_forces)
            values["palm_object_contact"].append(_palm_object_contact(env, "object"))
            values["task_success"].append(bool(info.get("task_success", False)))
            if terminated or truncated:
                break
        arrays = {name: np.asarray(items) for name, items in values.items()}
        if len(arrays["raw_tactile"]) != len(controls):
            raise RuntimeError("Replay ended before all controls were exported.")
        for name in ("raw_tactile", "processed_tactile", "source_processed_tactile"):
            if arrays[name].shape != (len(controls), 712) or not np.isfinite(arrays[name]).all():
                raise ValueError(f"Invalid {name}: {arrays[name].shape}.")
        source_qpos = np.asarray(source.arrays["qpos"])[source_offset : source_offset + len(controls)]
        source_qpos_error = float(np.max(np.abs(arrays["qpos"] - source_qpos)))
        source_processed_error = float(
            np.max(np.abs(arrays["processed_tactile"] - arrays["source_processed_tactile"]))
        )
        time_differences = np.diff(arrays["simulation_time"])
        time_error = float(np.max(np.abs(time_differences - control_dt))) if len(time_differences) else 0.0
        output.mkdir(parents=True, exist_ok=True)
        array_path = output / "tactile_dual.npz"
        np.savez_compressed(array_path, **arrays)
        code_paths = (
            Path("source/sensors/tactile/dex_hand.py"),
            Path("source/sensors/tactile/signal_processing.py"),
            Path("configs/robot_profiles/rm75b_dex_hand.json"),
        )
        contract = {
            "schema_version": 1,
            "source_manifest": str(manifest),
            "source_manifest_sha256": _sha256_file(manifest),
            "source_episode_sha256": _sha256_file(manifest.parent / str(payload["arrays"])),
            "object_id": source.object_id,
            "source_seed": source_seed,
            "root_id": manifest.parents[1].name,
            "mode": manifest.parents[3].name,
            "frames": len(controls),
            "source_frame_offset": source_offset,
            "control_dt": control_dt,
            "source_success": bool(payload["success"]),
            "raw_contract": "MuJoCo mjSENS_TOUCH scalar values copied from data.sensordata at bound sensor addresses after env.step",
            "processed_contract": processor_config,
            "patch_metadata": sensor.metadata()["patches"],
            "code_sha256": {str(path): _sha256_file(path) for path in code_paths},
            "checks": {
                "reset_reconstruction_max_abs_error": reset_error,
                "frame_reconstruction_max_abs_error": max(reconstruction_errors, default=0.0),
                "read_raw_state_mutation_max_abs_error": max(state_mutation_errors, default=0.0),
                "source_qpos_max_abs_error": source_qpos_error,
                "source_processed_max_abs_error": source_processed_error,
                "simulation_time_step_max_abs_error": time_error,
                "all_finite": True,
            },
            "array_sha256": _array_hashes(arrays),
        }
        contract["tactile_config_hash"] = _sha256_bytes(
            json.dumps(
                {"processor": processor_config, "patches": contract["patch_metadata"], "code": contract["code_sha256"]},
                sort_keys=True,
            ).encode()
        )
        contract["npz_sha256"] = _sha256_file(array_path)
        _write_json(output / "manifest.json", contract)
        return contract
    finally:
        env.close()


def run(args) -> int:
    manifests = []
    for pattern in args.pattern:
        manifests.extend(args.source_root.glob(pattern))
    manifests = sorted(dict.fromkeys(path.resolve() for path in manifests))
    if args.maximum is not None:
        manifests = manifests[: args.maximum]
    if not manifests:
        raise ValueError("No source manifests found.")
    results = []
    for manifest in manifests:
        relative = manifest.relative_to(args.source_root.resolve())
        target = args.output / relative.parent
        result = export_manifest(manifest, target)
        results.append(result)
        print(
            f"EXPORTED mode={result['mode']} root={result['root_id']} frames={result['frames']} "
            f"raw_mutation={result['checks']['read_raw_state_mutation_max_abs_error']:.3g} "
            f"source_qpos={result['checks']['source_qpos_max_abs_error']:.3g}",
            flush=True,
        )
    summary = {
        "schema_version": 1,
        "source_root": str(args.source_root.resolve()),
        "episodes": len(results),
        "roots": len({result["root_id"] for result in results}),
        "modes": sorted({result["mode"] for result in results}),
        "tactile_config_hashes": sorted({result["tactile_config_hash"] for result in results}),
        "maximum_checks": {
            key: max(result["checks"][key] for result in results)
            for key in results[0]["checks"]
            if key != "all_finite"
        },
        "all_finite": all(result["checks"]["all_finite"] for result in results),
        "manifests": [str((args.output / Path(result["mode"]) / "ycb_004_sugar_box" / result["root_id"] / "candidate_00" / "manifest.json")) for result in results],
    }
    _write_json(args.output / "summary.json", summary)
    return 0


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=Path("outputs/dp_p15_2026-10-24/candidates"))
    parser.add_argument("--output", type=Path, default=Path("outputs/tactile_p18_2_2026-10-25/raw_domain_audit/sim_dual"))
    parser.add_argument("--pattern", action="append", default=["mode_*/ycb_004_sugar_box/p*/candidate_00/manifest.json"])
    parser.add_argument("--maximum", type=int)
    return parser


if __name__ == "__main__":
    raise SystemExit(run(build_parser().parse_args()))
