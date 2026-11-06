"""One bounded, isolated simulation job with online recording and replay."""

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import signal
import time
import traceback

from apps.replay_p32_task import replay
from apps.run_p32_online_contract import run
from source.verification.strict_replay import strict_replay_manifest
from source.data.production_templates import asset_digest, validate_candidate_sampling
from source.data.production_failures import failure_policy
from source.data.production_profile import PhaseProfile


def execute(payload, output, budget):
    def expired(signum, _):
        if signum == signal.SIGTERM:
            raise InterruptedError('Worker received SIGTERM')
        raise TimeoutError("Bounded worker wall-clock budget expired")

    signal.signal(signal.SIGALRM, expired)
    signal.signal(signal.SIGTERM, expired)
    signal.alarm(int(budget))
    started = time.monotonic()
    profile = PhaseProfile()
    result = {"complete": False}
    phase = 'INPUT_VALIDATION'
    try:
        validate_candidate_sampling(payload)
        source = Path(payload["reference_manifest"])
        if asset_digest(Path(payload['asset_path'])) != payload['asset_hash']:
            raise ValueError('Live asset changed after planning')
        if hashlib.sha256(source.read_bytes()).hexdigest() != payload["reference_manifest_sha256"]:
            raise ValueError("Template hash changed after planning")
        for dependency in payload.get('dependencies', []):
            if hashlib.sha256(Path(dependency['path']).read_bytes()).hexdigest() != dependency['sha256']:
                raise ValueError('Frozen source dependency changed after planning')
        if payload.get("handle_manifest"):
            handle = Path(payload["handle_manifest"])
            if hashlib.sha256(handle.read_bytes()).hexdigest() != payload["handle_manifest_sha256"]:
                raise ValueError("Handle template hash changed after planning")
        if payload.get('lift_source_manifest'):
            if hashlib.sha256(Path(payload['lift_source_manifest']).read_bytes()).hexdigest() != payload['lift_source_manifest_sha256']:
                raise ValueError('Pinned Lift source changed after planning')
        profile.add('input_validation', time.monotonic() - started)
        phase = 'SIMULATION'
        record = profile.call('online_execution_inclusive', run, payload['task'], source, payload['seed'], output, rgb=True,
                     handle_manifest=Path(payload["handle_manifest"]) if payload.get("handle_manifest") else None,
                     nut_grasp_offset_z=payload.get("nut_grasp_offset_z", 0.),
                     nut_grip_policy=payload.get("nut_grip_policy", "adaptive"),
                     conditions=payload["conditions"],
                     lift_source_manifest=Path(payload['lift_source_manifest']) if payload.get('lift_source_manifest') else None,
                     capture=payload.get('rgb_capture'))
        phase = 'REPLAY'
        manifest = output / "source_episode/manifest.json"
        verification = {"manifest": str(manifest), "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
                        **profile.call('independent_replay', replay, manifest)}
        with (output / "independent_replay.json").open("x") as stream:
            json.dump(verification, stream, indent=2)
        if not verification["passed"]:
            raise ValueError("Independent replay did not reproduce task outcome")
        if payload["task"] == "lift":
            phase = 'STRICT'
            strict = asdict(profile.call('lift_final_strict', strict_replay_manifest, manifest, use_cache=False))
            with (output / "strict_final.json").open("x") as stream:
                json.dump({"manifest_sha256": verification["manifest_sha256"], "result": strict}, stream, indent=2)
            if record["success"] and not strict["success"]:
                raise ValueError("Lift FINAL Strict rejected generated episode")
        result.update(complete=True, success=record["success"], replay_passed=True)
        return result
    except BaseException as error:
        result.update(error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc(),
                      failure=failure_policy(error, phase))
        raise
    finally:
        signal.alarm(0)
        result["wall_seconds"] = time.monotonic() - started
        if phase == 'INPUT_VALIDATION':
            profile.add('input_validation', result['wall_seconds'])
        result['wall_profile'] = profile.snapshot()
        output.mkdir(parents=True, exist_ok=True)
        with (output / "worker_result.json").open("x") as stream:
            json.dump(result, stream, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--budget-seconds", type=int, default=180)
    args = parser.parse_args()
    if args.output.exists() or args.budget_seconds < 1:
        raise ValueError("Worker requires a fresh output and finite positive budget")
    print(json.dumps(execute(json.loads(args.job.read_text()), args.output, args.budget_seconds), indent=2))
