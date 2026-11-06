"""Independent low-level control replay using unchanged task success criteria."""

import argparse
import hashlib
import json
from pathlib import Path
import time

import mujoco
import numpy as np

from apps.run_p29_hd_replay_smoke import FACTORIES, _task_config, _task_name
from source.data.production_conditions import environment_conditions


def replay(manifest, *, exact_state=True):
    payload = json.loads(manifest.read_text())
    task = _task_name(payload)
    with np.load(manifest.parent / payload["arrays"], allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    has_pre_state = "pre_qpos" in arrays and "pre_qvel" in arrays
    first = 0 if has_pre_state else 1
    if len(arrays["qpos"]) <= first:
        raise ValueError("Not enough recorded controls for independent replay")
    conditions = payload.get("metadata", {}).get("online_contract", {}).get("environment_conditions", {})
    config = environment_conditions(task, _task_config(payload, task), conditions)
    env = FACTORIES[task](task_config=config, control_mode="position",
                          enable_tactile_sensors=True, control_dt=.05,
                          episode_length=len(arrays["qpos"]) + 20)
    try:
        env.reset(seed=int(payload["seed"]))
        exact = exact_state and 'pre_integration_state' in arrays and 'applied_ctrl64' in arrays
        if exact:
            spec = mujoco.mjtState.mjSTATE_INTEGRATION
            contract = payload['metadata']['online_contract']
            if (contract.get('integration_state_spec') != int(spec)
                    or arrays['pre_integration_state'].shape[1] != mujoco.mj_stateSize(env.model, spec)
                    or arrays['applied_ctrl64'].shape[1] != env.model.nu):
                raise ValueError('Full integration/control state differs from replay model')
            mujoco.mj_setState(env.model, env.data, arrays['pre_integration_state'][0], spec)
        else:
            env.data.qpos[:] = arrays["pre_qpos"][0] if has_pre_state else arrays["qpos"][0]
            env.data.qvel[:] = arrays["pre_qvel"][0] if has_pre_state else arrays["qvel"][0]
            env.data.ctrl[:] = arrays["ctrl"][0]
        mujoco.mj_forward(env.model, env.data)
        ids = np.concatenate((env.controller.arm_controller.actuator_ids,
                              env.controller.hand_controller.actuator_ids))
        results = []
        max_position_error = 0.
        tactile_max = 0.
        for index in range(first, len(arrays["qpos"])):
            action = arrays.get("position_action", arrays["ctrl"][:, ids])[index]
            if exact:
                env.data.ctrl[:] = arrays['applied_ctrl64'][index]
                env.step_physics(env.physics_steps_per_control, control_updates=1)
                env.elapsed_steps += 1
                observation = env._get_observation()
                result = env.task.evaluate(observation, np.asarray(action), env.model, env.data)
                terminated, truncated = result.terminated, env.elapsed_steps >= env.config.episode_length
                info = {'task_success': result.success}
            else:
                observation, _, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                raise ValueError("Replay terminated before completing recorded controls")
            if not all(np.isfinite(observation[key]).all() for key in ("qpos", "qvel", "tactile")):
                raise ValueError("Non-finite replay observation")
            tactile_max = max(tactile_max, float(observation["tactile"].max()))
            max_position_error = max(max_position_error, float(np.max(np.abs(
                observation["qpos"] - arrays["qpos"][index]))))
            results.append(bool(info["task_success"]))
        return {"task": task, "frames": len(results), "initial_state": "pre_action" if has_pre_state else "post_first_action",
                "skipped_initial_actions": first, "source_success": bool(payload["success"]),
                "exact_integration_state": exact,
                "exact_recorded_ctrl": exact,
                "replay_final_success": results[-1], "success_frames": sum(results),
                "max_qpos_component_error": max_position_error, "tactile_max": tactile_max,
                "unchanged_task_criterion": True,
                "passed": bool(results[-1] == bool(payload["success"]) and (not payload["success"] or tactile_max > 0)),
                "note": "Task-specific independent replay; not the Lift FINAL Strict profile."}
    finally:
        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument('--legacy-precision-diagnostic', action='store_true',
                        help='Diagnostic only: replay float32 qpos/qvel/commands even when v4 checkpoints exist')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    started = time.monotonic()
    result = {"manifest": str(args.manifest), "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest()}
    try:
        result.update(replay(args.manifest, exact_state=not args.legacy_precision_diagnostic))
    except Exception as error:
        result.update(passed=False, error=f"{type(error).__name__}: {error}")
        raise
    finally:
        result["wall_seconds"] = time.monotonic() - started
        with args.output.open("x") as stream:
            json.dump(result, stream, indent=2)
        print(json.dumps(result, indent=2))
