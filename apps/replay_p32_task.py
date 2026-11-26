"""Independent low-level control replay using unchanged task success criteria."""

import argparse
import hashlib
import gzip
import json
from pathlib import Path
import time

import mujoco
import numpy as np

from apps.run_p29_hd_replay_smoke import FACTORIES, _task_config, _task_name
from source.data.production_conditions import environment_conditions


class _RecordedObjectSpec:
    def __init__(self, original, geom_names):
        self.original=original;self.geom_names=tuple(geom_names)
    def __getattr__(self, name):return getattr(self.original,name)


def bind_recorded_objects(task, model):
    """Bind the captured collision pieces, not a host's rebuilt decomposition cache."""
    objects=[]
    for obj in task.objects:
        body=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,obj.body_name)
        if body<0:raise ValueError('Recorded object body missing: '+obj.body_name)
        names=[mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_GEOM,i) for i in range(model.ngeom)
               if model.geom_bodyid[i]==body and (model.geom_contype[i] or model.geom_conaffinity[i])]
        if set(obj.geom_names)!=set(names):
            if not names or any(n is None for n in names):raise ValueError('Recorded collision identities missing')
            objects.append(_RecordedObjectSpec(obj,names))
        else:objects.append(obj)
    task._objects=tuple(objects)
    task.bind(model)


def bind_recorded_model(env, manifest, payload):
    """Replay the captured compiled model, not a numerically recompiled mesh."""
    contract=payload.get('metadata',{}).get('online_contract',{})
    proof=contract.get('model_provenance')
    if proof is None:return False
    if proof.get('mujoco_version')!=mujoco.__version__:
        raise ValueError('Recorded model requires its original MuJoCo version')
    name=proof.get('compiled_model_file')
    if name not in ('model.mjb','model.mjb.gz'):raise ValueError('Unexpected recorded model filename')
    path=manifest.parent/name
    if path.stat().st_size>256*1024**2:raise ValueError("Stored model exceeds replay size bound")
    stored=path.read_bytes()
    if proof.get('stored_model_sha256') and hashlib.sha256(stored).hexdigest()!=proof['stored_model_sha256']:
        raise ValueError('Recorded model file checksum mismatch')
    if name.endswith('.gz'):
        with gzip.open(path,'rb') as stream:raw=stream.read(256*1024**2+1)
    else:raw=stored
    if len(raw)>256*1024**2 or hashlib.sha256(raw).hexdigest()!=proof['compiled_model_sha256']:
        raise ValueError('Recorded compiled model checksum mismatch')
    model=mujoco.MjModel.from_binary_path('model.mjb',assets={'model.mjb':raw})
    if any(getattr(model,k)!=getattr(env.model,k) for k in ('nq','nv','nu','nsensor','nsensordata')):
        raise ValueError('Recorded model layout differs from environment bindings')
    env.model=model;env.data=mujoco.MjData(model)
    env.controller.bind(model,env.data);env.tactile_sensor.bind(model,env.data);bind_recorded_objects(env.task,model)
    env._initial_qpos=env.data.qpos.copy();env._initial_qvel=env.data.qvel.copy()
    return True


def replay(manifest, *, exact_state=True, require_trajectory=True, trajectory_tolerance=1e-6):
    if not np.isfinite(trajectory_tolerance) or not 0<trajectory_tolerance<=1e-6:
        raise ValueError("Trajectory precision gate cannot exceed 1e-6 numeric qpos units")
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
        recorded_model=bind_recorded_model(env,manifest,payload)
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
        trajectory_ok=bool(max_position_error<=trajectory_tolerance)
        if require_trajectory is None:
            require_trajectory=payload.get('metadata',{}).get('online_contract',{}).get('probe_state_version')=='full-integration-and-sensor-v1'
        outcome_ok=bool(results[-1]==bool(payload['success']) and (not payload['success'] or tactile_max>0))
        return {"task": task, "frames": len(results), "initial_state": "pre_action" if has_pre_state else "post_first_action",
                "skipped_initial_actions": first, "source_success": bool(payload["success"]),
                "exact_integration_state": exact,
                "exact_recorded_ctrl": exact,
                "replay_final_success": results[-1], "success_frames": sum(results),
                "max_qpos_component_error": max_position_error, "tactile_max": tactile_max,
                "unchanged_task_criterion": True,
                "recorded_model_used": recorded_model,
                "trajectory_reproduced": trajectory_ok,
                "trajectory_tolerance": trajectory_tolerance,
                "trajectory_error_units": "numeric qpos components at recorded float32 precision; not a unified physical unit",
                "acceptance_scope": "outcome_and_open_loop_trajectory" if require_trajectory else "legacy_outcome_only",
                "outcome_reproduced": outcome_ok,
                "passed": bool(outcome_ok and (not require_trajectory or trajectory_ok)),
                "note": "Task-specific independent replay; not the Lift FINAL Strict profile."}
    finally:
        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument('--legacy-precision-diagnostic', action='store_true',
                        help='Diagnostic only: replay float32 qpos/qvel/commands even when v4 checkpoints exist')
    parser.add_argument("--legacy-outcome-only",action="store_true",help="Historical diagnostic only; never production admission")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    started = time.monotonic()
    result = {"manifest": str(args.manifest), "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest()}
    try:
        result.update(replay(args.manifest, exact_state=not args.legacy_precision_diagnostic, require_trajectory=not args.legacy_outcome_only))
    except Exception as error:
        result.update(passed=False, error=f"{type(error).__name__}: {error}")
        raise
    finally:
        result["wall_seconds"] = time.monotonic() - started
        with args.output.open("x") as stream:
            json.dump(result, stream, indent=2)
        print(json.dumps(result, indent=2))
