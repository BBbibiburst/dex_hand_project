"""Bounded P32 online contract check; one template, one actual execution."""

from __future__ import annotations

import argparse
from dataclasses import replace
from contextlib import ExitStack
import importlib
import json
import hashlib
from pathlib import Path
import time
from unittest.mock import patch

import numpy as np
import mujoco

from apps.produce_five_task_dataset import _execute_task
from source.data.production_recorder import SCHEMA, OnlineEnvironment, recorder_type, validate_online_arrays, attach_stage_encoding
from source.data.production_video import OnlineVideo
from source.grasping.executor import _Recorder
from source.grasping.task_transfer import PickPlaceTransferConfig
from source.envs.manipulation.placement import FixedTablePlacementSampler
from source.grasping.nut_transfer import execute_nut_transfer, NutTransferConfig
from tools.grasping.transfer_lift_to_pick_place import load_lift_source
from tools.grasping.transfer_lift_to_stack import _execution_config
from source.grasping.executor import quaternion_wxyz_to_matrix
from source.data.production_conditions import environment_conditions, shifted_pose, validate_conditions
from source.grasping.push_transfer import PushTransferConfig
from source.data.production_profile import PhaseProfile


MODULES = {
    "executor": "make_lift_env",
    "task_transfer": "make_pick_place_env",
    "stack_transfer": "make_stack_env",
    "push_transfer": "make_push_env",
    "nut_transfer": "make_nut_assembly_env",
}


def run(task, manifest, seed, output, rgb=False, handle_manifest=None, nut_grasp_offset_z=0.,
        nut_grip_policy="adaptive", conditions=None, lift_source_manifest=None, capture=None):
    conditions = validate_conditions(task, conditions or {})
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    profile = PhaseProfile()
    record = {"task": task, "reference_manifest": str(manifest), "seed": seed,
              "conditions": conditions,
              "simulation_executed": False, "rgb_status": "NOT_YET_IMPLEMENTED"}
    online_recorder = recorder_type(_Recorder)
    environments = []
    video = OnlineVideo(output / "agentview.mp4", capture=capture) if rgb else None
    reference = json.loads(manifest.read_text())
    record["reference_sha256"] = hashlib.sha256(manifest.read_bytes()).hexdigest()

    def factory(original):
        def create(*args, **kwargs):
            kwargs["enable_tactile_sensors"] = True
            if task == "lift":
                metadata = reference["metadata"]
                position, quaternion = shifted_pose(metadata["object_initial_position"],
                                                    metadata["object_initial_quaternion_wxyz"], conditions)
                task_config = dict(kwargs.get("task_config", {}))
                task_config.update({
                    "reward_shaping": False, "terminate_on_success": False,
                    "placement_sampler": FixedTablePlacementSampler(
                        xy=(position[0] - .55, position[1]),
                        quaternion_wxyz=quaternion,
                        world_z=position[2]),
                })
                kwargs["task_config"] = task_config
            kwargs["task_config"] = environment_conditions(task, kwargs.get("task_config", {}), conditions)
            env = profile.call('environment_create', original, *args, **kwargs)
            wrapped = OnlineEnvironment(env, video=video, profile=profile)
            if not np.isclose(wrapped.model.opt.timestep, .001, rtol=0, atol=1e-12):
                wrapped.close()
                raise ValueError("Production contract requires 1000Hz physics")
            environments.append(wrapped)
            return wrapped
        return create

    try:
        with ExitStack() as stack:
            for name, attribute in MODULES.items():
                module = importlib.import_module("source.grasping." + name)
                stack.enter_context(patch.object(module, attribute, factory(getattr(module, attribute))))
                stack.enter_context(patch.object(module, "_Recorder", online_recorder))
                if hasattr(module, "_select_transport_hand"):
                    original_probe = module._select_transport_hand

                    def suppress_probe(env, *args, _probe=original_probe, **kwargs):
                        before = env.capture_video
                        env.capture_video = False
                        try:
                            return _probe(env, *args, **kwargs)
                        finally:
                            env.capture_video = before

                    stack.enter_context(patch.object(module, "_select_transport_hand", suppress_probe))
            if task == "nut_assembly" and nut_grip_policy == "retain":
                module = importlib.import_module("source.grasping.nut_transfer")
                stack.enter_context(patch.object(module, "_select_transport_hand",
                                                lambda env, grip, name: grip.copy()))
            # The legacy default copies the old Lift prefix into a new episode.
            # P32 must retain the newly executed prefix and its fresh sensors.
            import apps.produce_five_task_dataset as producer
            original_config = producer._dataclass_from_dict
            if lift_source_manifest is not None:
                stack.enter_context(patch.object(producer, '_successful_source',
                                                lambda _: load_lift_source(lift_source_manifest)))
                record['lift_source_manifest'] = str(lift_source_manifest)

            def config(kind, payload):
                value = original_config(kind, payload)
                if kind is PickPlaceTransferConfig:
                    value = replace(value, reuse_verified_lift_terminal_state=False)
                elif kind is PushTransferConfig:
                    value = replace(value, direction_degrees=value.direction_degrees + conditions["push_direction_delta_deg"])
                return value

            stack.enter_context(patch.object(producer, "_dataclass_from_dict", config))
            if task == "pick_place":
                original_pick = producer.execute_pick_place_transfer

                def planned_pick(candidate, **kwargs):
                    if any(conditions[key] for key in ('dx_m', 'dy_m', 'yaw_deg')):
                        position, quaternion = shifted_pose(kwargs["source_object_position"],
                                                            kwargs["source_object_quaternion_wxyz"], conditions)
                        kwargs.update(source_object_position=position, source_object_quaternion_wxyz=quaternion,
                                      source_episode=None, source_trajectory=None)
                    elif kwargs.get('source_episode') is not None:
                        trajectory = kwargs['source_episode'].metadata.get('ppo_source_manifest')
                        if trajectory:
                            # The legacy trajectory branch restores an approach
                            # state in the middle of a recorded episode. New
                            # production must solve and execute from reset.
                            kwargs.update(source_episode=None, source_trajectory=None)
                    return original_pick(candidate, **kwargs)

                stack.enter_context(patch.object(producer, "execute_pick_place_transfer", planned_pick))
            record["simulation_executed"] = True
            if task == "nut_assembly":
                if handle_manifest is None:
                    raise ValueError("Nut requires original --handle-manifest, not an already transformed Nut candidate")
                source = load_lift_source(handle_manifest)
                # A planned grasp offset changes the hand target, never the
                # bolt geometry or success criterion. Convert world-up to the
                # original handle source frame before the executor transforms it.
                if abs(nut_grasp_offset_z) > .02:
                    raise ValueError("Nut diagnostic grasp offset limited to 20mm")
                rotation = quaternion_wxyz_to_matrix(np.asarray(
                    source.metadata["object_initial_quaternion_wxyz"]))
                source.candidate = replace(source.candidate, hand_translation=(
                    source.candidate.hand_translation + rotation.T @ np.array([0., 0., nut_grasp_offset_z])))
                record["nut_grasp_offset_z"] = nut_grasp_offset_z
                record["nut_grip_policy"] = nut_grip_policy
                record["handle_manifest"] = str(handle_manifest)
                record["handle_sha256"] = hashlib.sha256(handle_manifest.read_bytes()).hexdigest()
                episode = profile.call('task_execution_inclusive', execute_nut_transfer,
                    source, variant_index=int(reference["object_id"].rsplit(":", 1)[1]),
                    seed=seed, grasp_config=_execution_config(source),
                    transfer_config=NutTransferConfig(orientation_corrections=0, yaw_corrections=0))
            else:
                episode = profile.call('task_execution_inclusive', _execute_task,
                                       {"task": task, "reference_manifest": str(manifest), "seed": seed})
        from source.data.production_provenance import capture_provenance

        provenance = profile.call('model_capture_write', capture_provenance, environments[-1], output / 'model.mjb.gz')
        episode.metadata["online_contract"] = {
            "schema": SCHEMA,
            "integration_state_spec": int(mujoco.mjtState.mjSTATE_INTEGRATION),
            "exact_control_layout": "full_model_ctrl_float64",
            "environment_conditions": conditions,
            "observation_timing": "before_action",
            "state_command_timing": "before_command_assignment",
            "task_outcome_timing": "after_action",
            "control_dt": 0.05,
            "physics_dt": float(environments[-1].model.opt.timestep),
            "state_layout": "actuator_position_velocity_command_v1",
            "position_action_layout": "arm7_hand6_position_command_v1",
            "ik_action_layout": "xyz_quaternion_wxyz_hand6",
            "object_names": list(environments[-1].object_names),
            "model_provenance": provenance,
            "sensor": environments[-1].tactile_sensor.metadata(),
            "sensor_sampling": "one_processed_read_per_time_and_raw_state",
            "reference_manifest_sha256": record["reference_sha256"],
            "rgb_status": "NOT_YET_IMPLEMENTED",
            "production_eligible": False,
        }
        record['stage_encoding_check'] = attach_stage_encoding(episode)
        if video is not None:
            record["video"] = video.finish()
            if record["video"]["control_frames"] != len(episode.arrays["timestamp"]):
                raise ValueError("Video sampling control count differs from online modality count")
            episode.metadata['online_contract']['rgb_capture'] = video.capture
            episode.metadata['online_contract']['rgb_control_indices'] = list(video.control_indices)
            record["rgb_status"] = "DECODE_VERIFIED"
            episode.metadata["online_contract"]["rgb_status"] = "DECODE_VERIFIED"
        profile.call('source_episode_write', episode.save, output / 'source_episode')
        (output / "model.mjb.gz").rename(output / "source_episode/model.mjb.gz")
        record["success"] = episode.success
        record["failure_reason"] = episode.failure_reason
        record["contract"] = validate_online_arrays(episode.arrays, require_causal=True)
        record["object_names"] = list(environments[-1].object_names)
        # Canonical v2 arrays use actual arm7+hand6 position commands. Legacy
        # source_episode/action remains Cartesian for existing task replay tools.
        canonical = {
            "state": episode.arrays["state"],
            "action": episode.arrays["position_action"],
            "tactile": episode.arrays["sensor_tactile"],
            "object_state": episode.arrays["object_state"],
            "timestamp": episode.arrays["timestamp"],
            "next_timestamp": episode.arrays["next_timestamp"],
            "stage": episode.arrays["stage"],
            "success": episode.arrays["task_success"],
            "frame_index": np.arange(len(episode.arrays["timestamp"]), dtype=np.int64),
        }
        profile.call('canonical_arrays_write', np.savez_compressed, output / 'modalities.npz', **canonical)
        record["modalities_sha256"] = hashlib.sha256((output / "modalities.npz").read_bytes()).hexdigest()
        record["contract_passed"] = True
    except Exception as error:
        record["error"] = f"{type(error).__name__}: {error}"
        record["contract_passed"] = False
        raise
    finally:
        if video is not None and not video.closed:
            video.abort()
        for env in environments:
            if env.renderer is not None:
                env.close()
        record["wall_seconds"] = time.monotonic() - started
        record['wall_profile'] = profile.snapshot()
        if video is not None:
            record['video_wall_profile'] = video.profile.snapshot()
        (output / "result.json").write_text(json.dumps(record, indent=2) + "\n")
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("lift", "pick_place", "stack", "push", "nut_assembly"), required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rgb", action="store_true", help="Capture native 1080p20 during execution")
    parser.add_argument("--handle-manifest", type=Path)
    parser.add_argument("--nut-grasp-offset-z", type=float, default=0.)
    parser.add_argument("--nut-grip-policy", choices=("adaptive", "retain"), default="adaptive")
    parser.add_argument("--conditions", type=Path, help="JSON task-specific initial/goal perturbation")
    parser.add_argument('--lift-source-manifest', type=Path)
    args = parser.parse_args()
    print(json.dumps(run(args.task, args.manifest, args.seed, args.output, args.rgb,
                         args.handle_manifest, args.nut_grasp_offset_z, args.nut_grip_policy,
                         json.loads(args.conditions.read_text()) if args.conditions else None,
                         args.lift_source_manifest), indent=2))


if __name__ == "__main__":
    main()
