"""Online pre-action records for the versioned production contract.

This opt-in environment adapter keeps legacy task recorders intact. Only steps
accepted by their recorder are committed, excluding planner probe rollouts.
"""

from __future__ import annotations

import mujoco
import numpy as np

from source.imitation.observation import real_observable_state


SCHEMA = "five-task-online-v4"
STAGE_ENCODING = 'executor-stage-codes-v1'
STAGE_CODES_V1 = dict(zip(('settle', 'transit', 'pregrasp', 'approach', 'close', 'hold', 'lift', 'verify',
                          'transport', 'descend', 'release', 'retreat', 'task_verify', 'clear'), range(14)))


def validate_stage_encoding(arrays, metadata):
    encoding = metadata.get('online_contract', {}).get('stage_encoding')
    if encoding is None:
        return {'checked': False, 'reason': 'Legacy recording has no versioned stage mapping'}
    if encoding != STAGE_ENCODING:
        raise ValueError('Unsupported stage encoding')
    mapping = metadata.get('stage_codes')
    if (not isinstance(mapping, dict) or not mapping
            or any(not isinstance(name, str) or not name or type(code) is not int or code < 0
                   for name, code in mapping.items())
            or len(set(mapping.values())) != len(mapping)):
        raise ValueError('Stage mapping must have unique nonnegative integer codes')
    stages = arrays.get('stage')
    if stages is None or stages.ndim != 1 or not np.issubdtype(stages.dtype, np.integer):
        raise ValueError('Stage array must be a one-dimensional integer array')
    if 'timestamp' in arrays and stages.shape != arrays['timestamp'].shape:
        raise ValueError('Stage and timestamp lengths differ')
    observed = {int(value) for value in np.unique(stages)}
    if not observed.issubset(mapping.values()):
        raise ValueError('Recorded stage code is absent from the stage mapping')
    if mapping != STAGE_CODES_V1:
        raise ValueError('Stage names/codes differ from the versioned contract')
    return {'checked': True, 'encoding': encoding, 'observed_codes': sorted(observed)}


def attach_stage_encoding(episode):
    from source.grasping.executor import STAGE_CODES

    if STAGE_CODES != STAGE_CODES_V1:
        raise ValueError('Executor stage codes changed; a new encoding version is required')
    episode.metadata['stage_codes'] = dict(STAGE_CODES)
    episode.metadata['online_contract']['stage_encoding'] = STAGE_ENCODING
    return validate_stage_encoding(episode.arrays, episode.metadata)


def validate_online_arrays(arrays: dict[str, np.ndarray], *, require_causal=False) -> dict:
    n = len(arrays["timestamp"])
    if n < 2:
        raise ValueError("Online episode requires at least two control steps")
    for key, tail in (("state", (39,)), ("position_action", (13,)),
                      ("ik_action", (13,)), ("sensor_tactile", (712,))):
        if arrays[key].shape != (n, *tail) or arrays[key].dtype != np.float32:
            raise ValueError(f"Invalid {key} contract: {arrays[key].shape}/{arrays[key].dtype}")
    poses = arrays["object_state"]
    if poses.ndim != 3 or poses.shape[0] != n or poses.shape[2] != 7 or poses.dtype != np.float32:
        raise ValueError("object_state must be float32[T,N,7]")
    if not np.allclose(np.linalg.norm(poses[..., 3:], axis=-1), 1, atol=1e-4):
        raise ValueError("Invalid object quaternion")
    if not all(np.isfinite(a).all() for a in arrays.values() if np.issubdtype(a.dtype, np.number)):
        raise ValueError("Non-finite online data")
    if np.any(arrays["sensor_tactile"] < 0):
        raise ValueError("Negative processed tactile")
    if not np.allclose(np.diff(arrays["timestamp"]), 0.05, atol=1e-7):
        raise ValueError("Non-contiguous 20Hz control timestamps")
    if not np.allclose(arrays["next_timestamp"] - arrays["timestamp"], 0.05, atol=1e-7):
        raise ValueError("Recorded transition duration is not 0.05 seconds")
    if require_causal and not np.allclose(arrays["state"][1:, -13:], arrays["position_action"][:-1],
                                          rtol=0, atol=1e-6):
        raise ValueError("Pre-action state command differs from previous applied action")
    continuity = require_causal and all(key in arrays for key in ('pre_qpos', 'pre_qvel', 'qpos', 'qvel'))
    if continuity and any(not np.array_equal(arrays[before][1:], arrays[after][:-1])
                          for before, after in (('pre_qpos', 'qpos'), ('pre_qvel', 'qvel'))):
        raise ValueError('Unrecorded physical state discontinuity between accepted control steps')
    exact_fields = ('pre_integration_state', 'applied_ctrl64')
    has_exact = any(key in arrays for key in exact_fields)
    if has_exact:
        if not all(key in arrays for key in exact_fields):
            raise ValueError('Incomplete exact replay state/control pair')
        for key in exact_fields:
            if arrays[key].dtype != np.float64 or arrays[key].ndim != 2 or arrays[key].shape[0] != n:
                raise ValueError('Exact replay arrays must be float64[T,D]')
    return {"schema": SCHEMA if has_exact else "five-task-online-v3", "frames": n, "duration_seconds": float(
        arrays["next_timestamp"][-1] - arrays["timestamp"][0]),
        "command_causality_checked": require_causal,
        "physics_state_continuity_checked": continuity,
        "tactile_nonzero_fraction": float(np.mean(arrays["sensor_tactile"] > 0))}


class OnlineEnvironment:
    """Capture actual sensor/state values before each physics control step."""

    def __init__(self, env, video=None, profile=None):
        from source.data.production_profile import PhaseProfile

        self.env = env
        self.profile = profile if profile is not None else PhaseProfile()
        self.video = video
        self.renderer = None
        self.capture_video = True
        self.pending = None
        self.prepared = None
        self.object_names = tuple(env.task.bindings.objects)
        self.body_ids = [env.task.bindings.objects[name].body_id for name in self.object_names]
        self.target_site = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_SITE, "target_region")
        nut = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, "fixed_nut_body")
        if nut >= 0:
            self.object_names += ("fixed_nut_body",)
            self.body_ids.append(nut)
        if self.target_site >= 0:
            self.object_names += ("target_region",)
        # Process each observed physical sensor state once. Additional pre-action
        # reads must not advance the low-pass filter or draw fresh sensor noise.
        sensor = env.tactile_sensor
        original_read = sensor.read
        self._sensor_cache = None

        def read_once(model, data):
            key = (float(data.time), sensor.read_raw(model, data).tobytes())
            if self._sensor_cache is None or self._sensor_cache[0] != key:
                self._sensor_cache = (key, original_read(model, data).copy())
            return self._sensor_cache[1].copy()

        sensor.read = read_once

    def __getattr__(self, name):
        return getattr(self.env, name)

    @property
    def elapsed_steps(self):
        return self.env.elapsed_steps

    @elapsed_steps.setter
    def elapsed_steps(self, value):
        self.env.elapsed_steps = value

    def _before(self):
        import time

        started = time.monotonic()
        env = self.env
        observation = env._get_observation()
        tactile = np.asarray(observation["tactile"], dtype=np.float32)
        if tactile.shape != (712,):
            raise ValueError(f"Actual sensor must return 712 taxels, got {tactile.shape}")
        poses = []
        for body in self.body_ids:
            poses.append(np.concatenate((env.data.xpos[body], env.data.xquat[body])))
        if self.target_site >= 0:
            # The target is a real model site, not a second dynamic object.
            quaternion = np.empty(4)
            mujoco.mju_mat2Quat(quaternion, env.data.site_xmat[self.target_site])
            poses.append(np.concatenate((env.data.site_xpos[self.target_site], quaternion)))
        row = {
            "timestamp": float(env.data.time),
            "state": real_observable_state(env).copy(),
            "sensor_tactile": tactile.copy(),
            "object_state": np.asarray(poses, dtype=np.float32),
            "pre_qpos": env.data.qpos.astype(np.float32).copy(),
            "pre_qvel": env.data.qvel.astype(np.float32).copy(),
        }
        spec = mujoco.mjtState.mjSTATE_INTEGRATION
        row['pre_integration_state'] = np.empty(mujoco.mj_stateSize(env.model, spec), dtype=np.float64)
        mujoco.mj_getState(env.model, env.data, row['pre_integration_state'], spec)
        self.profile.add('state_tactile_capture', time.monotonic() - started)
        if self.video is not None and self.capture_video and self.video.should_capture:
            if self.renderer is None:
                width, height = self.video.capture['width'], self.video.capture['height']
                env.model.vis.global_.offwidth = width
                env.model.vis.global_.offheight = height
                self.renderer = self.profile.call('renderer_create', mujoco.Renderer, env.model, height=height, width=width)
            with self.profile.measure('rgb_render'):
                self.renderer.update_scene(env.data, camera="agentview")
                row["_rgb"] = self.renderer.render().copy()
        return row

    def close(self):
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None
        self.env.close()

    def _after(self, row):
        env = self.env
        row["next_timestamp"] = float(env.data.time)
        ids = np.concatenate((env.controller.arm_controller.actuator_ids,
                              env.controller.hand_controller.actuator_ids))
        row["position_action"] = env.data.ctrl[ids].astype(np.float32).copy()
        row['applied_ctrl64'] = env.data.ctrl.copy()
        self.pending = row

    def step(self, action):
        row = self._before()
        env = self.env
        result = self.profile.call('control_physics_step', env.step, action)
        self._after(row)
        return result

    def step_physics(self, *args, **kwargs):
        # Legacy PickPlace re-executes controls directly rather than env.step.
        row = self.prepared if self.prepared is not None else self._before()
        self.prepared = None
        result = self.profile.call('control_physics_step', self.env.step_physics, *args, **kwargs)
        self._after(row)
        return result

    def reset(self, *args, **kwargs):
        return self.profile.call('environment_reset', self.env.reset, *args, **kwargs)

    def prepare_recorded_control(self):
        """Called before legacy low-level code overwrites the command buffer."""
        if self.prepared is not None:
            raise RuntimeError("Previous prepared control was not executed")
        self.prepared = self._before()


def recorder_type(legacy):
    class OnlineRecorder(legacy):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.online = {}

        def append(self, env, observation, action, stage, reward, success):
            row = env.pending
            if row is None:
                raise RuntimeError("No freshly executed control step to record")
            row["ik_action"] = np.asarray(action, dtype=np.float32).copy()
            rgb = row.pop("_rgb", None)
            if env.video is not None and env.capture_video:
                env.video.record_control(rgb)
            super().append(env, observation, action, stage, reward, success)
            for key, value in row.items():
                self.online.setdefault(key, []).append(value)
            env.pending = None

        def arrays(self):
            result = super().arrays()
            result.update({key: np.asarray(value) for key, value in self.online.items()})
            return result

    return OnlineRecorder
