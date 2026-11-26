"""Restore speculative motion probes without hidden physics or sensor transitions."""
import copy

import mujoco
import numpy as np


class SimulationCheckpoint:
    """A probe is not part of the recorded trajectory, including solver history."""
    def __init__(self, env):
        self.env = env
        self.spec = mujoco.mjtState.mjSTATE_INTEGRATION
        self.state = np.empty(mujoco.mj_stateSize(env.model, self.spec), dtype=np.float64)
        mujoco.mj_getState(env.model, env.data, self.state, self.spec)
        self.elapsed = env.elapsed_steps
        arm = env.controller.arm_controller
        self.arm = arm
        self.arm_state = {name: copy.deepcopy(getattr(arm, name))
                          for name in ('_prev_target_q', '_filtered_velocity', '_prev_ee_target')
                          if hasattr(arm, name)}
        self.processor = getattr(env.tactile_sensor, 'signal_processor', None)
        self.filtered = copy.deepcopy(self.processor._filtered) if self.processor else None
        self.rng = copy.deepcopy(self.processor._rng.bit_generator.state) if self.processor else None
        # Only attributes owned by the wrapper; do not follow __getattr__ proxies.
        self.wrapper = {name: copy.deepcopy(env.__dict__[name])
                        for name in ('pending', 'prepared', '_sensor_cache') if name in env.__dict__}

    def restore(self):
        env = self.env
        mujoco.mj_setState(env.model, env.data, self.state, self.spec)
        mujoco.mj_forward(env.model, env.data)
        # mj_forward recomputes derived fields; it must not alter the checkpoint's
        # integration state used by the next actual physics step.
        mujoco.mj_setState(env.model, env.data, self.state, self.spec)
        env.elapsed_steps = self.elapsed
        for name, value in self.arm_state.items():
            setattr(self.arm, name, copy.deepcopy(value))
        if self.processor:
            self.processor._filtered = copy.deepcopy(self.filtered)
            self.processor._rng.bit_generator.state = copy.deepcopy(self.rng)
        for name, value in self.wrapper.items():
            setattr(env, name, copy.deepcopy(value))
