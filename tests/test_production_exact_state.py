import mujoco
import numpy as np


def test_full_integration_restore_reproduces_recorded_double_precision_controls():
    model = mujoco.MjModel.from_xml_string('''<mujoco><option timestep="0.001" gravity="0 0 0"/>
      <worldbody><body><joint name="hinge" damping=".1"/><geom size=".05"/></body></worldbody>
      <actuator><position joint="hinge" kp="10"/></actuator></mujoco>''')
    original = mujoco.MjData(model)
    original.qpos[0] = .123456789123
    original.qvel[0] = .234567891234
    for _ in range(10):
        mujoco.mj_step(model, original)
    spec = mujoco.mjtState.mjSTATE_INTEGRATION
    saved = np.empty(mujoco.mj_stateSize(model, spec), np.float64)
    mujoco.mj_getState(model, original, saved, spec)
    replay = mujoco.MjData(model)
    mujoco.mj_setState(model, replay, saved, spec)
    mujoco.mj_forward(model, replay)
    for target in np.linspace(.198765432198, .298765432198, 20):
        original.ctrl[0] = replay.ctrl[0] = target
        for _ in range(5):
            mujoco.mj_step(model, original)
            mujoco.mj_step(model, replay)
        np.testing.assert_allclose(replay.qpos, original.qpos, rtol=0, atol=1e-14)
        np.testing.assert_allclose(replay.qvel, original.qvel, rtol=0, atol=1e-14)
