import numpy as np
import pytest
from types import SimpleNamespace

from source.data.production_recorder import OnlineEnvironment, validate_online_arrays


def test_current_stage_contract_replaces_incomplete_legacy_prefix_without_changing_arrays():
    from source.data.production_recorder import attach_stage_encoding, validate_stage_encoding

    stages = np.array([1, 6, 8, 9, 10, 11, 12, 13], dtype=np.int16)
    episode = SimpleNamespace(arrays={'stage': stages}, metadata={
        'stage_codes': {'transit': 1, 'lift': 6}, 'online_contract': {}})
    result = attach_stage_encoding(episode)
    assert result['checked']
    assert episode.arrays['stage'] is stages
    assert episode.metadata['stage_codes']['transport'] == 8
    assert episode.metadata['stage_codes']['task_verify'] == 12
    assert validate_stage_encoding({'stage': stages}, {'stage_codes': {'old': 1}})['checked'] is False
    valid = dict(episode.metadata['stage_codes'])
    episode.metadata['stage_codes']['transit'], episode.metadata['stage_codes']['lift'] = 6, 1
    with pytest.raises(ValueError, match='versioned contract'):
        validate_stage_encoding(episode.arrays, episode.metadata)
    episode.metadata['stage_codes'] = valid
    episode.metadata['stage_codes'].pop('release')
    with pytest.raises(ValueError, match='absent'):
        validate_stage_encoding(episode.arrays, episode.metadata)


def test_stage_contract_rejects_unknown_version_ambiguous_codes_and_noninteger_stages():
    from source.data.production_recorder import validate_stage_encoding, STAGE_ENCODING

    arrays = {'stage': np.array([1], dtype=np.int16)}
    metadata = {'online_contract': {'stage_encoding': 'unknown'}, 'stage_codes': {'transit': 1}}
    with pytest.raises(ValueError, match='Unsupported'):
        validate_stage_encoding(arrays, metadata)
    metadata['online_contract']['stage_encoding'] = STAGE_ENCODING
    metadata['stage_codes']['alias'] = 1
    with pytest.raises(ValueError, match='unique'):
        validate_stage_encoding(arrays, metadata)
    metadata['stage_codes'].pop('alias')
    with pytest.raises(ValueError, match='integer array'):
        validate_stage_encoding({'stage': arrays['stage'].astype(float)}, metadata)


def sample():
    return {"timestamp": np.array([1., 1.05]), "next_timestamp": np.array([1.05, 1.1]),
            "state": np.zeros((2, 39), np.float32),
            "position_action": np.ones((2, 13), np.float32),
            "ik_action": np.zeros((2, 13), np.float32),
            "sensor_tactile": np.zeros((2, 712), np.float32),
            "object_state": np.tile(np.array([0, 0, 0, 1, 0, 0, 0], np.float32), (2, 2, 1))}


def test_uncontacted_sensor_is_valid_and_duration_includes_last_step():
    assert validate_online_arrays(sample())["duration_seconds"] == pytest.approx(.1)


def test_previous_command_causality():
    arrays = sample()
    with pytest.raises(ValueError, match="previous applied action"):
        validate_online_arrays(arrays, require_causal=True)
    arrays["state"][1, -13:] = arrays["position_action"][0]
    assert validate_online_arrays(arrays, require_causal=True)["command_causality_checked"]


def test_v4_requires_complete_float64_replay_pair_and_preserves_legacy_schema():
    arrays = sample()
    assert validate_online_arrays(arrays)['schema'] == 'five-task-online-v3'
    arrays['pre_integration_state'] = np.zeros((2, 10), dtype=np.float64)
    with pytest.raises(ValueError, match='Incomplete exact'):
        validate_online_arrays(arrays)
    arrays['applied_ctrl64'] = np.ones((2, 13), dtype=np.float64)
    assert validate_online_arrays(arrays)['schema'] == 'five-task-online-v4'
    arrays['applied_ctrl64'] = arrays['applied_ctrl64'].astype(np.float32)
    with pytest.raises(ValueError, match='float64'):
        validate_online_arrays(arrays)


def test_unrecorded_state_jump_is_rejected_even_if_commands_are_causal():
    arrays = sample()
    arrays['state'][1, -13:] = arrays['position_action'][0]
    for key in ('pre_qpos', 'pre_qvel', 'qpos', 'qvel'):
        arrays[key] = np.zeros((2, 26), np.float32)
    assert validate_online_arrays(arrays, require_causal=True)['physics_state_continuity_checked']
    arrays['pre_qpos'][1, 0] = .1
    with pytest.raises(ValueError, match='physical state discontinuity'):
        validate_online_arrays(arrays, require_causal=True)


@pytest.mark.parametrize("fault", ["missing_sensor", "gap", "nan", "quaternion"])
def test_invalid_capture_rejected(fault):
    a = sample()
    if fault == "missing_sensor":
        a["sensor_tactile"] = np.empty((2, 0), np.float32)
    elif fault == "gap":
        a["timestamp"][1] += .05
    elif fault == "nan":
        a["state"][0, 0] = np.nan
    else:
        a["object_state"][0, 0, 3] = 0
    with pytest.raises(ValueError):
        validate_online_arrays(a)


def test_capture_is_before_action_and_low_level_command_is_after_controller(monkeypatch):
    import source.data.production_recorder as module

    monkeypatch.setattr(module.mujoco, "mj_name2id", lambda *args: -1)
    monkeypatch.setattr(module.mujoco, 'mj_stateSize', lambda *_: 1)
    monkeypatch.setattr(module.mujoco, 'mj_getState', lambda m, d, state, spec: state.__setitem__(0, d.time))
    monkeypatch.setattr(module, "real_observable_state", lambda env: env.state)
    data = SimpleNamespace(time=1., qpos=np.zeros(26), qvel=np.zeros(25),
                           xpos=np.zeros((1, 3)), xquat=np.array([[1., 0, 0, 0]]),
                           ctrl=np.zeros(13))
    env = SimpleNamespace(data=data, model=None, state=np.zeros(39, np.float32),
                          task=SimpleNamespace(bindings=SimpleNamespace(objects={
                              "object": SimpleNamespace(body_id=0)})),
                          tactile_sensor=SimpleNamespace(
                              read=lambda *args: np.ones(712, np.float32),
                              read_raw=lambda *args: np.ones(712, np.float32)),
                          controller=SimpleNamespace(
                              arm_controller=SimpleNamespace(actuator_ids=np.arange(7)),
                              hand_controller=SimpleNamespace(actuator_ids=np.arange(7, 13))))
    reads = []

    def read(*args):
        reads.append(data.time)
        return np.ones(712, np.float32)

    env.tactile_sensor.read = read
    env._get_observation = lambda: {"tactile": env.tactile_sensor.read(None, data)}

    def step(action):
        data.time += .05
        env.state[:] = 3
        data.ctrl[:] = action
        return "result"

    env.step = step
    env.step_physics = lambda *args, **kwargs: step(data.ctrl.copy())
    wrapper = OnlineEnvironment(env)
    assert wrapper.step(np.ones(13)) == "result"
    assert np.all(wrapper.pending["state"] == 0)
    assert wrapper.pending["timestamp"] == 1
    assert np.all(wrapper.pending["position_action"] == 1)
    # Repeated consumers of the same physical state share one processed sample.
    env._get_observation()
    env._get_observation()
    assert reads == [1., 1.05]
    env.elapsed_steps = 0
    wrapper.elapsed_steps += 1
    assert env.elapsed_steps == 1
    # Direct-control callers must capture before assigning the new ctrl buffer.
    env.state[-13:] = data.ctrl
    wrapper.prepare_recorded_control()
    data.ctrl[:] = 7
    env.state[-13:] = 7
    wrapper.step_physics(50, control_updates=1)
    assert np.all(wrapper.pending["state"][-13:] == 1)
    assert np.all(wrapper.pending["position_action"] == 7)
