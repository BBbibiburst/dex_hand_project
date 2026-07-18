"""Fast headless reset/step coverage for core environments."""

from pathlib import Path

import numpy as np
import pytest

from source.envs.manipulation import make_manipulation_env, registered_tasks
from source.envs.manipulation.object_catalog import MANIFEST_PATH
from source.envs.rl_env import RobotGymEnv, make_env


PROFILES = (
    "configs/robot_profiles/rm75b_dex_hand.json",
    "configs/robot_profiles/rm75b_pika_gripper.json",
)


def assert_observation(env: RobotGymEnv, observation: dict) -> None:
    assert env.observation_space.contains(observation)
    for value in observation.values():
        array = np.asarray(value)
        if np.issubdtype(array.dtype, np.number):
            assert np.all(np.isfinite(array))


def reset_and_step_once(env: RobotGymEnv) -> None:
    observation, _ = env.reset(seed=0)
    assert_observation(env, observation)
    action = env.controller.current_action(env.model, env.data).astype(np.float32)
    assert env.action_space.contains(action)
    observation, reward, _, _, info = env.step(action)
    assert_observation(env, observation)
    assert np.isfinite(reward)
    assert isinstance(info, dict)


@pytest.mark.parametrize("profile", PROFILES, ids=lambda value: Path(value).stem)
@pytest.mark.parametrize("control_mode", ("position", "ik"))
def test_noop_profile_can_reset_and_step(
    profile: str,
    control_mode: str,
) -> None:
    env = make_env(
        robot_config_path=profile,
        control_mode=control_mode,
        enable_tactile_sensors=False,
        render_mode=None,
    )
    try:
        reset_and_step_once(env)
    finally:
        env.close()


@pytest.mark.parametrize("profile", PROFILES, ids=lambda value: Path(value).stem)
@pytest.mark.parametrize("task_name", registered_tasks())
def test_builtin_task_can_reset_and_step(task_name: str, profile: str) -> None:
    if task_name != "nut_assembly" and not MANIFEST_PATH.is_file():
        pytest.skip("optional ManiSkill object assets are not installed")
    env = make_manipulation_env(
        task_name,
        robot_config_path=profile,
        control_mode="ik",
        enable_tactile_sensors=False,
        render_mode=None,
    )
    try:
        reset_and_step_once(env)
    finally:
        env.close()
