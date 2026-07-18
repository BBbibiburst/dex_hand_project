"""Built-in manipulation task registration checks."""

import pytest

from source.envs.manipulation import make_task, registered_tasks
from source.envs.manipulation.object_catalog import MANIFEST_PATH


def test_builtin_tasks_are_registered() -> None:
    expected = {
        "lift",
        "stack",
        "pick_place",
        "nut_assembly",
        "push",
    }
    assert expected <= set(registered_tasks())


@pytest.mark.parametrize("task_name", registered_tasks())
def test_registered_task_can_be_created(task_name: str) -> None:
    if task_name != "nut_assembly" and not MANIFEST_PATH.is_file():
        pytest.skip("optional ManiSkill object assets are not installed")
    task = make_task(task_name)
    assert task.name == task_name
