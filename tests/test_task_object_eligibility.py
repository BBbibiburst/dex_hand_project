from source.envs.manipulation.object_catalog import (
    partition_task_objects,
    task_object_eligibility,
)


def test_stack_eligibility_uses_declared_stable_object_set() -> None:
    assert task_object_eligibility("stack", "ycb:008_pudding_box") == (True, None)
    accepted, reason = task_object_eligibility("stack", "ycb:013_apple")
    assert not accepted
    assert "stable-support" in str(reason)


def test_push_eligibility_excludes_ball_from_four_finger_domain() -> None:
    assert task_object_eligibility("push", "ycb:008_pudding_box") == (True, None)
    accepted, reason = task_object_eligibility("push", "ycb:055_baseball")
    assert not accepted
    assert "four-finger" in str(reason)


def test_partition_preserves_order_and_exclusion_reason() -> None:
    eligible, excluded = partition_task_objects(
        "stack", ["ycb:013_apple", "ycb:008_pudding_box"]
    )
    assert eligible == ("ycb:008_pudding_box",)
    assert excluded[0]["object_id"] == "ycb:013_apple"
    assert excluded[0]["reason"]
