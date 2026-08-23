"""Physics-level regressions for the adaptive underactuated Dex Hand."""

from __future__ import annotations

import mujoco
import numpy as np
import pytest

from source.control.end_effectors import EndEffectorPositionController
from source.envs.manipulation.objects import _configure_object_collision
from source.robots.hands.dex_hand import DEX_HAND


def _model() -> mujoco.MjModel:
    return mujoco.MjModel.from_xml_path(str(DEX_HAND.xml_path.resolve()))


def _joint_position(model: mujoco.MjModel, data: mujoco.MjData, name: str) -> float:
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    return float(data.qpos[model.jnt_qposadr[joint_id]])


def _settle_finger(*, proximal_limit: float | None = None) -> tuple[np.ndarray, float]:
    model = _model()
    if proximal_limit is not None:
        joint_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            "finger_first_0_j",
        )
        model.jnt_range[joint_id, 1] = proximal_limit

    data = mujoco.MjData(model)
    actuator_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_ACTUATOR,
        "act_push_0_j",
    )
    data.ctrl[actuator_id] = 0.01
    for _ in range(4000):
        mujoco.mj_step(model, data)

    positions = np.asarray(
        [
            _joint_position(model, data, "finger_first_0_j"),
            _joint_position(model, data, "finger_second_0_j"),
        ]
    )
    return positions, float(data.actuator_force[actuator_id])


def test_dex_hand_is_six_drive_and_genuinely_underactuated() -> None:
    model = _model()

    assert model.nq == model.nv == 12
    assert model.nu == 6
    assert model.ntendon == 6
    assert model.neq == 0
    assert np.all(model.actuator_trntype == mujoco.mjtTrn.mjTRN_TENDON)


def test_only_tactile_skin_meshes_participate_in_collision() -> None:
    model = _model()
    collision_names = {
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
        for geom_id in range(model.ngeom)
        if model.geom_contype[geom_id] or model.geom_conaffinity[geom_id]
    }

    expected = {"skin_palm_p"}
    expected.update(f"skin_{digit}_{segment}_p" for digit in range(5) for segment in range(3))
    assert collision_names == expected
    for geom_id in range(model.ngeom):
        if mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) in expected:
            assert model.geom_contype[geom_id] == 2
            assert model.geom_conaffinity[geom_id] == 3
            assert model.geom_condim[geom_id] == 4
            assert model.geom_priority[geom_id] == 1
            np.testing.assert_allclose(model.geom_solref[geom_id], [0.01, 1.0])


def test_every_skin_geom_is_collision_compatible_with_every_other_skin() -> None:
    model = _model()
    skin_ids = [
        geom_id
        for geom_id in range(model.ngeom)
        if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or "").startswith(
            "skin_"
        )
    ]

    for offset, first in enumerate(skin_ids):
        for second in skin_ids[offset + 1 :]:
            assert model.geom_contype[first] & model.geom_conaffinity[second]
            assert model.geom_contype[second] & model.geom_conaffinity[first]


def test_task_object_contact_does_not_inherit_robot_geom_defaults() -> None:
    spec = mujoco.MjSpec()
    geom = spec.worldbody.add_geom()
    geom.name = "object_collision"
    geom.type = mujoco.mjtGeom.mjGEOM_BOX
    geom.size = [0.02, 0.03, 0.01]
    geom.density = 500.0
    geom.solref = [0.2, 0.3]
    geom.priority = 7

    _configure_object_collision(
        geom,
        friction=(1.0, 0.005, 0.0001),
        condim=4,
    )
    model = spec.compile()
    geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "object_collision")

    assert model.geom_condim[geom_id] == 4
    assert model.geom_priority[geom_id] == 0
    assert model.geom_solmix[geom_id] == pytest.approx(1.0)
    np.testing.assert_allclose(model.geom_friction[geom_id], [1.0, 0.005, 0.0001])
    np.testing.assert_allclose(model.geom_solref[geom_id], [0.001, 2.0])
    np.testing.assert_allclose(
        model.geom_solimp[geom_id],
        [0.9, 0.95, 0.001, 0.5, 2.0],
    )


def test_finger_redistributes_tendon_travel_after_proximal_blockage() -> None:
    free, free_force = _settle_finger()
    blocked, blocked_force = _settle_finger(proximal_limit=0.25)

    assert 1.25 < free[0] < 1.50
    assert 1.40 < free[1] < 1.65
    assert blocked[0] < free[0] - 0.35
    assert blocked[1] > free[1] + 0.20
    assert abs(free_force) < 16.0
    assert abs(blocked_force) <= 20.0


def test_maximum_index_thumb_flexion_releases_back_to_zero() -> None:
    model = _model()
    data = mujoco.MjData(model)
    actuator_ids = np.asarray(
        [
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            for name in (
                "act_push_3_j",
                "thumb_rotate_act_push_j",
                "thumb_grasp_act_push_j",
            )
        ]
    )
    data.ctrl[actuator_ids] = np.asarray([0.01, 0.004, 0.01])
    for _ in range(4000):
        mujoco.mj_step(model, data)

    assert model.npair == 22
    assert data.ncon >= 1
    assert min(data.contact[index].dist for index in range(data.ncon)) > -0.0005

    data.ctrl[actuator_ids] = 0.0
    for _ in range(2500):
        mujoco.mj_step(model, data)

    assert data.ncon == 0
    assert np.max(np.abs(data.actuator_length[actuator_ids])) < 5e-5
    assert np.max(np.abs(data.qvel)) < 1e-3


def test_maximum_middle_thumb_flexion_generates_bounded_contact() -> None:
    model = _model()
    data = mujoco.MjData(model)
    actuator_ids = np.asarray(
        [
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            for name in (
                "act_push_2_j",
                "thumb_rotate_act_push_j",
                "thumb_grasp_act_push_j",
            )
        ]
    )
    data.ctrl[actuator_ids] = np.asarray([0.01, 0.004, 0.01])
    for _ in range(4000):
        mujoco.mj_step(model, data)

    middle_thumb_contacts = []
    for index in range(data.ncon):
        contact = data.contact[index]
        names = {
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(contact.geom1)),
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(contact.geom2)),
        }
        if any(name.startswith("skin_2_") for name in names) and any(
            name.startswith("skin_4_") for name in names
        ):
            middle_thumb_contacts.append(contact)

    assert middle_thumb_contacts
    assert min(contact.dist for contact in middle_thumb_contacts) > -0.0005


def test_position_controller_reads_tendon_length_instead_of_joint_qpos() -> None:
    model = _model()
    data = mujoco.MjData(model)
    controller = EndEffectorPositionController(
        hand_descriptor=DEX_HAND,
        hand_prefix="",
    )
    controller.bind(model, data)
    data.ctrl[:] = np.asarray([0.008, 0.006, 0.004, 0.002, 0.003, 0.007])
    for _ in range(2000):
        mujoco.mj_step(model, data)

    np.testing.assert_allclose(
        controller.current_position(model, data),
        data.actuator_length,
        atol=1e-7,
    )
    with pytest.raises(RuntimeError, match="non-joint actuator transmissions"):
        _ = controller.qpos_addrs
