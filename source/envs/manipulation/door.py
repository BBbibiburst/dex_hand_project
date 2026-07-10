"""Articulated door-opening task implemented directly with MuJoCo MjSpec."""

from __future__ import annotations

from dataclasses import dataclass

from gymnasium import spaces
import mujoco
import numpy as np

from source.assets import asset_path
from source.envs.core.registry import register_task
from source.envs.core.tasks import RobotTask, TaskStepResult
from source.envs.manipulation.arenas import TableArena


@dataclass(frozen=True)
class DoorBindings:
    root_body_id: int
    door_body_id: int
    handle_site_id: int
    hinge_qpos_adr: int
    ee_site_id: int | None
    latch_qpos_adr: int | None


@register_task("door")
class DoorTask(RobotTask):
    success_reward = 1.0

    def __init__(
        self,
        *,
        ee_site_name: str = "right_hand_site",
        use_latch: bool = True,
        reward_scale: float | None = 1.0,
        reward_shaping: bool = False,
        terminate_on_success: bool = False,
    ) -> None:
        self.ee_site_name = ee_site_name
        self.use_latch = use_latch
        self.reward_scale = reward_scale
        self.reward_shaping = reward_shaping
        self.terminate_on_success = terminate_on_success
        self.arena = TableArena()
        self._bindings: DoorBindings | None = None

    @property
    def name(self) -> str:
        return "door"

    @property
    def observation_space(self):
        result = {
            "door_pos": spaces.Box(-np.inf, np.inf, (3,), dtype=np.float32),
            "handle_pos": spaces.Box(-np.inf, np.inf, (3,), dtype=np.float32),
            "hinge_qpos": spaces.Box(-np.inf, np.inf, (1,), dtype=np.float32),
            "handle_to_gripper_pos": spaces.Box(
                -np.inf, np.inf, (3,), dtype=np.float32
            ),
        }
        if self.use_latch:
            result["handle_qpos"] = spaces.Box(
                -np.inf, np.inf, (1,), dtype=np.float32
            )
        return result

    def augment_spec(self, spec) -> None:
        self.arena.augment_spec(spec)
        filename = "door_lock.xml" if self.use_latch else "door.xml"
        xml_path = asset_path("objects", filename)
        if not xml_path.exists():
            raise FileNotFoundError(f"Door object XML not found: {xml_path}")

        door_spec = mujoco.MjSpec.from_file(str(xml_path))
        wrapper = door_spec.worldbody.first_body()
        if wrapper is None or wrapper.first_body() is None:
            raise ValueError(f"Door XML has no object body: {xml_path}")

        object_body = wrapper.first_body()
        object_body.pos = [0.63, 0.0, 1.1]
        attach_frame = spec.worldbody.add_frame()
        attach_frame.attach_body(object_body, prefix="Door_", suffix="")

    def bind(self, model: mujoco.MjModel) -> None:
        hinge_id = self._id_or_raise(
            model, mujoco.mjtObj.mjOBJ_JOINT, "Door_hinge"
        )
        hinge_dof_adr = int(model.jnt_dofadr[hinge_id])
        model.dof_frictionloss[hinge_dof_adr] = 0.0
        model.dof_damping[hinge_dof_adr] = 0.1

        ee_site_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_SITE, self.ee_site_name
        )
        latch_qpos_adr = None
        if self.use_latch:
            latch_joint_id = self._id_or_raise(
                model, mujoco.mjtObj.mjOBJ_JOINT, "Door_latch_joint"
            )
            latch_qpos_adr = int(model.jnt_qposadr[latch_joint_id])

        self._bindings = DoorBindings(
            root_body_id=self._id_or_raise(
                model, mujoco.mjtObj.mjOBJ_BODY, "Door_object"
            ),
            door_body_id=self._id_or_raise(
                model, mujoco.mjtObj.mjOBJ_BODY, "Door_door"
            ),
            handle_site_id=self._id_or_raise(
                model, mujoco.mjtObj.mjOBJ_SITE, "Door_handle"
            ),
            hinge_qpos_adr=int(model.jnt_qposadr[hinge_id]),
            ee_site_id=None if ee_site_id < 0 else int(ee_site_id),
            latch_qpos_adr=latch_qpos_adr,
        )

    def reset(self, model, data, *, rng, options):
        _ = options
        bindings = self._require_bindings()
        model.body_pos[bindings.root_body_id] = [
            self.arena.table_offset[0] + rng.uniform(0.07, 0.09),
            self.arena.table_offset[1] + rng.uniform(-0.01, 0.01),
            self.arena.table_top_z + 0.30,
        ]
        yaw = rng.uniform(-np.pi / 2 - 0.25, -np.pi / 2)
        model.body_quat[bindings.root_body_id] = [
            np.cos(yaw / 2),
            0.0,
            0.0,
            np.sin(yaw / 2),
        ]
        data.qpos[bindings.hinge_qpos_adr] = 0.0
        if bindings.latch_qpos_adr is not None:
            data.qpos[bindings.latch_qpos_adr] = 0.0
        return {"task": self.name, "use_latch": self.use_latch}

    def get_observation(self, model, data):
        _ = model
        bindings = self._require_bindings()
        handle = data.site_xpos[bindings.handle_site_id].astype(np.float32).copy()
        ee = (
            np.zeros(3, dtype=np.float32)
            if bindings.ee_site_id is None
            else data.site_xpos[bindings.ee_site_id]
        )
        result = {
            "door_pos": data.xpos[bindings.door_body_id].astype(np.float32).copy(),
            "handle_pos": handle,
            "hinge_qpos": np.asarray(
                [data.qpos[bindings.hinge_qpos_adr]], dtype=np.float32
            ),
            "handle_to_gripper_pos": (handle - ee).astype(np.float32),
        }
        if bindings.latch_qpos_adr is not None:
            result["handle_qpos"] = np.asarray(
                [data.qpos[bindings.latch_qpos_adr]], dtype=np.float32
            )
        return result

    def evaluate(self, obs, action, model, data):
        _ = action, model
        bindings = self._require_bindings()
        success = bool(data.qpos[bindings.hinge_qpos_adr] > 0.3)
        reward = self.success_reward if success else 0.0
        info = {}

        if not success and self.reward_shaping:
            distance = float(np.linalg.norm(obs["handle_to_gripper_pos"]))
            reaching = 0.25 * (1.0 - np.tanh(10.0 * distance))
            reward += reaching
            info["reward_reaching"] = reaching

            if bindings.latch_qpos_adr is not None:
                rotating = float(
                    np.clip(
                        0.25 * abs(obs["handle_qpos"][0] / (0.5 * np.pi)),
                        0.0,
                        0.25,
                    )
                )
                reward += rotating
                info["reward_rotating"] = rotating

        return TaskStepResult(
            self._scale_reward(float(reward)),
            success,
            self.terminate_on_success and success,
            {"task_success": success, **info},
        )

    def _scale_reward(self, reward: float) -> float:
        if self.reward_scale is None:
            return reward
        return reward * self.reward_scale / self.success_reward

    def _require_bindings(self) -> DoorBindings:
        if self._bindings is None:
            raise RuntimeError("DoorTask.bind() must be called first.")
        return self._bindings

    @staticmethod
    def _id_or_raise(model: mujoco.MjModel, kind, name: str) -> int:
        value = mujoco.mj_name2id(model, kind, name)
        if value < 0:
            raise ValueError(f"Missing compiled door element {name!r}")
        return int(value)
