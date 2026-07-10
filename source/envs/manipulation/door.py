"""Articulated door-opening task implemented directly with MuJoCo MjSpec."""
from __future__ import annotations

from gymnasium import spaces
import mujoco
import numpy as np

from source.envs.core.registry import register_task
from source.envs.core.tasks import RobotTask, TaskStepResult
from source.envs.manipulation.arenas import TableArena
from source.assets import asset_path


@register_task("door")
class DoorTask(RobotTask):
    def __init__(self, *, ee_site_name="right_hand_site", use_latch=True,
                 reward_scale=1.0, reward_shaping=False, terminate_on_success=False):
        self.ee_site_name = ee_site_name
        self.use_latch = use_latch
        self.reward_scale = reward_scale
        self.reward_shaping = reward_shaping
        self.terminate_on_success = terminate_on_success
        self.arena = TableArena()
        self._ids = {}

    @property
    def name(self):
        return "door"

    @property
    def observation_space(self):
        result = {
            "door_pos": spaces.Box(-np.inf, np.inf, (3,), dtype=np.float32),
            "handle_pos": spaces.Box(-np.inf, np.inf, (3,), dtype=np.float32),
            "hinge_qpos": spaces.Box(-np.inf, np.inf, (1,), dtype=np.float32),
            "handle_to_gripper_pos": spaces.Box(-np.inf, np.inf, (3,), dtype=np.float32),
        }
        if self.use_latch:
            result["handle_qpos"] = spaces.Box(-np.inf, np.inf, (1,), dtype=np.float32)
        return result

    def augment_spec(self, spec):
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
        # The unnamed outer body only contains robosuite metadata sites. The
        # named object body is the movable placement root used by DoorObject.
        object_body.pos = [0.63, 0.0, 1.1]
        attach_frame = spec.worldbody.add_frame()
        attach_frame.attach_body(object_body, prefix="Door_", suffix="")

    def bind(self, model):
        def ident(kind, name):
            value = mujoco.mj_name2id(model, kind, name)
            if value < 0: raise ValueError(f"Missing compiled door element {name!r}")
            return int(value)
        hinge_id = ident(mujoco.mjtObj.mjOBJ_JOINT, "Door_hinge")
        hinge_dof = int(model.jnt_dofadr[hinge_id])
        # Match DoorObject(friction=0.0, damping=0.1) from robosuite's task.
        model.dof_frictionloss[hinge_dof] = 0.0
        model.dof_damping[hinge_dof] = 0.1
        self._ids = {
            "root": ident(mujoco.mjtObj.mjOBJ_BODY, "Door_object"),
            "frame": ident(mujoco.mjtObj.mjOBJ_BODY, "Door_frame"),
            "door": ident(mujoco.mjtObj.mjOBJ_BODY, "Door_door"),
            "latch": ident(mujoco.mjtObj.mjOBJ_BODY, "Door_latch"),
            "handle_site": ident(mujoco.mjtObj.mjOBJ_SITE, "Door_handle"),
            "hinge_adr": int(model.jnt_qposadr[hinge_id]),
            "ee_site": mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, self.ee_site_name),
        }
        if self.use_latch:
            jid = ident(mujoco.mjtObj.mjOBJ_JOINT, "Door_latch_joint")
            self._ids["handle_adr"] = int(model.jnt_qposadr[jid])

    def reset(self, model, data, *, rng, options):
        _ = options
        # Randomize the complete frame pose, matching robosuite's narrow distribution.
        root = self._ids["root"]
        model.body_pos[root] = [
            self.arena.table_offset[0] + rng.uniform(0.07, 0.09),
            self.arena.table_offset[1] + rng.uniform(-0.01, 0.01),
            self.arena.table_top_z + 0.30,
        ]
        yaw = rng.uniform(-np.pi / 2 - 0.25, -np.pi / 2)
        model.body_quat[root] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
        data.qpos[self._ids["hinge_adr"]] = 0
        if self.use_latch: data.qpos[self._ids["handle_adr"]] = 0
        return {"task": self.name, "use_latch": self.use_latch}

    def get_observation(self, model, data):
        _ = model
        handle = data.site_xpos[self._ids["handle_site"]].astype(np.float32).copy()
        ee_id = self._ids["ee_site"]
        ee = np.zeros(3) if ee_id < 0 else data.site_xpos[ee_id]
        result = {
            "door_pos": data.xpos[self._ids["door"]].astype(np.float32).copy(),
            "handle_pos": handle,
            "hinge_qpos": np.asarray([data.qpos[self._ids["hinge_adr"]]], np.float32),
            "handle_to_gripper_pos": (handle - ee).astype(np.float32),
        }
        if self.use_latch:
            result["handle_qpos"] = np.asarray([data.qpos[self._ids["handle_adr"]]], np.float32)
        return result

    def evaluate(self, obs, action, model, data):
        _ = action, model
        success = bool(data.qpos[self._ids["hinge_adr"]] > 0.3)
        reward = 1.0 if success else 0.0
        info = {}
        if not success and self.reward_shaping:
            dist = float(np.linalg.norm(obs["handle_to_gripper_pos"]))
            reaching = 0.25 * (1 - np.tanh(10 * dist)); reward += reaching
            info["reward_reaching"] = reaching
            if self.use_latch:
                rotating = float(np.clip(0.25 * abs(obs["handle_qpos"][0] / (0.5 * np.pi)), 0, 0.25))
                reward += rotating; info["reward_rotating"] = rotating
        if self.reward_scale is not None: reward *= self.reward_scale
        return TaskStepResult(float(reward), success, self.terminate_on_success and success,
                              {"task_success": success, **info})
