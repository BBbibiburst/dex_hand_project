# -*- coding: utf-8 -*-
"""
机械臂与灵巧手模型装配工具。

本模块负责加载 RM75B 机械臂、底座和灵巧手 XML，并把手模型挂载到
机械臂的指定 body（默认 ``right_hand``）下。公开接口刻意区分
``MjSpec`` 构建和模型编译，方便调用方在 ``spec.compile()`` 前继续添加
相机、物体、任务逻辑或传感器。
"""

import traceback
from pathlib import Path
from typing import Optional, Tuple, Union

import mujoco
from mujoco import viewer
import numpy as np
from scipy.spatial.transform import Rotation as R


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

DEFAULT_ARM_PATH = PROJECT_ROOT / "assets" / "robots" / "rm75b" / "rm75b.xml"
DEFAULT_HAND_PATH = PROJECT_ROOT / "assets" / "grippers" / "dex_hand" / "dex_hand.xml"
DEFAULT_BASE_PATH = PROJECT_ROOT / "assets" / "bases" / "rethink_minimal_mount.xml"

# 手相对机械臂挂载点的安装姿态，使用 xyz 欧拉角，单位为度。
DEFAULT_HAND_ROT_XYZ_DEG = (-90.0, -90.0, 0.0)

BASE_PREFIX = "mount_"
HAND_PREFIX = "dexhand_"
DEFAULT_ATTACH_POINT_NAME = "right_hand"
DEFAULT_BASE_ARM_MOUNT_SITE_NAME = "arm_mount"

PathLike = Union[str, Path]
RotXyzDeg = Tuple[float, float, float]


def _resolve_path(path: Optional[PathLike], default_path: Path) -> Path:
    """将可选路径解析成实际路径；调用方传 None 时使用默认路径。"""
    return Path(path) if path is not None else default_path


def _load_spec_or_raise(path: Path, description: str) -> mujoco.MjSpec:
    """加载 XML 为 ``MjSpec``，并在文件缺失时给出清晰错误。"""
    if not path.exists():
        raise FileNotFoundError(f"{description} XML 文件不存在: {path}")
    return mujoco.MjSpec.from_file(str(path))


def _first_body_or_raise(spec: mujoco.MjSpec, description: str) -> mujoco.MjsBody:
    """返回 worldbody 下第一个 body；缺失时带模型上下文报错。"""
    body = spec.worldbody.first_body()
    if body is None:
        raise ValueError(f"{description} XML 的 <worldbody> 下没有 body。")
    return body


def _site_or_raise(
    spec: mujoco.MjSpec,
    site_name: str,
    description: str,
) -> mujoco.MjsSite:
    """按名字查找 site；缺失时列出当前模型中的 site。"""
    try:
        return spec.site(site_name)
    except KeyError as exc:
        available = [site.name for site in spec.sites()]
        raise ValueError(
            f"{description} XML 中没有 site '{site_name}'。"
            f"可用 site: {available}"
        ) from exc


def _euler_deg_to_wxyz(rot_xyz_deg: RotXyzDeg) -> list:
    """将 xyz 欧拉角（度）转换成 MuJoCo 使用的 wxyz 四元数。"""
    x, y, z, w = R.from_euler("xyz", rot_xyz_deg, degrees=True).as_quat()
    return [w, x, y, z]


def _reset_body_pos(body: mujoco.MjsBody) -> None:
    """清零根 body 偏移，让父级 attach frame 统一负责放置。"""
    if np.linalg.norm(np.asarray(body.pos, dtype=float)) > 1e-6:
        body.pos = [0.0, 0.0, 0.0]


def _mount_arm_on_base(
    arm_spec: mujoco.MjSpec,
    base_path: Path,
    mount_site_name: str,
) -> None:
    """把底座挂到 worldbody，并把机械臂根节点放到底座挂载 site 上。"""
    base_spec = _load_spec_or_raise(base_path, "底座模型")
    base_root = _first_body_or_raise(base_spec, "底座模型")
    mount_site = _site_or_raise(base_spec, mount_site_name, "底座模型")

    mount_frame = arm_spec.worldbody.add_frame()
    mount_frame.attach_body(base_root, prefix=BASE_PREFIX, suffix="")

    arm_root = arm_spec.worldbody.first_body()
    if arm_root is None:
        return

    arm_root.pos = list(mount_site.pos)
    arm_root.quat = list(mount_site.quat)


def _attach_hand_to_arm(
    arm_spec: mujoco.MjSpec,
    hand_root: mujoco.MjsBody,
    attach_point_name: str,
    rot_xyz_deg: RotXyzDeg,
) -> None:
    """通过带旋转的 frame，把手模型根节点挂到机械臂指定 body 下。"""
    try:
        attach_point = arm_spec.body(attach_point_name)
    except KeyError as exc:
        available = [body.name for body in arm_spec.worldbody.bodies()]
        raise ValueError(
            f"机械臂模型中没有挂载 body '{attach_point_name}'。"
            f"可用 body: {available}"
        ) from exc

    attach_frame = attach_point.add_frame()
    attach_frame.pos = [0.0, 0.0, 0.0]
    attach_frame.quat = _euler_deg_to_wxyz(rot_xyz_deg)
    attach_frame.attach_body(hand_root, prefix=HAND_PREFIX, suffix="")


def _configure_solver(spec: mujoco.MjSpec) -> None:
    """给合并后的多关节模型设置更稳定的求解器参数。"""
    spec.option.timestep = 0.001
    spec.option.solver = mujoco.mjtSolver.mjSOL_NEWTON
    spec.option.iterations = 100


def _add_default_scene(spec: mujoco.MjSpec) -> None:
    """添加独立预览用的简单天空、地面和灯光。"""
    skybox_tex = spec.add_texture()
    skybox_tex.name = "skybox_tex"
    skybox_tex.type = mujoco.mjtTexture.mjTEXTURE_SKYBOX
    skybox_tex.builtin = mujoco.mjtBuiltin.mjBUILTIN_GRADIENT
    skybox_tex.rgb1 = [0.3, 0.5, 0.7]
    skybox_tex.rgb2 = [0.0, 0.0, 0.0]
    skybox_tex.width = 512
    skybox_tex.height = 3072

    ground_tex = spec.add_texture()
    ground_tex.name = "groundplane_tex"
    ground_tex.type = mujoco.mjtTexture.mjTEXTURE_2D
    ground_tex.builtin = mujoco.mjtBuiltin.mjBUILTIN_CHECKER
    ground_tex.rgb1 = [0.2, 0.3, 0.4]
    ground_tex.rgb2 = [0.1, 0.2, 0.3]
    ground_tex.width = 512
    ground_tex.height = 512

    ground_mat = spec.add_material()
    ground_mat.name = "groundplane"
    ground_mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = ground_tex.name
    ground_mat.texrepeat = [5, 5]
    ground_mat.reflectance = 0.2
    ground_mat.shininess = 0.1
    ground_mat.specular = 0.1

    spec.worldbody.add_light(
        name="top_light",
        pos=[0.0, 0.0, 4.0],
        dir=[0.0, 0.0, -1.0],
        diffuse=[2.0, 2.0, 2.0],
        ambient=[0.8, 0.8, 0.8],
        specular=[0.3, 0.3, 0.3],
    )

    floor = spec.worldbody.add_geom()
    floor.name = "floor"
    floor.type = mujoco.mjtGeom.mjGEOM_PLANE
    floor.size = [0.0, 0.0, 0.05]
    floor.material = ground_mat.name


def build_combined_spec(
    arm_path: Optional[PathLike] = None,
    hand_path: Optional[PathLike] = None,
    base_path: Optional[PathLike] = None,
    rot_xyz_deg: RotXyzDeg = DEFAULT_HAND_ROT_XYZ_DEG,
    attach_point_name: str = DEFAULT_ATTACH_POINT_NAME,
    base_mount_site_name: str = DEFAULT_BASE_ARM_MOUNT_SITE_NAME,
) -> mujoco.MjSpec:
    """
    构建未编译的机械臂 + 灵巧手 ``MjSpec``。

    Args:
        arm_path: 机械臂 XML 路径；省略时使用 ``DEFAULT_ARM_PATH``。
        hand_path: 手模型 XML 路径；省略时使用 ``DEFAULT_HAND_PATH``。
        base_path: 底座 XML 路径；省略时使用 ``DEFAULT_BASE_PATH``。
        rot_xyz_deg: 手相对 ``attach_point_name`` 的 xyz 欧拉角，单位为度。
        attach_point_name: 机械臂上用于挂载手模型的 body 名称。
        base_mount_site_name: 底座 XML 中声明机械臂根节点位置和姿态的 site。

    Returns:
        已合并但尚未编译的 ``MjSpec``，可继续定制或直接编译。
    """
    arm_path = _resolve_path(arm_path, DEFAULT_ARM_PATH)
    hand_path = _resolve_path(hand_path, DEFAULT_HAND_PATH)
    base_path = _resolve_path(base_path, DEFAULT_BASE_PATH)

    arm_spec = _load_spec_or_raise(arm_path, "机械臂模型")
    hand_spec = _load_spec_or_raise(hand_path, "手模型")
    _configure_solver(arm_spec)

    _mount_arm_on_base(arm_spec, base_path, base_mount_site_name)

    hand_root = _first_body_or_raise(hand_spec, "手模型")
    _reset_body_pos(hand_root)
    _attach_hand_to_arm(
        arm_spec=arm_spec,
        hand_root=hand_root,
        attach_point_name=attach_point_name,
        rot_xyz_deg=rot_xyz_deg,
    )

    return arm_spec


def build_combined_model(
    arm_path: Optional[PathLike] = None,
    hand_path: Optional[PathLike] = None,
    base_path: Optional[PathLike] = None,
    rot_xyz_deg: Optional[RotXyzDeg] = None,
    attach_point_name: str = DEFAULT_ATTACH_POINT_NAME,
    base_mount_site_name: str = DEFAULT_BASE_ARM_MOUNT_SITE_NAME,
    add_scene: bool = True,
) -> Tuple[mujoco.MjModel, mujoco.MjData]:
    """
    构建、按需添加预览场景，并编译合并后的机器人模型。

    ``rot_xyz_deg=None`` 表示使用 ``DEFAULT_HAND_ROT_XYZ_DEG``。这样预览入口
    和 spec 构建入口会共享同一个默认安装姿态。
    """
    spec = build_combined_spec(
        arm_path=arm_path,
        hand_path=hand_path,
        base_path=base_path,
        rot_xyz_deg=DEFAULT_HAND_ROT_XYZ_DEG if rot_xyz_deg is None else rot_xyz_deg,
        attach_point_name=attach_point_name,
        base_mount_site_name=base_mount_site_name,
    )

    if add_scene:
        _add_default_scene(spec)

    model = spec.compile()
    data = mujoco.MjData(model)
    return model, data


if __name__ == "__main__":
    print("--- 独立预览：RM75B + 灵巧手 ---")
    try:
        model, data = build_combined_model()

        with viewer.launch_passive(model, data) as v:
            while v.is_running():
                mujoco.mj_step(model, data)
                v.sync()

    except FileNotFoundError as e:
        print(f"\n[错误] 文件缺失: {e}")
    except Exception as e:
        print(f"\n[错误] 发生未预期异常: {e}")
        traceback.print_exc()
