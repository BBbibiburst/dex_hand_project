"""只测试 Vive Tracker 链路，不连接机械臂。

用途：
1. 检查 SteamVR/OpenVR 是否能启动。
2. 检查能否找到 Vive Tracker。
3. 持续打印 Tracker 的绝对位置和相对初始位置的位移。

使用前请先启动 SteamVR，并确认 Tracker 已经配对、定位正常。
"""
import math
import time

import openvr


# ============================ 可调参数 ============================
# None 表示自动选择第一个 GenericTracker；也可以手动填 SteamVR 设备编号，如 3、4、5。
TARGET_DEVICE_INDEX = None

# 打印周期，单位秒。调小会刷屏更快，调大更容易看清。
PRINT_INTERVAL_SEC = 0.10

# True 时打印旋转四元数，False 时只看位置链路。
PRINT_QUATERNION = True
# =================================================================


DEVICE_CLASS_NAMES = {
    openvr.TrackedDeviceClass_Invalid: "Invalid",
    openvr.TrackedDeviceClass_HMD: "HMD",
    openvr.TrackedDeviceClass_Controller: "Controller",
    openvr.TrackedDeviceClass_GenericTracker: "GenericTracker",
    openvr.TrackedDeviceClass_TrackingReference: "TrackingReference",
    openvr.TrackedDeviceClass_DisplayRedirect: "DisplayRedirect",
}


def _safe_device_string(vr_system, device_index, prop):
    """读取设备字符串属性，失败时返回空字符串。"""
    try:
        return vr_system.getStringTrackedDeviceProperty(device_index, prop)
    except Exception:
        return ""


def _rot_to_quat(matrix_3x4):
    """把 OpenVR 的 3x4 姿态矩阵转换成四元数 [w, x, y, z]。"""
    r00, r01, r02 = matrix_3x4[0][0], matrix_3x4[0][1], matrix_3x4[0][2]
    r10, r11, r12 = matrix_3x4[1][0], matrix_3x4[1][1], matrix_3x4[1][2]
    r20, r21, r22 = matrix_3x4[2][0], matrix_3x4[2][1], matrix_3x4[2][2]
    trace = r00 + r11 + r22

    if trace > 0:
        s = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (r21 - r12) / s
        y = (r02 - r20) / s
        z = (r10 - r01) / s
    elif r00 > r11 and r00 > r22:
        s = math.sqrt(1.0 + r00 - r11 - r22) * 2.0
        w = (r21 - r12) / s
        x = 0.25 * s
        y = (r01 + r10) / s
        z = (r02 + r20) / s
    elif r11 > r22:
        s = math.sqrt(1.0 + r11 - r00 - r22) * 2.0
        w = (r02 - r20) / s
        x = (r01 + r10) / s
        y = 0.25 * s
        z = (r12 + r21) / s
    else:
        s = math.sqrt(1.0 + r22 - r00 - r11) * 2.0
        w = (r10 - r01) / s
        x = (r02 + r20) / s
        y = (r12 + r21) / s
        z = 0.25 * s

    norm = math.sqrt(w * w + x * x + y * y + z * z) + 1e-12
    return [w / norm, x / norm, y / norm, z / norm]


def _list_devices(vr_system):
    """枚举当前 SteamVR 识别到的设备。"""
    print("当前 OpenVR 设备列表：")
    for idx in range(openvr.k_unMaxTrackedDeviceCount):
        device_class = vr_system.getTrackedDeviceClass(idx)
        if device_class == openvr.TrackedDeviceClass_Invalid:
            continue

        class_name = DEVICE_CLASS_NAMES.get(device_class, f"Unknown({device_class})")
        serial = _safe_device_string(vr_system, idx, openvr.Prop_SerialNumber_String)
        model = _safe_device_string(vr_system, idx, openvr.Prop_ModelNumber_String)
        print(f"  [{idx:02d}] {class_name:17s} serial={serial} model={model}")


def _select_tracker(vr_system):
    """选择要测试的 Vive Tracker 设备编号。"""
    if TARGET_DEVICE_INDEX is not None:
        return int(TARGET_DEVICE_INDEX)

    for idx in range(openvr.k_unMaxTrackedDeviceCount):
        try:
            if vr_system.getTrackedDeviceClass(idx) == openvr.TrackedDeviceClass_GenericTracker:
                return idx
        except Exception:
            continue
    return -1


def main():
    vr_system = None
    try:
        openvr.init(openvr.VRApplication_Utility)
        vr_system = openvr.VRSystem()
        _list_devices(vr_system)

        tracker_index = _select_tracker(vr_system)
        if tracker_index < 0:
            print("\n未找到 GenericTracker。请确认 Vive Tracker 已开机、配对，并在 SteamVR 中显示为已定位。")
            return

        print(f"\n开始读取设备 [{tracker_index}]。按 Ctrl+C 退出。")
        pose_array_type = openvr.TrackedDevicePose_t * openvr.k_unMaxTrackedDeviceCount
        first_pos = None
        frame = 0

        while True:
            poses = vr_system.getDeviceToAbsoluteTrackingPose(
                openvr.TrackingUniverseStanding, 0, pose_array_type()
            )
            pose = poses[tracker_index]
            if not pose.bPoseIsValid:
                print("Tracker 姿态暂时无效，请检查遮挡、基站、SteamVR 状态。")
                time.sleep(PRINT_INTERVAL_SEC)
                continue

            matrix = pose.mDeviceToAbsoluteTracking.m
            pos = [matrix[0][3], matrix[1][3], matrix[2][3]]
            if first_pos is None:
                first_pos = pos[:]
                print(f"已记录初始位置: {[round(v, 4) for v in first_pos]}")

            delta = [pos[i] - first_pos[i] for i in range(3)]
            text = (
                f"帧 {frame:05d} | "
                f"pos(m)={[round(v, 4) for v in pos]} | "
                f"delta(m)={[round(v, 4) for v in delta]}"
            )
            if PRINT_QUATERNION:
                quat = _rot_to_quat(matrix)
                text += f" | quat(wxyz)={[round(v, 4) for v in quat]}"
            print(text)

            frame += 1
            time.sleep(PRINT_INTERVAL_SEC)
    except KeyboardInterrupt:
        print("\n已退出 Vive 链路测试。")
    except Exception as exc:
        print("OpenVR 测试失败：", exc)
    finally:
        if vr_system is not None:
            openvr.shutdown()


if __name__ == "__main__":
    main()
