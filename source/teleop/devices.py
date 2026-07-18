"""Hardware adapters for the stretch glove and Vive tracker."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Protocol

import numpy as np

from source.geometry import normalize_quat, quat_multiply


@dataclass(frozen=True)
class GloveSample:
    stretch: np.ndarray  # normalized six channels, 0=open and 1=flexed
    timestamp: float


@dataclass(frozen=True)
class ViveSample:
    position: np.ndarray
    quaternion_wxyz: np.ndarray
    timestamp: float
    valid: bool = True


class StretchGlove(Protocol):
    def connect(self) -> None: ...
    def read(self) -> GloveSample: ...
    def close(self) -> None: ...


class ViveTracker(Protocol):
    def connect(self) -> None: ...
    def read(self) -> ViveSample: ...
    def close(self) -> None: ...


class MockStretchGlove:
    """Stationary open-hand source used before the hardware API is available."""

    def connect(self) -> None:
        pass

    def read(self) -> GloveSample:
        return GloveSample(np.zeros(6, dtype=np.float32), time.monotonic())

    def close(self) -> None:
        pass


class MockViveTracker:
    """Stationary pose source; call ``set_pose`` from tests or a GUI adapter."""

    def __init__(self) -> None:
        self.position = np.zeros(3, dtype=np.float32)
        self.quaternion_wxyz = np.asarray([1, 0, 0, 0], dtype=np.float32)

    def connect(self) -> None:
        pass

    def set_pose(self, position, quaternion_wxyz) -> None:
        self.position = np.asarray(position, dtype=np.float32).copy()
        self.quaternion_wxyz = np.asarray(quaternion_wxyz, dtype=np.float32).copy()

    def read(self) -> ViveSample:
        return ViveSample(self.position.copy(), self.quaternion_wxyz.copy(), time.monotonic())

    def close(self) -> None:
        pass


class SineStretchGlove:
    """Fake glove API producing six smooth, phase-shifted flexion channels."""

    def __init__(self, *, frequency_hz: float = 0.15, amplitude: float = 0.45) -> None:
        if frequency_hz <= 0:
            raise ValueError("frequency_hz must be positive.")
        if not 0 <= amplitude <= 0.5:
            raise ValueError("amplitude must be in [0, 0.5].")
        self.frequency_hz = float(frequency_hz)
        self.amplitude = float(amplitude)
        self._start = None

    def connect(self) -> None:
        self._start = time.monotonic()

    def read(self) -> GloveSample:
        if self._start is None:
            raise RuntimeError("SineStretchGlove.connect() must be called first.")
        now = time.monotonic()
        phase = 2 * np.pi * self.frequency_hz * (now - self._start)
        offsets = np.linspace(0, np.pi, 6, dtype=np.float32)
        stretch = 0.5 + self.amplitude * np.sin(phase + offsets)
        return GloveSample(np.clip(stretch, 0, 1).astype(np.float32), now)

    def close(self) -> None:
        self._start = None


class SineViveTracker:
    """Fake Vive API producing bounded translation and xyz orientation motion."""

    def __init__(
        self,
        *,
        frequency_hz: float = 0.10,
        translation_amplitude=(0.04, 0.04, 0.03),
        rotation_amplitude_deg=(8.0, 8.0, 12.0),
    ) -> None:
        if frequency_hz <= 0:
            raise ValueError("frequency_hz must be positive.")
        self.frequency_hz = float(frequency_hz)
        self.translation_amplitude = np.asarray(translation_amplitude, dtype=np.float64)
        self.rotation_amplitude = np.deg2rad(rotation_amplitude_deg)
        self._base_position = np.zeros(3, dtype=np.float64)
        self._base_quaternion = np.asarray([1, 0, 0, 0], dtype=np.float64)
        self._start = None

    def connect(self) -> None:
        self._start = time.monotonic()

    def set_pose(self, position, quaternion_wxyz) -> None:
        self._base_position = np.asarray(position, dtype=np.float64).copy()
        q = np.asarray(quaternion_wxyz, dtype=np.float64)
        self._base_quaternion = q / max(np.linalg.norm(q), 1e-9)

    def read(self) -> ViveSample:
        if self._start is None:
            raise RuntimeError("SineViveTracker.connect() must be called first.")
        now = time.monotonic()
        phase = 2 * np.pi * self.frequency_hz * (now - self._start)
        waves = np.sin(phase + np.asarray([0, 2 * np.pi / 3, 4 * np.pi / 3]))
        position = self._base_position + self.translation_amplitude * waves
        angles = self.rotation_amplitude * waves
        cx, cy, cz = np.cos(angles / 2)
        sx, sy, sz = np.sin(angles / 2)
        delta = np.asarray(
            [
                cx * cy * cz + sx * sy * sz,
                sx * cy * cz - cx * sy * sz,
                cx * sy * cz + sx * cy * sz,
                cx * cy * sz - sx * sy * cz,
            ]
        )
        quaternion = normalize_quat(quat_multiply(delta, self._base_quaternion))
        return ViveSample(position.astype(np.float32), quaternion.astype(np.float32), now)

    def close(self) -> None:
        self._start = None


def _parse_glove_f_line(line: str) -> list[float] | None:
    """Parse the glove's ``Fv:v:v:v:v`` flex-sensor packet."""
    line = line.strip()
    if not line.startswith("F"):
        return None
    try:
        values = [float(value) for value in line[1:].split(":")]
    except ValueError:
        return None
    return values if len(values) == 5 else None


class StretchGloveApiDevice:
    """Classic-Bluetooth (RFCOMM) adapter for the five-sensor glove.

    The Dex Hand has four finger actuators and two thumb actuators. Since the
    glove exposes only one thumb flex sensor, it is duplicated for thumb rotate
    and thumb grasp. The incoming glove order is assumed to be
    ``thumb,index,middle,ring,pinky``.
    """

    def __init__(
        self,
        mac_address: str,
        *,
        channel: int = 1,
        serial_port: str | None = None,
        baudrate: int = 9600,
        calibration_seconds: float = 3.0,
        calibration_pose_delay_seconds: float = 0.0,
        calibration_confirmation: Callable[[str], None] | None = None,
        calibration_minimum: list[float] | None = None,
        calibration_maximum: list[float] | None = None,
    ):
        if not mac_address:
            raise ValueError("A glove Bluetooth MAC address is required.")
        if calibration_seconds <= 0:
            raise ValueError("calibration_seconds must be positive.")
        if calibration_pose_delay_seconds < 0:
            raise ValueError("calibration_pose_delay_seconds must be non-negative.")
        if baudrate <= 0:
            raise ValueError("baudrate must be positive.")
        self.mac_address = mac_address
        self.channel = int(channel)
        self.serial_port = serial_port
        self.baudrate = int(baudrate)
        self.calibration_seconds = float(calibration_seconds)
        self.calibration_pose_delay_seconds = float(calibration_pose_delay_seconds)
        self.calibration_confirmation = calibration_confirmation
        self._socket = None
        self._serial = None
        self._buffer = ""
        if (calibration_minimum is None) != (calibration_maximum is None):
            raise ValueError("Both calibration_minimum and calibration_maximum are required.")
        self._minimum = (
            None
            if calibration_minimum is None
            else np.asarray(calibration_minimum, dtype=np.float32)
        )
        self._maximum = (
            None
            if calibration_maximum is None
            else np.asarray(calibration_maximum, dtype=np.float32)
        )
        if self._minimum is not None:
            if self._minimum.shape != (5,) or self._maximum.shape != (5,):
                raise ValueError("Saved glove calibration must contain five channels.")
            if np.any(self._maximum <= self._minimum):
                raise ValueError("Saved glove fist values must exceed open-hand values.")

    def connect(self) -> None:
        if self.serial_port:
            try:
                import serial
            except ImportError as exc:
                raise RuntimeError("pyserial is required for a configured glove COM port.") from exc
            resolved_port = self._resolve_serial_port()
            try:
                self._serial = serial.Serial(
                    resolved_port,
                    self.baudrate,
                    timeout=max(2.0, self.calibration_seconds),
                )
            except serial.SerialException as exc:
                if "121" in str(exc) or "信号灯超时时间已到" in str(exc):
                    raise ConnectionError(
                        f"Windows found {resolved_port}, but HC-06 did not accept the SPP "
                        "connection. Power-cycle the glove, disconnect other phones/PCs, "
                        "and remove then re-pair HC-06 if the error persists."
                    ) from exc
                raise
            self._serial.reset_input_buffer()
            print(f"Glove connected: {resolved_port} baudrate={self.baudrate}")
            self._initialize_calibration()
            return
        try:
            import bluetooth
        except ImportError as exc:
            raise RuntimeError(
                "PyBluez is required for the glove: "
                "pip install git+https://github.com/pybluez/pybluez.git#egg=pybluez"
            ) from exc
        self._socket = bluetooth.BluetoothSocket(bluetooth.RFCOMM)
        self._socket.connect((self.mac_address, self.channel))
        print(f"Glove connected: {self.mac_address} channel={self.channel}")
        self._initialize_calibration()

    def _initialize_calibration(self) -> None:
        if self._minimum is not None and self._maximum is not None:
            print(
                "Glove calibration loaded: "
                f"open={self._minimum.tolist()} fist={self._maximum.tolist()}"
            )
            return
        self._calibrate()

    def calibration_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """Return copies of the active five-channel raw calibration bounds."""
        if self._minimum is None or self._maximum is None:
            raise RuntimeError("The glove has not been calibrated.")
        return self._minimum.copy(), self._maximum.copy()

    def _resolve_serial_port(self) -> str:
        if str(self.serial_port).lower() != "auto":
            return str(self.serial_port)
        from serial.tools import list_ports

        address = "".join(character for character in self.mac_address if character.isalnum())
        address = address.upper()
        matches = [
            port.device
            for port in list_ports.comports()
            if address and address in str(port.hwid).upper()
        ]
        if len(matches) != 1:
            raise RuntimeError(
                f"Cannot uniquely find glove COM port for MAC {self.mac_address}; "
                f"detected matches={matches}. Set glove_serial_port explicitly."
            )
        return matches[0]

    def _calibrate(self) -> None:
        fist = self._collect_calibration("握紧拳头", self.calibration_seconds)
        opened = self._collect_calibration("完全张开手掌", self.calibration_seconds)
        # Raw extrema are dominated by sensor noise.  In particular,
        # min(opened) makes a normally held flat hand remain well above zero.
        # Inner percentiles describe stable poses the operator can reproduce.
        self._minimum = np.percentile(opened, 90.0, axis=0).astype(np.float32)
        self._maximum = np.percentile(fist, 10.0, axis=0).astype(np.float32)
        if np.any(self._maximum <= self._minimum):
            self.close()
            raise RuntimeError(
                "Glove calibration failed: stable fist values must exceed "
                "stable open-hand values on every channel."
            )
        print(f"Glove calibration open={self._minimum.tolist()} fist={self._maximum.tolist()}")

    def _read_flex(self) -> np.ndarray:
        if self._socket is None and self._serial is None:
            raise RuntimeError("StretchGloveApiDevice.connect() must be called first.")
        while True:
            while "\n" in self._buffer:
                line, self._buffer = self._buffer.split("\n", 1)
                values = _parse_glove_f_line(line)
                if values is not None:
                    return np.asarray(values, dtype=np.float32)
            if self._serial is not None:
                data = self._serial.readline()
                if not data:
                    raise TimeoutError(
                        f"No glove data received from {self.serial_port}; "
                        "confirm that the middle transmit button was pressed."
                    )
            else:
                data = self._socket.recv(1024)
            if not data:
                raise ConnectionError("The glove Bluetooth connection was closed.")
            self._buffer += data.decode(errors="ignore").replace("\r", "")

    def _collect_calibration(self, pose: str, duration: float) -> np.ndarray:
        if self.calibration_pose_delay_seconds > 0:
            print(
                f"请准备{pose}，{self.calibration_pose_delay_seconds:g} 秒后开始采样...",
                flush=True,
            )
            time.sleep(self.calibration_pose_delay_seconds)
        if self.calibration_confirmation is not None:
            self.calibration_confirmation(pose)
            self.discard_pending()
        print(f"Glove calibration: 请{pose}并保持 {duration:g} 秒...")
        samples = []
        deadline = time.monotonic() + duration
        while time.monotonic() < deadline:
            samples.append(self._read_flex())
        if not samples:
            raise RuntimeError(f"No glove samples received while calibrating: {pose}.")
        return np.stack(samples)

    def discard_pending(self) -> None:
        """Discard packets queued while waiting for an operator confirmation."""
        self._buffer = ""
        if self._serial is not None:
            self._serial.reset_input_buffer()
            return
        if self._socket is None:
            return
        self._socket.setblocking(False)
        try:
            while self._socket.recv(4096):
                pass
        except OSError:
            pass
        finally:
            self._socket.setblocking(True)

    def read(self) -> GloveSample:
        if self._minimum is None or self._maximum is None:
            raise RuntimeError("StretchGloveApiDevice.connect() must be called first.")
        raw = self._read_flex()
        flex = np.clip((raw - self._minimum) / (self._maximum - self._minimum), 0.0, 1.0)
        # glove: thumb,index,middle,ring,pinky -> hand: index,middle,ring,pinky,thumb x2
        stretch = flex[[1, 2, 3, 4, 0, 0]].astype(np.float32)
        return GloveSample(stretch, time.monotonic())

    def close(self) -> None:
        if self._socket is not None:
            try:
                self._socket.shutdown(2)
            except (AttributeError, OSError):
                pass
            finally:
                self._socket.close()
        self._socket = None
        if self._serial is not None:
            try:
                self._serial.cancel_read()
            except (AttributeError, OSError):
                pass
            try:
                self._serial.reset_input_buffer()
                self._serial.reset_output_buffer()
                self._serial.dtr = False
                self._serial.rts = False
            except OSError:
                pass
            finally:
                self._serial.close()
        self._serial = None
        self._buffer = ""
        self._minimum = None
        self._maximum = None


def _openvr_matrix_to_quat(matrix) -> np.ndarray:
    """Convert an OpenVR 3x4 pose matrix to a normalized wxyz quaternion."""
    rotation = np.asarray([[matrix[row][col] for col in range(3)] for row in range(3)])
    trace = float(np.trace(rotation))
    if trace > 0:
        scale = np.sqrt(trace + 1.0) * 2
        quat = [
            0.25 * scale,
            (rotation[2, 1] - rotation[1, 2]) / scale,
            (rotation[0, 2] - rotation[2, 0]) / scale,
            (rotation[1, 0] - rotation[0, 1]) / scale,
        ]
    else:
        index = int(np.argmax(np.diag(rotation)))
        j, k = (index + 1) % 3, (index + 2) % 3
        scale = np.sqrt(1.0 + rotation[index, index] - rotation[j, j] - rotation[k, k]) * 2
        xyz = np.zeros(3)
        xyz[index] = 0.25 * scale
        xyz[j] = (rotation[j, index] + rotation[index, j]) / scale
        xyz[k] = (rotation[k, index] + rotation[index, k]) / scale
        quat = [(rotation[k, j] - rotation[j, k]) / scale, *xyz]
    return normalize_quat(np.asarray(quat, dtype=np.float64)).astype(np.float32)


class ViveApiTracker:
    """Read a GenericTracker pose from the OpenVR/SteamVR runtime."""

    def __init__(self, *, device_index: int | None = None, serial: str | None = None):
        self.requested_device_index = device_index
        self.serial = serial
        self._openvr = None
        self._system = None
        self._device_index = None

    def connect(self) -> None:
        try:
            import openvr
        except ImportError as exc:
            raise RuntimeError("OpenVR is required for Vive: pip install openvr") from exc
        self._openvr = openvr
        openvr.init(openvr.VRApplication_Utility)
        self._system = openvr.VRSystem()
        candidates = []
        for index in range(openvr.k_unMaxTrackedDeviceCount):
            if (
                self._system.getTrackedDeviceClass(index)
                != openvr.TrackedDeviceClass_GenericTracker
            ):
                continue
            device_serial = self._system.getStringTrackedDeviceProperty(
                index, openvr.Prop_SerialNumber_String
            )
            candidates.append((index, device_serial))
        if self.requested_device_index is not None:
            match = [item for item in candidates if item[0] == self.requested_device_index]
        elif self.serial:
            match = [item for item in candidates if item[1] == self.serial]
        else:
            match = candidates[:1]
        if not match:
            self.close()
            raise RuntimeError(f"No matching Vive GenericTracker found; detected={candidates}.")
        self._device_index = match[0][0]
        print(f"Vive connected: index={match[0][0]} serial={match[0][1]}")

    def set_pose(self, position, quaternion_wxyz) -> None:
        # The real tracker pose is calibrated relatively by TeleopMapper.
        del position, quaternion_wxyz

    def read(self) -> ViveSample:
        if self._system is None or self._device_index is None:
            raise RuntimeError("ViveApiTracker.connect() must be called first.")
        openvr = self._openvr
        pose_type = openvr.TrackedDevicePose_t * openvr.k_unMaxTrackedDeviceCount
        poses = self._system.getDeviceToAbsoluteTrackingPose(
            openvr.TrackingUniverseStanding, 0, pose_type()
        )
        pose = poses[self._device_index]
        if not pose.bPoseIsValid:
            return ViveSample(
                np.zeros(3, dtype=np.float32),
                np.asarray([1, 0, 0, 0], dtype=np.float32),
                time.monotonic(),
                valid=False,
            )
        matrix = pose.mDeviceToAbsoluteTracking.m
        position = np.asarray([matrix[0][3], matrix[1][3], matrix[2][3]], dtype=np.float32)
        return ViveSample(position, _openvr_matrix_to_quat(matrix), time.monotonic())

    def close(self) -> None:
        if self._openvr is not None:
            self._openvr.shutdown()
        self._openvr = None
        self._system = None
        self._device_index = None
