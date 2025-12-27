#!/usr/bin/env python3
"""
DRFL 기반 Doosan E0509 로봇 인터페이스

DRFL (Doosan Robot Framework Library)을 사용하여 로봇을 직접 제어합니다.
ROS2 없이 독립적으로 실행 가능합니다.

Usage:
    from robot_interface import DoosanRobot

    robot = DoosanRobot("192.168.137.100")
    robot.connect()

    # 현재 상태 읽기
    joint_pos = robot.get_joint_positions()  # 라디안
    tcp_pos, tcp_rot = robot.get_tcp_pose()

    # 이동 명령
    robot.move_joint(target_rad, vel=30, acc=30)
    robot.move_linear(target_pos, target_rot, vel=100, acc=100)
"""

import numpy as np
import time
import math
from typing import Tuple, Optional, List

# DRFL import 시도
try:
    import DRFL
    DRFL_AVAILABLE = True
except ImportError:
    DRFL_AVAILABLE = False
    print("[Warning] DRFL not found.")

# ROS2 import 시도
try:
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import JointState
    from std_srvs.srv import Trigger
    import threading
    ROS2_AVAILABLE = True
except ImportError:
    ROS2_AVAILABLE = False
    print("[Warning] ROS2 not found.")


class DoosanRobot:
    """Doosan E0509 로봇 인터페이스 (DRFL 또는 ROS2 기반)"""

    # E0509 관절 한계 (도)
    JOINT_LIMITS_DEG = {
        'lower': [-360, -95, -135, -360, -135, -360],
        'upper': [360, 95, 135, 360, 135, 360]
    }

    # Home 자세 (도)
    HOME_JOINT_DEG = [0, 0, 90, 0, 90, 0]

    def __init__(self, ip: str = "192.168.137.100", simulation: bool = False,
                 use_ros2: bool = True, ros2_namespace: str = "dsr01"):
        """
        Args:
            ip: 로봇 컨트롤러 IP 주소
            simulation: True면 시뮬레이션 모드로 실행
            use_ros2: True면 ROS2 토픽 사용 (우선), False면 DRFL 직접 사용
            ros2_namespace: ROS2 네임스페이스 (예: "dsr01")
        """
        self.ip = ip
        self.ros2_namespace = ros2_namespace
        self.connected = False

        # 모드 결정: ROS2 > DRFL > Simulation
        self.use_ros2 = use_ros2 and ROS2_AVAILABLE
        self.use_drfl = not self.use_ros2 and DRFL_AVAILABLE
        self.simulation = simulation or (not self.use_ros2 and not self.use_drfl)

        # 관절 한계 (라디안)
        self.joint_limits_lower = np.radians(self.JOINT_LIMITS_DEG['lower'])
        self.joint_limits_upper = np.radians(self.JOINT_LIMITS_DEG['upper'])

        # 시뮬레이션용 상태
        self._sim_joint_pos = np.radians(self.HOME_JOINT_DEG)
        self._sim_joint_vel = np.zeros(6)

        # ROS2 관련
        self._ros2_node = None
        self._ros2_thread = None
        self._ros2_joint_pos = None
        self._ros2_tcp_pos = None  # [x, y, z, rx, ry, rz]

        if self.use_ros2:
            print(f"[Robot] ROS2 mode, namespace: /{ros2_namespace}")
        elif self.use_drfl:
            print(f"[Robot] DRFL mode, target IP: {ip}")
        else:
            print(f"[Robot] Simulation mode")

    def connect(self) -> bool:
        """로봇 연결"""
        if self.simulation:
            self.connected = True
            print("[Robot] Connected (simulation)")
            return True

        if self.use_ros2:
            return self._connect_ros2()

        # DRFL 모드
        try:
            DRFL.open_connection(self.ip)
            self.connected = True
            print(f"[Robot] Connected to {self.ip}")
            return True
        except Exception as e:
            print(f"[Robot] Connection failed: {e}")
            return False

    def _connect_ros2(self) -> bool:
        """ROS2 연결"""
        try:
            if not rclpy.ok():
                rclpy.init()

            self._ros2_node = rclpy.create_node('robot_interface_node')

            # Joint state 구독
            joint_topic = f'/{self.ros2_namespace}/joint_states'
            self._ros2_node.create_subscription(
                JointState, joint_topic,
                self._joint_state_callback, 10
            )

            # TCP 토픽 구독 (Doosan은 /dsr01/current_posx 등)
            # 또는 서비스를 통해 읽기
            try:
                from dsr_msgs2.msg import RobotState
                state_topic = f'/{self.ros2_namespace}/state'
                self._ros2_node.create_subscription(
                    RobotState, state_topic,
                    self._robot_state_callback, 10
                )
            except ImportError:
                print("[Robot] dsr_msgs2 not found, TCP reading limited")

            # 백그라운드에서 spin
            self._ros2_thread = threading.Thread(target=self._ros2_spin, daemon=True)
            self._ros2_thread.start()

            # 첫 데이터 대기
            timeout = 3.0
            start = time.time()
            while self._ros2_joint_pos is None and (time.time() - start) < timeout:
                time.sleep(0.1)

            if self._ros2_joint_pos is not None:
                self.connected = True
                print(f"[Robot] Connected via ROS2")
                return True
            else:
                print(f"[Robot] ROS2 connection timeout - no joint_states received")
                print(f"[Robot] Check if topic '{joint_topic}' exists")
                return False

        except Exception as e:
            print(f"[Robot] ROS2 connection failed: {e}")
            return False

    def _ros2_spin(self):
        """ROS2 spin in background"""
        while rclpy.ok() and self._ros2_node:
            rclpy.spin_once(self._ros2_node, timeout_sec=0.1)

    def _joint_state_callback(self, msg: 'JointState'):
        """Joint state 콜백"""
        if len(msg.position) >= 6:
            self._ros2_joint_pos = np.array(msg.position[:6])

    def _robot_state_callback(self, msg):
        """Robot state 콜백 (TCP 포함)"""
        try:
            # dsr_msgs2.msg.RobotState의 current_posx
            if hasattr(msg, 'current_posx'):
                self._ros2_tcp_pos = np.array(msg.current_posx)
        except:
            pass

    def disconnect(self):
        """로봇 연결 해제"""
        if self.use_ros2 and self._ros2_node:
            self._ros2_node.destroy_node()
            self._ros2_node = None

        if self.use_drfl and self.connected:
            try:
                DRFL.close_connection()
            except:
                pass

        self.connected = False
        print("[Robot] Disconnected")

    def get_joint_positions(self) -> np.ndarray:
        """
        현재 관절 위치 읽기

        Returns:
            joint_pos: (6,) 라디안 단위
        """
        if self.simulation:
            return self._sim_joint_pos.copy()

        if self.use_ros2:
            if self._ros2_joint_pos is not None:
                return self._ros2_joint_pos.copy()
            return np.zeros(6)

        pos_deg = DRFL.get_current_posj()
        return np.radians(pos_deg)

    def get_joint_velocities(self) -> np.ndarray:
        """
        현재 관절 속도 읽기

        Returns:
            joint_vel: (6,) 라디안/초 단위
        """
        if self.simulation:
            return self._sim_joint_vel.copy()

        # DRFL에서 직접 속도를 읽는 API가 없으면 0 반환
        # 실제로는 이전 위치와 현재 위치로 계산하거나,
        # get_current_velj() 같은 함수가 있다면 사용
        try:
            vel_deg = DRFL.get_current_velj()
            return np.radians(vel_deg)
        except:
            return np.zeros(6)

    def get_tcp_pose(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        현재 TCP pose 읽기 (Forward Kinematics)

        Returns:
            position: (3,) [x, y, z] 미터 단위
            rotation: (3, 3) 회전 행렬
        """
        if self.simulation:
            # 간단한 시뮬레이션 FK (실제로는 정확한 FK 필요)
            position = np.array([0.4, 0.0, 0.4])
            rotation = np.eye(3)
            return position, rotation

        if self.use_ros2:
            return self._get_tcp_pose_ros2()

        # DRFL은 [x, y, z, rx, ry, rz] (mm, deg) 반환
        posx = DRFL.get_current_posx()

        position = np.array(posx[:3]) / 1000.0  # mm -> m

        # ZYZ Euler -> 회전 행렬
        rx, ry, rz = np.radians(posx[3:6])
        rotation = self._euler_to_rotation_matrix(rx, ry, rz)

        return position, rotation

    def _get_tcp_pose_ros2(self, force_refresh: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """ROS2를 통해 TCP pose 읽기

        Args:
            force_refresh: True면 캐시 무시하고 강제로 새로 읽기 (캡처 시 사용)
        """
        import subprocess
        import re

        current_time = time.time()

        # 캐시된 결과 사용 (force_refresh가 아니고, 0.5초 이내면 재사용)
        if not force_refresh:
            if hasattr(self, '_tcp_cache') and hasattr(self, '_tcp_cache_time'):
                if current_time - self._tcp_cache_time < 0.5:
                    return self._tcp_cache
                # 캐시가 오래되면 새로 읽기 (아래로 진행)

        # subprocess로 ros2 service call 직접 실행
        try:
            bash_cmd = (
                f"source /opt/ros/humble/setup.bash && "
                f"source ~/doosan_ws/install/setup.bash && "
                f"ros2 service call /{self.ros2_namespace}/aux_control/get_current_posx "
                f"dsr_msgs2/srv/GetCurrentPosx '{{ref: 0}}'"
            )

            # force_refresh일 때는 더 긴 타임아웃
            timeout = 10.0 if force_refresh else 3.0

            result = subprocess.run(
                ['bash', '-c', bash_cmd],
                capture_output=True, text=True, timeout=timeout
            )

            if result.returncode == 0 and result.stdout:
                # 출력에서 data 배열 파싱
                match = re.search(r'data=\[([^\]]+)\]', result.stdout)
                if match:
                    data_str = match.group(1)
                    values = [float(x.strip()) for x in data_str.split(',')]
                    if len(values) >= 6:
                        posx = values[:6]
                        position = np.array([posx[0], posx[1], posx[2]]) / 1000.0

                        # Doosan uses ZYZ Euler angles (degrees): [A, B, C]
                        # posx = [x, y, z, A, B, C] where A,B,C are ZYZ intrinsic rotations
                        # Reference: https://manual.doosanrobotics.com/en/user/2.12.2/2.-A-Series/what-is-euler-angle-a-b-c
                        from scipy.spatial.transform import Rotation as R
                        a, b, c = np.radians(posx[3]), np.radians(posx[4]), np.radians(posx[5])
                        rotation = R.from_euler('ZYZ', [a, b, c]).as_matrix()

                        # 캐시에 저장
                        self._tcp_cache = (position, rotation)
                        self._tcp_cache_time = current_time
                        return position, rotation

        except subprocess.TimeoutExpired:
            if force_refresh:
                print("[Robot] TCP service timeout - 로봇이 정지한 상태에서 다시 시도하세요")
        except Exception as e:
            if force_refresh:
                print(f"[Robot] TCP read failed: {e}")

        # 캐시가 있으면 반환, 없으면 FK fallback
        if hasattr(self, '_tcp_cache'):
            return self._tcp_cache

        print("[Robot] Warning: Using simple FK (inaccurate)")
        return self._simple_fk(self.get_joint_positions())

    def get_tcp_pose_accurate(self) -> Tuple[np.ndarray, np.ndarray]:
        """정확한 TCP pose 읽기 (캡처용, 느림)"""
        return self._get_tcp_pose_ros2(force_refresh=True)

    def _simple_fk(self, joint_pos: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """간단한 Forward Kinematics (E0509 근사)"""
        # E0509의 대략적인 link 길이 사용
        # 정확한 FK가 필요하면 DH 파라미터 사용해야 함
        # 여기서는 joint_states 기반 대략적인 추정만
        q = joint_pos

        # 매우 단순화된 FK (정확하지 않음, 실제로는 DH 파라미터 필요)
        L1, L2, L3, L4 = 0.155, 0.410, 0.085, 0.410  # 대략적인 링크 길이

        c1, s1 = np.cos(q[0]), np.sin(q[0])
        c2, s2 = np.cos(q[1]), np.sin(q[1])
        c3, s3 = np.cos(q[2]), np.sin(q[2])

        # 대략적인 위치 계산
        x = (L2 * c2 + L4 * np.cos(q[1] + q[2])) * c1
        y = (L2 * c2 + L4 * np.cos(q[1] + q[2])) * s1
        z = L1 + L2 * s2 + L4 * np.sin(q[1] + q[2]) + L3

        position = np.array([x, y, z])
        rotation = np.eye(3)  # 회전은 생략

        return position, rotation

    def get_tcp_position(self) -> np.ndarray:
        """TCP 위치만 반환 (미터)"""
        pos, _ = self.get_tcp_pose()
        return pos

    def move_joint(self, target_rad: np.ndarray, vel: float = 30, acc: float = 30,
                   wait: bool = True) -> bool:
        """
        관절 공간 이동

        Args:
            target_rad: (6,) 목표 관절 위치 (라디안)
            vel: 속도 (도/초)
            acc: 가속도 (도/초^2)
            wait: 완료까지 대기 여부

        Returns:
            성공 여부
        """
        # 관절 한계 클램핑
        target_rad = np.clip(target_rad, self.joint_limits_lower, self.joint_limits_upper)
        target_deg = np.degrees(target_rad).tolist()

        if self.simulation:
            self._sim_joint_pos = target_rad.copy()
            print(f"[Robot] Move joint: {np.round(target_deg, 1)} deg")
            if wait:
                time.sleep(0.1)
            return True

        try:
            DRFL.movej(target_deg, vel=vel, acc=acc)
            if wait:
                DRFL.wait_motion()
            return True
        except Exception as e:
            print(f"[Robot] Move joint failed: {e}")
            return False

    def move_joint_delta(self, delta_rad: np.ndarray, vel: float = 30, acc: float = 30,
                         wait: bool = True) -> bool:
        """
        현재 위치에서 상대적 관절 이동

        Args:
            delta_rad: (6,) 관절 변화량 (라디안)
        """
        current = self.get_joint_positions()
        target = current + delta_rad
        return self.move_joint(target, vel, acc, wait)

    def move_linear(self, position: np.ndarray, rotation: np.ndarray = None,
                    vel: float = 100, acc: float = 100, wait: bool = True) -> bool:
        """
        직선 이동 (TCP 공간)

        Args:
            position: (3,) 목표 위치 [x, y, z] (미터)
            rotation: (3, 3) 목표 회전 행렬 (None이면 현재 유지)
            vel: 속도 (mm/초)
            acc: 가속도 (mm/초^2)
            wait: 완료까지 대기 여부
        """
        if rotation is None:
            _, rotation = self.get_tcp_pose()

        # 회전 행렬 -> ZYZ Euler (도)
        rx, ry, rz = self._rotation_matrix_to_euler(rotation)

        # [x, y, z, rx, ry, rz] (mm, deg)
        posx = [
            position[0] * 1000,  # m -> mm
            position[1] * 1000,
            position[2] * 1000,
            np.degrees(rx),
            np.degrees(ry),
            np.degrees(rz)
        ]

        if self.simulation:
            print(f"[Robot] Move linear: pos={np.round(position, 3)}")
            if wait:
                time.sleep(0.1)
            return True

        try:
            DRFL.movel(posx, vel=vel, acc=acc)
            if wait:
                DRFL.wait_motion()
            return True
        except Exception as e:
            print(f"[Robot] Move linear failed: {e}")
            return False

    def move_to_home(self, vel: float = 30, acc: float = 30) -> bool:
        """Home 위치로 이동"""
        print("[Robot] Moving to home position...")
        home_rad = np.radians(self.HOME_JOINT_DEG)
        return self.move_joint(home_rad, vel, acc, wait=True)

    def stop(self):
        """긴급 정지"""
        if not self.simulation:
            try:
                DRFL.stop()
            except:
                pass
        print("[Robot] STOP")

    def servo_on(self) -> bool:
        """서보 ON"""
        if self.simulation:
            print("[Robot] Servo ON (simulation)")
            return True

        try:
            DRFL.servo_on()
            print("[Robot] Servo ON")
            return True
        except Exception as e:
            print(f"[Robot] Servo ON failed: {e}")
            return False

    def servo_off(self) -> bool:
        """서보 OFF"""
        if self.simulation:
            print("[Robot] Servo OFF (simulation)")
            return True

        try:
            DRFL.servo_off()
            print("[Robot] Servo OFF")
            return True
        except Exception as e:
            print(f"[Robot] Servo OFF failed: {e}")
            return False

    # =========================================================================
    # 유틸리티 함수
    # =========================================================================

    def _euler_to_rotation_matrix(self, a: float, b: float, c: float) -> np.ndarray:
        """ZYZ Euler (Doosan 표준) 각도 -> 회전 행렬

        Doosan posx: [x, y, z, a, b, c] 형식
        ZYZ intrinsic: R = Rz(a) * Ry(b) * Rz(c)
        """
        ca, sa = np.cos(a), np.sin(a)
        cb, sb = np.cos(b), np.sin(b)
        cc, sc = np.cos(c), np.sin(c)

        # Rz(a)
        Rz1 = np.array([[ca, -sa, 0], [sa, ca, 0], [0, 0, 1]])
        # Ry(b)
        Ry = np.array([[cb, 0, sb], [0, 1, 0], [-sb, 0, cb]])
        # Rz(c)
        Rz2 = np.array([[cc, -sc, 0], [sc, cc, 0], [0, 0, 1]])

        return Rz1 @ Ry @ Rz2

    def _rotation_matrix_to_euler(self, R: np.ndarray) -> Tuple[float, float, float]:
        """회전 행렬 -> ZYZ Euler 각도"""
        # 간단한 구현 (gimbal lock 처리 없음)
        ry = np.arccos(np.clip(R[2, 2], -1, 1))

        if np.abs(np.sin(ry)) > 1e-6:
            rz = np.arctan2(R[1, 2], R[0, 2])
            rx = np.arctan2(R[2, 1], -R[2, 0])
        else:
            rz = 0
            rx = np.arctan2(-R[0, 1], R[0, 0])

        return rx, ry, rz

    def clamp_joint_positions(self, joint_pos: np.ndarray) -> np.ndarray:
        """관절 한계 내로 클램핑"""
        return np.clip(joint_pos, self.joint_limits_lower, self.joint_limits_upper)


# =============================================================================
# 테스트
# =============================================================================
def test_robot():
    """로봇 인터페이스 테스트 (시뮬레이션 모드)"""
    print("=" * 60)
    print("DoosanRobot 테스트 (Simulation Mode)")
    print("=" * 60)

    robot = DoosanRobot(simulation=True)
    robot.connect()

    # 현재 상태
    joint_pos = robot.get_joint_positions()
    print(f"\n현재 관절 위치: {np.degrees(joint_pos).round(1)} deg")

    tcp_pos, tcp_rot = robot.get_tcp_pose()
    print(f"현재 TCP 위치: {tcp_pos.round(3)} m")

    # 이동 테스트
    print("\n관절 이동 테스트...")
    target = np.radians([10, -10, 80, 0, 100, 0])
    robot.move_joint(target)

    print(f"이동 후 관절 위치: {np.degrees(robot.get_joint_positions()).round(1)} deg")

    # Home
    robot.move_to_home()

    robot.disconnect()
    print("\n테스트 완료!")


if __name__ == "__main__":
    test_robot()
