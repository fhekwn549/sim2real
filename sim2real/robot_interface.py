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

    # 이동 명령 (위치 제어)
    robot.move_joint(target_rad, vel=30, acc=30)
    robot.move_linear(target_pos, target_rot, vel=100, acc=100)

    # 토크 제어 (실시간 제어)
    robot.start_rt_control()
    while running:
        state = robot.read_rt_state()
        torque = compute_torque(state)
        robot.set_torque(torque)
    robot.stop_rt_control()
"""

import numpy as np
import time
import math
import os
from typing import Tuple, Optional, List
from dataclasses import dataclass, field

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


@dataclass
class RobotStateRt:
    """실시간 로봇 상태 데이터 (OSC 제어용)

    Doosan ReadDataRt 서비스의 응답을 파이썬 객체로 변환한 것.
    모든 각도는 라디안, 위치는 미터 단위로 변환됨.
    """
    # 타임스탬프
    time_stamp: float = 0.0

    # 관절 상태 (라디안, 라디안/초)
    joint_position: np.ndarray = field(default_factory=lambda: np.zeros(6))
    joint_velocity: np.ndarray = field(default_factory=lambda: np.zeros(6))

    # TCP 상태 (미터, 라디안)
    tcp_position: np.ndarray = field(default_factory=lambda: np.zeros(6))  # [x,y,z,a,b,c]
    tcp_velocity: np.ndarray = field(default_factory=lambda: np.zeros(6))

    # 토크 정보 (Nm)
    gravity_torque: np.ndarray = field(default_factory=lambda: np.zeros(6))
    actual_joint_torque: np.ndarray = field(default_factory=lambda: np.zeros(6))
    external_joint_torque: np.ndarray = field(default_factory=lambda: np.zeros(6))

    # 동역학 행렬 (OSC용)
    mass_matrix: np.ndarray = field(default_factory=lambda: np.zeros((6, 6)))
    coriolis_matrix: np.ndarray = field(default_factory=lambda: np.zeros((6, 6)))
    jacobian_matrix: np.ndarray = field(default_factory=lambda: np.zeros((6, 6)))

    # 제어 모드 (0: position, 1: torque)
    control_mode: int = 0


class DoosanRobot:
    """Doosan E0509 로봇 인터페이스 (DRFL 또는 ROS2 기반)"""

    # E0509 관절 한계 (도)
    JOINT_LIMITS_DEG = {
        'lower': [-360, -95, -135, -360, -135, -360],
        'upper': [360, 95, 135, 360, 135, 360]
    }

    # Home 자세 (도) - 학습 초기 자세와 동일
    HOME_JOINT_DEG = [0, -17.2, 45.8, 0, 90, 0]

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

        # 실시간 제어 관련
        self._rt_connected = False
        self._rt_started = False
        self._rt_state = RobotStateRt()
        self._torque_publisher = None

        # Servo 제어 관련
        self._servo_publisher = None

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
            # 조인트 이름 순서대로 정렬 (joint_1, joint_2, ..., joint_6)
            joint_names = list(msg.name)
            positions = list(msg.position)

            # 예상 순서: joint_1, joint_2, joint_3, joint_4, joint_5, joint_6
            expected_order = ['joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'joint_6']

            reordered = np.zeros(6)
            for i, expected_name in enumerate(expected_order):
                if expected_name in joint_names:
                    idx = joint_names.index(expected_name)
                    reordered[i] = positions[idx]
                elif i < len(positions):
                    reordered[i] = positions[i]  # fallback

            self._ros2_joint_pos = reordered

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

    # =========================================================================
    # 그리퍼 제어
    # =========================================================================
    def gripper_open(self) -> bool:
        """그리퍼 열기"""
        return self._gripper_command("open")

    def gripper_close(self) -> bool:
        """그리퍼 닫기"""
        return self._gripper_command("close")

    def gripper_set_position(self, position: int) -> bool:
        """그리퍼 위치 설정 (0=열림, 700=닫힘)"""
        return self._gripper_command(f"pos {position}")

    def _gripper_command(self, cmd: str) -> bool:
        """그리퍼 명령 실행 (ROS2 Trigger 서비스 사용)"""
        if self.simulation:
            print(f"[Gripper] (시뮬레이션) {cmd}")
            return True

        import subprocess

        try:
            # /dsr01/gripper/open, /dsr01/gripper/close 서비스 사용
            # gripper_service_node.py가 실행 중이어야 함
            if cmd == "open":
                service_cmd = f"ros2 service call /dsr01/gripper/open std_srvs/srv/Trigger"
            elif cmd == "close":
                service_cmd = f"ros2 service call /dsr01/gripper/close std_srvs/srv/Trigger"
            else:
                print(f"[Gripper] 알 수 없는 명령: {cmd}")
                return False

            print(f"[Gripper] {cmd} 명령 실행 중...")
            result = subprocess.run(service_cmd, shell=True, capture_output=True, text=True, timeout=15)

            if "success=True" in result.stdout or "success: true" in result.stdout.lower():
                print(f"[Gripper] {cmd} 완료")
                return True
            else:
                print(f"[Gripper] {cmd} 실패: {result.stdout[:200]}")
                return False

        except subprocess.TimeoutExpired:
            print(f"[Gripper] {cmd} 타임아웃")
            return False
        except Exception as e:
            print(f"[Gripper] {cmd} 오류: {e}")
            return False

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

        # ROS2 서비스 사용
        if self.use_ros2:
            return self._move_joint_ros2(target_deg, vel, acc, wait)

        # DRFL 사용
        try:
            DRFL.movej(target_deg, vel=vel, acc=acc)
            if wait:
                DRFL.wait_motion()
            return True
        except Exception as e:
            print(f"[Robot] Move joint failed: {e}")
            return False

    def _move_joint_ros2(self, target_deg: list, vel: float, acc: float, wait: bool,
                         blend_type: int = 1, radius: float = 20.0) -> bool:
        """ROS2 서비스로 관절 이동

        Args:
            blend_type: 0=블렌딩없음, 1=다음모션과블렌딩
            radius: 블렌딩 반경 (mm), blend_type=1일때 사용
        """
        import subprocess
        try:
            # sync_type: 0=비동기, 1=동기
            sync_type = 1 if wait else 0
            # blend_type=1, radius=20mm로 부드러운 연속 이동
            cmd = [
                'ros2', 'service', 'call',
                f'/{self.ros2_namespace}/motion/move_joint',
                'dsr_msgs2/srv/MoveJoint',
                f'{{pos: {target_deg}, vel: {vel}, acc: {acc}, time: 0.0, radius: {radius}, mode: 0, blend_type: {blend_type}, sync_type: {sync_type}}}'
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            if 'success=True' in result.stdout or 'success: true' in result.stdout.lower():
                return True
            else:
                print(f"[Robot] MoveJoint failed: {result.stdout}")
                return False
        except subprocess.TimeoutExpired:
            print("[Robot] MoveJoint timeout")
            return False
        except Exception as e:
            print(f"[Robot] MoveJoint error: {e}")
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

    def move_spline_joint(self, waypoints_rad: list, vel: float = 30, acc: float = 30,
                          time_sec: float = 0.0, wait: bool = True) -> bool:
        """
        스플라인 궤적으로 여러 웨이포인트를 부드럽게 이동

        Args:
            waypoints_rad: list of (6,) 관절 위치들 (라디안), 최대 100개
            vel: 속도 (도/초)
            acc: 가속도 (도/초^2)
            time_sec: 전체 이동 시간 (0이면 vel/acc 사용)
            wait: 완료까지 대기 여부

        Returns:
            성공 여부
        """
        if len(waypoints_rad) == 0:
            return True

        if len(waypoints_rad) > 100:
            print(f"[Robot] Spline waypoints 최대 100개, {len(waypoints_rad)}개 → 100개로 제한")
            waypoints_rad = waypoints_rad[:100]

        # ROS2 서비스 사용
        if self.use_ros2:
            return self._move_spline_joint_ros2(waypoints_rad, vel, acc, time_sec, wait)

        # DRFL이나 시뮬레이션은 순차 이동으로 대체
        for wp in waypoints_rad:
            self.move_joint(wp, vel, acc, wait=True)
        return True

    def _move_spline_joint_ros2(self, waypoints_rad: list, vel: float, acc: float,
                                 time_sec: float, wait: bool) -> bool:
        """ROS2 서비스로 스플라인 관절 이동"""
        import subprocess
        import json

        try:
            # 웨이포인트를 도 단위로 변환
            pos_arrays = []
            for wp in waypoints_rad:
                wp_deg = np.degrees(np.array(wp)).tolist()
                pos_arrays.append(wp_deg)

            pos_cnt = len(waypoints_rad)
            vel_list = [float(vel)] * 6
            acc_list = [float(acc)] * 6
            sync_type = 0 if wait else 1

            # Float64MultiArray 형식으로 변환
            # 각 웨이포인트는 {data: [j1, j2, j3, j4, j5, j6]} 형태
            pos_str_list = []
            for wp_deg in pos_arrays:
                pos_str_list.append(f"{{data: {wp_deg}}}")
            pos_str = "[" + ", ".join(pos_str_list) + "]"

            cmd = [
                'ros2', 'service', 'call',
                f'/{self.ros2_namespace}/motion/move_spline_joint',
                'dsr_msgs2/srv/MoveSplineJoint',
                f'{{pos: {pos_str}, pos_cnt: {pos_cnt}, vel: {vel_list}, acc: {acc_list}, time: {time_sec}, mode: 0, sync_type: {sync_type}}}'
            ]

            timeout = max(30, len(waypoints_rad) * 2)  # 웨이포인트 수에 따라 타임아웃 조정
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)

            if 'success=True' in result.stdout or 'success: true' in result.stdout.lower():
                return True
            else:
                print(f"[Robot] MoveSplineJoint failed: {result.stdout[:200]}")
                return False

        except subprocess.TimeoutExpired:
            print("[Robot] MoveSplineJoint timeout")
            return False
        except Exception as e:
            print(f"[Robot] MoveSplineJoint error: {e}")
            return False

    def servoj(self, target_rad: np.ndarray, vel: float = 30.0, acc: float = 30.0,
               time_sec: float = 0.0) -> bool:
        """
        Servo 모드로 관절 이동 (연속 스트리밍용, 끊김 없음)

        Args:
            target_rad: (6,) 목표 관절 위치 (라디안)
            vel: 속도 (도/초)
            acc: 가속도 (도/초^2)
            time_sec: 이동 시간 (0이면 vel/acc 사용)

        Returns:
            성공 여부
        """
        if not self.use_ros2:
            # ROS2가 아니면 일반 move_joint 사용
            return self.move_joint(target_rad, vel, acc, wait=False)

        # Servo 퍼블리셔 생성 (처음 호출시)
        if self._servo_publisher is None:
            try:
                from dsr_msgs2.msg import ServojStream
                self._servo_publisher = self._ros2_node.create_publisher(
                    ServojStream, '/servoj_stream', 10
                )
                print("[Robot] Servo publisher created: /servoj_stream")
            except Exception as e:
                print(f"[Robot] Failed to create servo publisher: {e}")
                return self.move_joint(target_rad, vel, acc, wait=False)

        try:
            from dsr_msgs2.msg import ServojStream

            # 라디안 → 도 변환
            target_deg = np.degrees(target_rad)
            vel_arr = np.full(6, float(vel))
            acc_arr = np.full(6, float(acc))

            msg = ServojStream()
            msg.pos = target_deg.astype(np.float64).tolist()
            msg.vel = vel_arr.astype(np.float64).tolist()
            msg.acc = acc_arr.astype(np.float64).tolist()
            msg.time = float(time_sec)
            msg.mode = 0  # DR_SERVO_OVERRIDE: 즉시 새 위치로

            self._servo_publisher.publish(msg)
            return True

        except Exception as e:
            print(f"[Robot] Servoj error: {e}")
            return self.move_joint(target_rad, vel, acc, wait=False)

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

    # =========================================================================
    # 실시간 토크 제어 (ROS2 전용)
    # =========================================================================

    def start_rt_control(self) -> bool:
        """
        실시간 제어 모드 시작

        실시간 제어가 시작되면 로봇은 토크 명령을 기다립니다.
        반드시 주기적으로 (1kHz 권장) set_torque()를 호출해야 합니다.

        Returns:
            성공 여부
        """
        if not self.use_ros2:
            print("[RT] 실시간 제어는 ROS2 모드에서만 지원됩니다.")
            return False

        if self._rt_started:
            print("[RT] 이미 실시간 제어가 시작되었습니다.")
            return True

        try:
            import subprocess

            # 0. 먼저 기존 연결 정리 시도 (무시 가능)
            print("[RT] 기존 RT 연결 정리 중...")
            disconnect_cmd = (
                f"source /opt/ros/humble/setup.bash && "
                f"source ~/doosan_ws/install/setup.bash && "
                f"ros2 service call /{self.ros2_namespace}/realtime/disconnect_rt_control "
                f"dsr_msgs2/srv/DisconnectRtControl"
            )
            try:
                subprocess.run(['bash', '-c', disconnect_cmd], capture_output=True, text=True, timeout=5)
            except:
                pass  # 실패해도 무시

            import time
            time.sleep(0.5)  # 잠시 대기

            # 1. 실시간 제어 연결
            if not self._rt_connected:
                print("[RT] 실시간 제어 연결 중...")
                cmd = (
                    f"source /opt/ros/humble/setup.bash && "
                    f"source ~/doosan_ws/install/setup.bash && "
                    f"ros2 service call /{self.ros2_namespace}/realtime/connect_rt_control "
                    f"dsr_msgs2/srv/ConnectRtControl '{{ip_address: \"{self.ip}\", port: 12347}}'"
                )
                result = subprocess.run(['bash', '-c', cmd], capture_output=True, text=True, timeout=15)
                if 'success: true' in result.stdout.lower() or 'success=true' in result.stdout.lower():
                    self._rt_connected = True
                    print("[RT] 실시간 제어 연결 성공")
                elif result.returncode == 0:
                    # 이미 연결되어 있을 수 있음
                    print("[RT] 실시간 제어 연결 (이미 연결됨)")
                    self._rt_connected = True
                else:
                    print(f"[RT] 연결 실패: {result.stderr}")
                    return False

            # 2. 실시간 제어 시작
            print("[RT] 실시간 제어 시작 중...")
            cmd = (
                f"source /opt/ros/humble/setup.bash && "
                f"source ~/doosan_ws/install/setup.bash && "
                f"ros2 service call /{self.ros2_namespace}/realtime/start_rt_control "
                f"dsr_msgs2/srv/StartRtControl"
            )
            result = subprocess.run(['bash', '-c', cmd], capture_output=True, text=True, timeout=10)
            if 'success: true' in result.stdout.lower() or 'success=true' in result.stdout.lower():
                self._rt_started = True
                print("[RT] 실시간 제어 시작 성공")

                # 3. 토크 퍼블리셔 생성
                self._create_torque_publisher()
                return True
            else:
                print(f"[RT] 실시간 제어 시작 실패: {result.stdout}")
                return False

        except Exception as e:
            print(f"[RT] 실시간 제어 시작 실패: {e}")
            return False

    def stop_rt_control(self) -> bool:
        """
        실시간 제어 모드 종료

        Returns:
            성공 여부
        """
        if not self._rt_started:
            print("[RT] 실시간 제어가 시작되지 않았습니다.")
            return True

        try:
            import subprocess

            print("[RT] 실시간 제어 종료 중...")
            cmd = (
                f"source /opt/ros/humble/setup.bash && "
                f"source ~/doosan_ws/install/setup.bash && "
                f"ros2 service call /{self.ros2_namespace}/realtime/stop_rt_control "
                f"dsr_msgs2/srv/StopRtControl"
            )
            result = subprocess.run(['bash', '-c', cmd], capture_output=True, text=True, timeout=10)

            self._rt_started = False
            print("[RT] 실시간 제어 종료")
            return True

        except Exception as e:
            print(f"[RT] 실시간 제어 종료 실패: {e}")
            self._rt_started = False
            return False

    def _create_torque_publisher(self):
        """토크 명령 퍼블리셔 생성"""
        if self._torque_publisher is not None:
            return

        try:
            from dsr_msgs2.msg import TorqueRtStream
            topic = f'/{self.ros2_namespace}/torque_rt_stream'
            self._torque_publisher = self._ros2_node.create_publisher(TorqueRtStream, topic, 10)
            print(f"[RT] 토크 퍼블리셔 생성: {topic}")
        except Exception as e:
            print(f"[RT] 토크 퍼블리셔 생성 실패: {e}")

    def set_torque(self, torque: np.ndarray, time_sync: float = 0.0) -> bool:
        """
        관절 토크 명령 전송

        Args:
            torque: (6,) 관절 토크 [Nm]
            time_sync: 동기화 시간 (기본값 0.0)

        Returns:
            성공 여부
        """
        if not self._rt_started:
            print("[RT] 실시간 제어가 시작되지 않았습니다. start_rt_control()을 먼저 호출하세요.")
            return False

        if self._torque_publisher is None:
            self._create_torque_publisher()
            if self._torque_publisher is None:
                return False

        try:
            from dsr_msgs2.msg import TorqueRtStream
            msg = TorqueRtStream()
            msg.tor = [float(t) for t in torque]
            msg.time = float(time_sync)
            self._torque_publisher.publish(msg)
            return True
        except Exception as e:
            print(f"[RT] 토크 명령 전송 실패: {e}")
            return False

    def read_rt_state(self) -> Optional[RobotStateRt]:
        """
        실시간 로봇 상태 읽기

        Returns:
            RobotStateRt: 로봇 상태 (mass_matrix, jacobian, gravity_torque 등 포함)
            None: 실패 시
        """
        if not self.use_ros2:
            print("[RT] 실시간 상태 읽기는 ROS2 모드에서만 지원됩니다.")
            return None

        try:
            import subprocess
            import re

            cmd = (
                f"source /opt/ros/humble/setup.bash && "
                f"source ~/doosan_ws/install/setup.bash && "
                f"ros2 service call /{self.ros2_namespace}/realtime/read_data_rt "
                f"dsr_msgs2/srv/ReadDataRt"
            )
            result = subprocess.run(['bash', '-c', cmd], capture_output=True, text=True, timeout=5)

            if result.returncode != 0:
                return None

            # 응답 파싱
            state = RobotStateRt()
            output = result.stdout

            # 타임스탬프
            match = re.search(r'time_stamp:\s*([\d.]+)', output)
            if match:
                state.time_stamp = float(match.group(1))

            # 관절 위치 (deg -> rad)
            match = re.search(r'actual_joint_position:\s*\[([^\]]+)\]', output)
            if match:
                vals = [float(x) for x in match.group(1).split(',')]
                state.joint_position = np.radians(vals)

            # 관절 속도 (deg/s -> rad/s)
            match = re.search(r'actual_joint_velocity:\s*\[([^\]]+)\]', output)
            if match:
                vals = [float(x) for x in match.group(1).split(',')]
                state.joint_velocity = np.radians(vals)

            # 중력 토크
            match = re.search(r'gravity_torque:\s*\[([^\]]+)\]', output)
            if match:
                vals = [float(x) for x in match.group(1).split(',')]
                state.gravity_torque = np.array(vals)

            # 실제 관절 토크
            match = re.search(r'actual_joint_torque:\s*\[([^\]]+)\]', output)
            if match:
                vals = [float(x) for x in match.group(1).split(',')]
                state.actual_joint_torque = np.array(vals)

            # 외부 관절 토크
            match = re.search(r'external_joint_torque:\s*\[([^\]]+)\]', output)
            if match:
                vals = [float(x) for x in match.group(1).split(',')]
                state.external_joint_torque = np.array(vals)

            # TCP 위치 (mm -> m, deg -> rad)
            match = re.search(r'actual_tcp_position:\s*\[([^\]]+)\]', output)
            if match:
                vals = [float(x) for x in match.group(1).split(',')]
                # [x, y, z, a, b, c] -> [m, m, m, rad, rad, rad]
                state.tcp_position = np.array([
                    vals[0] / 1000, vals[1] / 1000, vals[2] / 1000,
                    np.radians(vals[3]), np.radians(vals[4]), np.radians(vals[5])
                ])

            # 제어 모드
            match = re.search(r'control_mode:\s*(\d+)', output)
            if match:
                state.control_mode = int(match.group(1))

            # TODO: mass_matrix, coriolis_matrix, jacobian_matrix 파싱
            # 이들은 Float64MultiArray 형식이라 파싱이 복잡함

            self._rt_state = state
            return state

        except Exception as e:
            print(f"[RT] 상태 읽기 실패: {e}")
            return None

    def read_rt_state_fast(self) -> Optional[RobotStateRt]:
        """
        빠른 실시간 로봇 상태 읽기 (ROS2 서비스 클라이언트 사용)

        subprocess 대신 직접 ROS2 서비스 호출.
        실시간 제어 루프에서 사용할 때 권장.

        Returns:
            RobotStateRt: 로봇 상태
            None: 실패 시
        """
        if not self.use_ros2 or self._ros2_node is None:
            return None

        try:
            from dsr_msgs2.srv import ReadDataRt

            # 서비스 클라이언트 생성 (한 번만)
            if not hasattr(self, '_read_rt_client'):
                srv_name = f'/{self.ros2_namespace}/realtime/read_data_rt'
                self._read_rt_client = self._ros2_node.create_client(ReadDataRt, srv_name)
                # 서비스 대기
                if not self._read_rt_client.wait_for_service(timeout_sec=2.0):
                    print(f"[RT] 서비스 {srv_name} 대기 시간 초과")
                    return None

            # 서비스 호출
            request = ReadDataRt.Request()
            future = self._read_rt_client.call_async(request)

            # 응답 대기 (짧은 타임아웃)
            rclpy.spin_until_future_complete(self._ros2_node, future, timeout_sec=0.1)

            if future.done():
                response = future.result()
                if response is not None:
                    return self._parse_rt_state(response.data)

            return None

        except Exception as e:
            print(f"[RT] 빠른 상태 읽기 실패: {e}")
            return None

    def _parse_rt_state(self, data) -> RobotStateRt:
        """RobotStateRt 메시지를 파이썬 객체로 변환"""
        state = RobotStateRt()

        state.time_stamp = data.time_stamp

        # 관절 상태 (deg -> rad)
        state.joint_position = np.radians(data.actual_joint_position)
        state.joint_velocity = np.radians(data.actual_joint_velocity)

        # 토크
        state.gravity_torque = np.array(data.gravity_torque)
        state.actual_joint_torque = np.array(data.actual_joint_torque)
        state.external_joint_torque = np.array(data.external_joint_torque)

        # TCP (mm -> m, deg -> rad)
        tcp = data.actual_tcp_position
        state.tcp_position = np.array([
            tcp[0] / 1000, tcp[1] / 1000, tcp[2] / 1000,
            np.radians(tcp[3]), np.radians(tcp[4]), np.radians(tcp[5])
        ])

        # 동역학 행렬
        if len(data.mass_matrix) == 6:
            state.mass_matrix = np.array([row.data for row in data.mass_matrix])
        if len(data.coriolis_matrix) == 6:
            state.coriolis_matrix = np.array([row.data for row in data.coriolis_matrix])
        if len(data.jacobian_matrix) == 6:
            state.jacobian_matrix = np.array([row.data for row in data.jacobian_matrix])

        state.control_mode = data.control_mode

        self._rt_state = state
        return state

    def get_gravity_compensation_torque(self) -> np.ndarray:
        """
        현재 자세에서의 중력 보상 토크 반환

        로봇이 현재 자세를 유지하기 위해 필요한 토크.
        토크 제어 시작 시 안전하게 시작하려면 이 값을 사용.

        Returns:
            (6,) 중력 보상 토크 [Nm]
        """
        state = self.read_rt_state()
        if state is not None:
            return state.gravity_torque.copy()
        return np.zeros(6)

    @property
    def is_rt_control_active(self) -> bool:
        """실시간 제어가 활성화되어 있는지 확인"""
        return self._rt_started


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
