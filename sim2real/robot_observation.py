#!/usr/bin/env python3
"""
Robot Observation Collector for Sim-to-Real Transfer

Isaac Lab 학습 환경과 동일한 형태의 observation을 실제 로봇에서 수집합니다.

Observation Space (36차원):
    - joint_pos (10): 6 arm + 4 gripper
    - joint_vel (10): 6 arm + 4 gripper
    - ee_pos (3): end-effector position (grasp point)
    - pen_pos (3): pen center position (카메라 필요)
    - pen_orientation (4): pen quaternion (카메라 필요)
    - relative_ee_pen (3): ee_pos - pen_pos
    - relative_ee_cap (3): ee_pos - cap_pos

Usage:
    from robot_observation import RobotObservationCollector

    collector = RobotObservationCollector(namespace='dsr01')
    obs = collector.get_observation()  # numpy array (36,)
"""

import numpy as np
import rclpy
from rclpy.node import Node
from dsr_msgs2.srv import (
    GetCurrentPosj,
    GetCurrentVelj,
    GetCurrentPosx,
)


class RobotObservationCollector(Node):
    """실제 로봇에서 observation을 수집하는 클래스"""

    # Isaac Lab 펜 규격
    PEN_LENGTH = 0.1207  # 120.7mm

    def __init__(self, namespace='dsr01'):
        super().__init__('robot_observation_collector')

        self.namespace = namespace
        self._setup_clients()

        # 그리퍼 상태 (마지막 명령값 기억)
        self._gripper_pos = 0.0  # 0=열림, 1.1=닫힘 (joint radian)

        # 카메라 데이터 (외부에서 설정)
        self._pen_pos = np.zeros(3)
        self._pen_orientation = np.array([1.0, 0.0, 0.0, 0.0])  # quaternion (w, x, y, z)

        # TCP to grasp point offset (나중에 캘리브레이션으로 조정)
        # 그리퍼 손가락 끝까지의 거리 (대략 5cm)
        self._tcp_to_grasp_offset = np.array([0.0, 0.0, 0.05])

        self.get_logger().info(f'RobotObservationCollector 초기화 (namespace: {namespace})')

    def _setup_clients(self):
        """ROS2 서비스 클라이언트 설정"""
        prefix = f'/{self.namespace}/aux_control'

        self.cli_get_posj = self.create_client(
            GetCurrentPosj, f'{prefix}/get_current_posj')
        self.cli_get_velj = self.create_client(
            GetCurrentVelj, f'{prefix}/get_current_velj')
        self.cli_get_posx = self.create_client(
            GetCurrentPosx, f'{prefix}/get_current_posx')

    def wait_for_services(self, timeout=5.0) -> bool:
        """서비스 연결 대기"""
        services = [
            (self.cli_get_posj, 'get_current_posj'),
            (self.cli_get_velj, 'get_current_velj'),
            (self.cli_get_posx, 'get_current_posx'),
        ]

        for client, name in services:
            if not client.wait_for_service(timeout_sec=timeout):
                self.get_logger().error(f'서비스 {name} 연결 실패')
                return False

        self.get_logger().info('모든 서비스 연결됨')
        return True

    def _call_service(self, client, request):
        """동기 서비스 호출"""
        future = client.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=2.0)
        return future.result()

    # ==================== Robot State ====================

    def get_arm_joint_pos(self) -> np.ndarray:
        """로봇 팔 joint position (6,) - 단위: radian"""
        req = GetCurrentPosj.Request()
        result = self._call_service(self.cli_get_posj, req)

        if result and result.success:
            # Doosan은 degree로 반환 -> radian으로 변환
            pos_deg = np.array(result.pos)
            pos_rad = np.deg2rad(pos_deg)
            return pos_rad

        self.get_logger().warn('get_arm_joint_pos 실패')
        return np.zeros(6)

    def get_arm_joint_vel(self) -> np.ndarray:
        """로봇 팔 joint velocity (6,) - 단위: rad/s"""
        req = GetCurrentVelj.Request()
        result = self._call_service(self.cli_get_velj, req)

        if result and result.success:
            # Doosan은 deg/s로 반환 -> rad/s로 변환
            vel_deg = np.array(result.joint_speed)
            vel_rad = np.deg2rad(vel_deg)
            return vel_rad

        self.get_logger().warn('get_arm_joint_vel 실패')
        return np.zeros(6)

    def get_gripper_joint_pos(self) -> np.ndarray:
        """그리퍼 joint position (4,) - mimic joints"""
        # 4개 joint 모두 같은 값 (mimic)
        return np.full(4, self._gripper_pos)

    def get_gripper_joint_vel(self) -> np.ndarray:
        """그리퍼 joint velocity (4,) - 현재는 0으로 반환"""
        return np.zeros(4)

    def get_joint_pos(self) -> np.ndarray:
        """전체 joint position (10,) = 6 arm + 4 gripper"""
        arm_pos = self.get_arm_joint_pos()
        gripper_pos = self.get_gripper_joint_pos()
        return np.concatenate([arm_pos, gripper_pos])

    def get_joint_vel(self) -> np.ndarray:
        """전체 joint velocity (10,) = 6 arm + 4 gripper"""
        arm_vel = self.get_arm_joint_vel()
        gripper_vel = self.get_gripper_joint_vel()
        return np.concatenate([arm_vel, gripper_vel])

    def get_tcp_pos(self) -> np.ndarray:
        """TCP (Tool Center Point) position (3,) - 단위: meter"""
        req = GetCurrentPosx.Request()
        req.ref = 0  # DR_BASE (로봇 베이스 기준)

        result = self._call_service(self.cli_get_posx, req)

        if result and result.success and len(result.task_pos_info) > 0:
            # task_pos_info[0].data = [x, y, z, rx, ry, rz, solution_space]
            pos_data = result.task_pos_info[0].data
            # Doosan은 mm로 반환 -> meter로 변환
            pos_m = np.array(pos_data[:3]) / 1000.0
            return pos_m

        self.get_logger().warn('get_tcp_pos 실패')
        return np.zeros(3)

    def get_ee_pos(self) -> np.ndarray:
        """End-effector (grasp point) position (3,)

        Isaac Lab의 get_grasp_point()와 대응:
        그리퍼 손가락 중심 + 2cm 앞 지점
        """
        tcp_pos = self.get_tcp_pos()
        # TODO: TCP에서 grasp point까지의 변환 적용
        # 현재는 단순히 오프셋만 적용 (나중에 캘리브레이션 필요)
        grasp_pos = tcp_pos + self._tcp_to_grasp_offset
        return grasp_pos

    # ==================== Gripper Control ====================

    def set_gripper_pos(self, pos: float):
        """그리퍼 position 설정 (observation용 상태 업데이트)

        Args:
            pos: joint position in radian (0.0=열림, 1.1=닫힘)
        """
        self._gripper_pos = np.clip(pos, 0.0, 1.1)

    def set_gripper_stroke(self, stroke: int):
        """그리퍼 stroke로 position 설정

        Args:
            stroke: 0~700 (0=열림, 700=닫힘)
        """
        # stroke 0~700 -> joint pos 0.0~1.1
        joint_pos = (stroke / 700.0) * 1.1
        self.set_gripper_pos(joint_pos)

    # ==================== Camera Data ====================

    def set_pen_pose(self, pos: np.ndarray, orientation: np.ndarray):
        """카메라에서 받은 펜 pose 설정

        Args:
            pos: 펜 중심 위치 (3,) - 로봇 좌표계
            orientation: 펜 자세 quaternion (4,) - (w, x, y, z)
        """
        self._pen_pos = np.array(pos)
        self._pen_orientation = np.array(orientation)

    def get_pen_pos(self) -> np.ndarray:
        """펜 중심 위치 (3,)"""
        return self._pen_pos.copy()

    def get_pen_orientation(self) -> np.ndarray:
        """펜 자세 quaternion (4,) - (w, x, y, z)"""
        return self._pen_orientation.copy()

    def get_pen_cap_pos(self) -> np.ndarray:
        """펜 캡 위치 (3,)

        펜 중심 + (펜 길이/2) * z축 방향
        """
        pen_pos = self._pen_pos
        quat = self._pen_orientation  # (w, x, y, z)

        # Quaternion으로 z축 방향 계산
        qw, qx, qy, qz = quat[0], quat[1], quat[2], quat[3]

        # 로컬 z축 [0,0,1]을 quaternion으로 회전
        cap_dir_x = 2.0 * (qx * qz + qw * qy)
        cap_dir_y = 2.0 * (qy * qz - qw * qx)
        cap_dir_z = 1.0 - 2.0 * (qx * qx + qy * qy)
        cap_dir = np.array([cap_dir_x, cap_dir_y, cap_dir_z])

        # 펜 캡 위치 = 중심 + 반길이 * 방향
        cap_pos = pen_pos + (self.PEN_LENGTH / 2) * cap_dir
        return cap_pos

    # ==================== Observation ====================

    def get_observation(self) -> np.ndarray:
        """전체 observation 벡터 (36,)

        Isaac Lab과 동일한 순서:
            joint_pos (10) + joint_vel (10) + ee_pos (3) +
            pen_pos (3) + pen_orientation (4) +
            relative_ee_pen (3) + relative_ee_cap (3)
        """
        joint_pos = self.get_joint_pos()      # (10,)
        joint_vel = self.get_joint_vel()      # (10,)
        ee_pos = self.get_ee_pos()            # (3,)
        pen_pos = self.get_pen_pos()          # (3,)
        pen_orientation = self.get_pen_orientation()  # (4,)

        # 상대 위치 계산
        relative_ee_pen = pen_pos - ee_pos    # (3,)
        cap_pos = self.get_pen_cap_pos()
        relative_ee_cap = cap_pos - ee_pos    # (3,)

        obs = np.concatenate([
            joint_pos,        # 10
            joint_vel,        # 10
            ee_pos,           # 3
            pen_pos,          # 3
            pen_orientation,  # 4
            relative_ee_pen,  # 3
            relative_ee_cap,  # 3
        ])

        return obs.astype(np.float32)

    def get_robot_only_observation(self) -> dict:
        """카메라 없이 로봇 상태만 반환 (디버깅용)"""
        return {
            'joint_pos': self.get_joint_pos(),
            'joint_vel': self.get_joint_vel(),
            'ee_pos': self.get_ee_pos(),
            'tcp_pos': self.get_tcp_pos(),
        }


def main():
    """테스트용 main 함수"""
    rclpy.init()

    collector = RobotObservationCollector(namespace='dsr01')

    if not collector.wait_for_services(timeout=10.0):
        print('서비스 연결 실패. 로봇이 실행 중인지 확인하세요.')
        rclpy.shutdown()
        return

    print('=' * 60)
    print('Robot Observation Collector 테스트')
    print('=' * 60)

    # 로봇 상태만 출력
    robot_state = collector.get_robot_only_observation()

    print(f"\n[Joint Position (rad)]")
    print(f"  Arm:     {robot_state['joint_pos'][:6]}")
    print(f"  Gripper: {robot_state['joint_pos'][6:]}")

    print(f"\n[Joint Velocity (rad/s)]")
    print(f"  Arm:     {robot_state['joint_vel'][:6]}")
    print(f"  Gripper: {robot_state['joint_vel'][6:]}")

    print(f"\n[TCP Position (m)]")
    print(f"  {robot_state['tcp_pos']}")

    print(f"\n[EE Position (grasp point, m)]")
    print(f"  {robot_state['ee_pos']}")

    # 전체 observation
    print(f"\n[Full Observation Shape]")
    obs = collector.get_observation()
    print(f"  Shape: {obs.shape}")
    print(f"  dtype: {obs.dtype}")

    print('=' * 60)

    rclpy.shutdown()


if __name__ == '__main__':
    main()
