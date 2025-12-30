#!/usr/bin/env python3
"""
Coordinate Transformer (Eye-in-Hand)

카메라 좌표계 → 로봇 좌표계 변환

Eye-in-Hand 캘리브레이션 결과(T_cam2tcp)를 사용하여
카메라에서 감지된 물체의 위치를 로봇 좌표계로 변환합니다.

Eye-in-Hand 변환 과정:
    1. 카메라 좌표 → TCP 좌표 (T_cam2tcp, 고정 값)
    2. TCP 좌표 → 로봇 베이스 좌표 (현재 TCP pose 필요)

Usage:
    from coordinate_transformer import CoordinateTransformer

    transformer = CoordinateTransformer()  # calibration_result.npz 자동 로드

    # 현재 TCP pose를 알고 있을 때
    robot_pos = transformer.camera_to_robot(camera_pos, tcp_position, tcp_rotation)

    # 카메라 → TCP 변환만 필요할 때
    tcp_pos = transformer.camera_to_tcp(camera_pos)
"""

import numpy as np
import os
from typing import Tuple, Optional
from scipy.spatial.transform import Rotation as R


# 캘리브레이션 파일 경로 (스크립트 위치 기준 상대 경로)
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CALIBRATION_FILE = os.path.join(_SCRIPT_DIR, "config", "calibration_eye_to_hand.npz")


class CoordinateTransformer:
    """카메라 좌표계 ↔ 로봇 좌표계 변환기 (Eye-in-Hand)"""

    def __init__(self, calibration_file: str = CALIBRATION_FILE):
        """
        Args:
            calibration_file: hand_eye_calibration.py로 생성된 결과 파일
        """
        self.calibration_file = calibration_file
        self.T_cam2tcp = None  # 카메라 → TCP 변환 행렬
        self.T_tcp2cam = None  # TCP → 카메라 변환 행렬
        self.calibration_type = 'eye_in_hand'

        self._load_calibration()

    def _load_calibration(self):
        """캘리브레이션 결과 로드"""
        if not os.path.exists(self.calibration_file):
            print(f"[경고] 캘리브레이션 파일이 없습니다: {self.calibration_file}")
            print("       먼저 manual_hand_eye_calibration.py를 실행하세요.")
            print("       기본 단위 행렬을 사용합니다.")

            # 기본값 (단위 행렬)
            self.T_cam2tcp = np.eye(4)
            self.T_tcp2cam = np.eye(4)
            return

        data = np.load(self.calibration_file)

        # Eye-in-Hand 결과 로드 (T_cam2tcp)
        if 'T_cam2tcp' in data:
            self.T_cam2tcp = data['T_cam2tcp']
            self.calibration_type = data.get('calibration_type', 'eye_in_hand')
        elif 'T_cam2base' in data:
            # 호환성: 기존 Eye-to-Hand 파일도 로드 가능
            self.T_cam2tcp = data['T_cam2base']
            self.calibration_type = data.get('calibration_type', 'eye_to_hand')
            print(f"[주의] Eye-to-Hand 캘리브레이션 파일 감지됨")
            print(f"       Eye-in-Hand 사용 시 재캘리브레이션 필요")

        # 역변환 계산
        self.T_tcp2cam = np.linalg.inv(self.T_cam2tcp)

        print(f"[좌표 변환기] 캘리브레이션 로드됨: {self.calibration_file}")
        print(f"  캘리브레이션 타입: {self.calibration_type}")
        print(f"  카메라 위치 (TCP 기준):")
        print(f"    x: {self.T_cam2tcp[0, 3]*1000:.1f} mm")
        print(f"    y: {self.T_cam2tcp[1, 3]*1000:.1f} mm")
        print(f"    z: {self.T_cam2tcp[2, 3]*1000:.1f} mm")

    def camera_to_tcp(self, position_cam: np.ndarray) -> np.ndarray:
        """
        카메라 좌표계 → TCP 좌표계 변환

        Args:
            position_cam: (3,) 카메라 좌표계에서의 위치 [x, y, z]

        Returns:
            position_tcp: (3,) TCP 좌표계에서의 위치 [x, y, z]
        """
        pos_h = np.array([position_cam[0], position_cam[1], position_cam[2], 1.0])
        pos_tcp_h = self.T_cam2tcp @ pos_h
        return pos_tcp_h[:3]

    def tcp_to_camera(self, position_tcp: np.ndarray) -> np.ndarray:
        """
        TCP 좌표계 → 카메라 좌표계 변환

        Args:
            position_tcp: (3,) TCP 좌표계에서의 위치 [x, y, z]

        Returns:
            position_cam: (3,) 카메라 좌표계에서의 위치 [x, y, z]
        """
        pos_h = np.array([position_tcp[0], position_tcp[1], position_tcp[2], 1.0])
        pos_cam_h = self.T_tcp2cam @ pos_h
        return pos_cam_h[:3]

    def camera_to_robot(
        self,
        position_cam: np.ndarray,
        tcp_position: np.ndarray,
        tcp_rotation: np.ndarray
    ) -> np.ndarray:
        """
        카메라 좌표계 → 로봇 베이스 좌표계 변환 (Eye-in-Hand)

        변환 과정: 카메라 → TCP → 로봇 베이스

        Args:
            position_cam: (3,) 카메라 좌표계에서의 위치 [x, y, z]
            tcp_position: (3,) 현재 TCP 위치 (로봇 베이스 기준) [x, y, z]
            tcp_rotation: (3,3) 현재 TCP 회전 행렬 (로봇 베이스 기준)

        Returns:
            position_robot: (3,) 로봇 베이스 좌표계에서의 위치 [x, y, z]
        """
        # Step 1: 카메라 → TCP
        pos_tcp = self.camera_to_tcp(position_cam)

        # Step 2: TCP → 로봇 베이스
        # T_tcp2base = [R_tcp2base, t_tcp2base; 0, 1]
        T_tcp2base = np.eye(4)
        T_tcp2base[:3, :3] = tcp_rotation
        T_tcp2base[:3, 3] = tcp_position

        pos_tcp_h = np.array([pos_tcp[0], pos_tcp[1], pos_tcp[2], 1.0])
        pos_base_h = T_tcp2base @ pos_tcp_h

        return pos_base_h[:3]

    def robot_to_camera(
        self,
        position_robot: np.ndarray,
        tcp_position: np.ndarray,
        tcp_rotation: np.ndarray
    ) -> np.ndarray:
        """
        로봇 베이스 좌표계 → 카메라 좌표계 변환 (Eye-in-Hand)

        변환 과정: 로봇 베이스 → TCP → 카메라

        Args:
            position_robot: (3,) 로봇 베이스 좌표계에서의 위치 [x, y, z]
            tcp_position: (3,) 현재 TCP 위치 (로봇 베이스 기준) [x, y, z]
            tcp_rotation: (3,3) 현재 TCP 회전 행렬 (로봇 베이스 기준)

        Returns:
            position_cam: (3,) 카메라 좌표계에서의 위치 [x, y, z]
        """
        # Step 1: 로봇 베이스 → TCP
        T_tcp2base = np.eye(4)
        T_tcp2base[:3, :3] = tcp_rotation
        T_tcp2base[:3, 3] = tcp_position
        T_base2tcp = np.linalg.inv(T_tcp2base)

        pos_robot_h = np.array([position_robot[0], position_robot[1], position_robot[2], 1.0])
        pos_tcp_h = T_base2tcp @ pos_robot_h
        pos_tcp = pos_tcp_h[:3]

        # Step 2: TCP → 카메라
        return self.tcp_to_camera(pos_tcp)

    def transform_pose_camera_to_robot(
        self,
        position_cam: np.ndarray,
        quaternion_cam: np.ndarray,
        tcp_position: np.ndarray,
        tcp_rotation: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        카메라 좌표계의 pose를 로봇 좌표계로 변환 (Eye-in-Hand)

        Args:
            position_cam: (3,) 카메라 좌표계에서의 위치
            quaternion_cam: (4,) 카메라 좌표계에서의 자세 (w, x, y, z)
            tcp_position: (3,) 현재 TCP 위치 (로봇 베이스 기준)
            tcp_rotation: (3,3) 현재 TCP 회전 행렬 (로봇 베이스 기준)

        Returns:
            position_robot: (3,) 로봇 좌표계에서의 위치
            quaternion_robot: (4,) 로봇 좌표계에서의 자세 (w, x, y, z)
        """
        # 위치 변환
        position_robot = self.camera_to_robot(position_cam, tcp_position, tcp_rotation)

        # 자세 변환: R_robot = R_tcp2base * R_cam2tcp * R_cam
        R_cam = R.from_quat([
            quaternion_cam[1],  # x
            quaternion_cam[2],  # y
            quaternion_cam[3],  # z
            quaternion_cam[0],  # w
        ])  # scipy는 xyzw 순서

        R_cam2tcp = R.from_matrix(self.T_cam2tcp[:3, :3])
        R_tcp2base = R.from_matrix(tcp_rotation)

        R_robot = R_tcp2base * R_cam2tcp * R_cam

        # (w, x, y, z) 순서로 반환
        quat_xyzw = R_robot.as_quat()
        quaternion_robot = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])

        return position_robot, quaternion_robot

    def get_camera_position_in_tcp(self) -> np.ndarray:
        """TCP 좌표계에서 카메라 위치 반환"""
        return self.T_cam2tcp[:3, 3].copy()

    def get_camera_rotation_in_tcp(self) -> np.ndarray:
        """TCP 좌표계에서 카메라 회전 행렬 반환"""
        return self.T_cam2tcp[:3, :3].copy()

    def get_transform_matrix(self) -> np.ndarray:
        """카메라 → TCP 변환 행렬 반환"""
        return self.T_cam2tcp.copy()


def test_transformer():
    """테스트 함수"""
    print("=" * 60)
    print("Coordinate Transformer 테스트 (Eye-in-Hand)")
    print("=" * 60)

    transformer = CoordinateTransformer()

    # 테스트 포인트: 카메라 앞 50cm
    test_pos_cam = np.array([0.0, 0.0, 0.5])

    # 가상의 TCP pose (로봇 베이스 기준)
    tcp_position = np.array([0.4, 0.0, 0.3])  # x=40cm, y=0, z=30cm
    tcp_rotation = np.eye(3)  # 회전 없음

    print(f"\n카메라 좌표계: {test_pos_cam}")
    print(f"TCP 위치: {tcp_position}")

    # 카메라 → TCP 변환
    pos_tcp = transformer.camera_to_tcp(test_pos_cam)
    print(f"TCP 좌표계:    {pos_tcp}")

    # 카메라 → 로봇 베이스 변환
    pos_robot = transformer.camera_to_robot(test_pos_cam, tcp_position, tcp_rotation)
    print(f"로봇 좌표계:   {pos_robot}")

    # 역변환 확인
    pos_cam_back = transformer.robot_to_camera(pos_robot, tcp_position, tcp_rotation)
    print(f"역변환 확인:   {pos_cam_back}")

    error = np.linalg.norm(test_pos_cam - pos_cam_back)
    print(f"변환 오차:     {error:.6f} m")

    print("=" * 60)


if __name__ == "__main__":
    test_transformer()
