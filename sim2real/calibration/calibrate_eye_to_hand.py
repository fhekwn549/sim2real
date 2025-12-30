#!/usr/bin/env python3
"""
Eye-to-Hand 캘리브레이션 스크립트

고정된 카메라와 로봇 베이스 간의 변환 행렬을 계산합니다.
ArUco 마커를 로봇 TCP(그리퍼)에 부착하고 여러 자세에서 데이터를 수집합니다.

Usage:
    python calibrate_eye_to_hand.py              # 캘리브레이션 수행
    python calibrate_eye_to_hand.py --test       # 기존 캘리브레이션 테스트
    python calibrate_eye_to_hand.py --marker-id 0 --marker-size 0.05

Output:
    config/calibration_eye_to_hand.npz
"""

import numpy as np
import cv2
import time
import argparse
import os
from typing import Optional, Tuple, List
from dataclasses import dataclass

# RealSense
try:
    import pyrealsense2 as rs
    REALSENSE_AVAILABLE = True
except ImportError:
    REALSENSE_AVAILABLE = False
    print("[Warning] pyrealsense2 not found")

# ArUco
try:
    from cv2 import aruco
    ARUCO_AVAILABLE = True
except ImportError:
    ARUCO_AVAILABLE = False
    print("[Warning] cv2.aruco not found")

# 로봇 인터페이스
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from robot_interface import DoosanRobot


@dataclass
class CalibrationConfig:
    """캘리브레이션 설정"""
    # 카메라
    camera_width: int = 640
    camera_height: int = 480
    camera_fps: int = 30

    # ArUco 마커
    marker_id: int = 0                    # 사용할 마커 ID
    marker_size: float = 0.05             # 마커 크기 (미터)
    aruco_dict_type: int = aruco.DICT_6X6_250 if ARUCO_AVAILABLE else 0

    # 데이터 수집
    min_samples: int = 3                  # 최소 샘플 수 (simple 방식은 3개면 충분)
    max_samples: int = 30                 # 최대 샘플 수

    # 로봇
    robot_ip: str = "192.168.137.100"

    # 출력
    output_path: str = "config/calibration_eye_to_hand.npz"


class ArucoDetector:
    """ArUco 마커 감지기"""

    # 시도할 딕셔너리 목록 (일반적인 것부터)
    DICT_TYPES = [
        (aruco.DICT_4X4_50, "4X4_50"),
        (aruco.DICT_4X4_100, "4X4_100"),
        (aruco.DICT_4X4_250, "4X4_250"),
        (aruco.DICT_5X5_50, "5X5_50"),
        (aruco.DICT_5X5_100, "5X5_100"),
        (aruco.DICT_5X5_250, "5X5_250"),
        (aruco.DICT_6X6_50, "6X6_50"),
        (aruco.DICT_6X6_100, "6X6_100"),
        (aruco.DICT_6X6_250, "6X6_250"),
        (aruco.DICT_7X7_50, "7X7_50"),
        (aruco.DICT_ARUCO_ORIGINAL, "ORIGINAL"),
    ] if ARUCO_AVAILABLE else []

    def __init__(self, marker_size: float, dict_type: int = None, auto_detect: bool = True):
        """
        Args:
            marker_size: 마커 한 변의 길이 (미터)
            dict_type: ArUco 사전 타입 (None이면 자동 탐지)
            auto_detect: True면 여러 딕셔너리 시도
        """
        self.marker_size = marker_size
        self.auto_detect = auto_detect and (dict_type is None)
        self.detected_dict_name = None

        if ARUCO_AVAILABLE:
            self.aruco_params = aruco.DetectorParameters()

            if dict_type is not None:
                # 지정된 딕셔너리 사용
                self.aruco_dict = aruco.getPredefinedDictionary(dict_type)
                self.detector = aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
                self.detectors = None
            else:
                # 여러 딕셔너리 준비
                self.detector = None
                self.detectors = []
                for dtype, name in self.DICT_TYPES:
                    d = aruco.getPredefinedDictionary(dtype)
                    detector = aruco.ArucoDetector(d, self.aruco_params)
                    self.detectors.append((detector, name))
        else:
            self.detector = None
            self.detectors = None

    def detect(self, color_image: np.ndarray, camera_matrix: np.ndarray,
               dist_coeffs: np.ndarray, target_id: int = 0) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        ArUco 마커 감지 및 pose 추정

        Args:
            color_image: BGR 이미지
            camera_matrix: 카메라 내부 파라미터 (3x3)
            dist_coeffs: 왜곡 계수
            target_id: 찾을 마커 ID

        Returns:
            (rvec, tvec) 또는 None
            - rvec: 회전 벡터 (Rodrigues)
            - tvec: 이동 벡터 (미터)
        """
        corners = None
        ids = None

        # 단일 딕셔너리 모드
        if self.detector is not None:
            corners, ids, rejected = self.detector.detectMarkers(color_image)
        # 여러 딕셔너리 시도 모드
        elif self.detectors is not None:
            for detector, name in self.detectors:
                corners, ids, rejected = detector.detectMarkers(color_image)
                if ids is not None and len(ids) > 0:
                    # 타겟 ID가 있는지 확인
                    if target_id in ids.flatten():
                        if self.detected_dict_name != name:
                            self.detected_dict_name = name
                            print(f"[ArUco] 딕셔너리 감지: {name}")
                        break
            else:
                return None
        else:
            return None

        if ids is None or len(ids) == 0:
            return None

        # 타겟 ID 찾기
        target_idx = None
        for i, marker_id in enumerate(ids.flatten()):
            if marker_id == target_id:
                target_idx = i
                break

        if target_idx is None:
            return None

        # Pose 추정 (solvePnP 사용)
        marker_corners = corners[target_idx].reshape(4, 2)

        # 마커 3D 좌표 (마커 중심이 원점)
        half_size = self.marker_size / 2
        object_points = np.array([
            [-half_size,  half_size, 0],
            [ half_size,  half_size, 0],
            [ half_size, -half_size, 0],
            [-half_size, -half_size, 0]
        ], dtype=np.float32)

        success, rvec, tvec = cv2.solvePnP(
            object_points, marker_corners,
            camera_matrix, dist_coeffs
        )

        if not success:
            return None

        return rvec, tvec

    def draw_marker(self, image: np.ndarray, corners: np.ndarray,
                    rvec: np.ndarray, tvec: np.ndarray,
                    camera_matrix: np.ndarray, dist_coeffs: np.ndarray) -> np.ndarray:
        """마커와 좌표축 그리기"""
        # 마커 외곽선
        cv2.polylines(image, [corners.astype(np.int32)], True, (0, 255, 0), 2)

        # 좌표축
        axis_length = self.marker_size * 0.5
        cv2.drawFrameAxes(image, camera_matrix, dist_coeffs, rvec, tvec, axis_length)

        return image


class EyeToHandCalibrator:
    """Eye-to-Hand 캘리브레이션"""

    def __init__(self, config: CalibrationConfig = None):
        self.config = config or CalibrationConfig()

        # RealSense
        self.pipeline = None
        self.align = None
        self.camera_matrix = None
        self.dist_coeffs = None

        # ArUco (자동 딕셔너리 탐지)
        self.aruco_detector = ArucoDetector(
            self.config.marker_size,
            dict_type=None,  # 자동 탐지
            auto_detect=True
        )

        # 로봇
        self.robot = None

        # 수집된 데이터
        self.robot_poses = []      # 로봇 TCP poses (R, t)
        self.marker_poses = []     # 카메라에서 본 마커 poses (R, t)
        self.cam_positions = []    # 카메라 좌표 (위치만)
        self.robot_positions = []  # 로봇 좌표 (위치만)

        # 결과 (simple 방식)
        self.t_offset = None       # 위치 오프셋
        self.T_cam_to_base = None  # 카메라→로봇 베이스 변환 (4x4)

    def start_camera(self) -> bool:
        """카메라 시작"""
        if not REALSENSE_AVAILABLE:
            print("[Calibrator] RealSense not available")
            return False

        try:
            self.pipeline = rs.pipeline()
            config = rs.config()

            config.enable_stream(
                rs.stream.color,
                self.config.camera_width, self.config.camera_height,
                rs.format.bgr8, self.config.camera_fps
            )

            profile = self.pipeline.start(config)

            # 카메라 내부 파라미터
            color_stream = profile.get_stream(rs.stream.color)
            intrinsics = color_stream.as_video_stream_profile().get_intrinsics()

            self.camera_matrix = np.array([
                [intrinsics.fx, 0, intrinsics.ppx],
                [0, intrinsics.fy, intrinsics.ppy],
                [0, 0, 1]
            ])
            self.dist_coeffs = np.array(intrinsics.coeffs)

            self.align = rs.align(rs.stream.color)

            print("[Calibrator] Camera started")
            print(f"  Camera matrix:\n{self.camera_matrix}")
            time.sleep(0.5)
            return True

        except Exception as e:
            print(f"[Calibrator] Camera start failed: {e}")
            return False

    def stop_camera(self):
        """카메라 정지"""
        if self.pipeline:
            self.pipeline.stop()
        print("[Calibrator] Camera stopped")

    def connect_robot(self) -> bool:
        """로봇 연결"""
        self.robot = DoosanRobot(self.config.robot_ip)
        return self.robot.connect()

    def disconnect_robot(self):
        """로봇 연결 해제"""
        if self.robot:
            self.robot.disconnect()

    def get_frame(self) -> Optional[np.ndarray]:
        """카메라 프레임 가져오기"""
        if self.pipeline is None:
            return None

        try:
            frames = self.pipeline.wait_for_frames(timeout_ms=1000)
            color_frame = frames.get_color_frame()

            if not color_frame:
                return None

            return np.asanyarray(color_frame.get_data())

        except Exception as e:
            print(f"[Calibrator] Get frame failed: {e}")
            return None

    def detect_marker(self, image: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """마커 감지"""
        if self.camera_matrix is None:
            return None

        return self.aruco_detector.detect(
            image, self.camera_matrix, self.dist_coeffs,
            target_id=self.config.marker_id
        )

    def capture_sample(self) -> bool:
        """
        현재 자세에서 샘플 캡처

        Returns:
            성공 여부
        """
        # 카메라에서 마커 감지
        image = self.get_frame()
        if image is None:
            print("[Calibrator] Failed to get camera frame")
            return False

        result = self.detect_marker(image)
        if result is None:
            print("[Calibrator] Marker not detected")
            return False

        rvec, tvec = result

        # 마커 pose (카메라 좌표계)
        R_marker, _ = cv2.Rodrigues(rvec)
        t_marker = tvec.flatten()

        # 로봇 TCP pose (로봇 베이스 좌표계) - 정확한 값 필요
        print("[Calibrator] TCP 읽는 중... (로봇이 정지한 상태여야 합니다)")
        tcp_pos, tcp_rot = self.robot.get_tcp_pose_accurate()

        # 저장 (기존 방식 호환)
        self.marker_poses.append((R_marker, t_marker))
        self.robot_poses.append((tcp_rot, tcp_pos))

        # simple 방식용 위치 저장
        self.cam_positions.append(t_marker.copy())
        self.robot_positions.append(tcp_pos.copy())

        # 축 변환 후 오프셋 미리 보기
        R_axes = self.get_axes_rotation_matrix()
        cam_transformed = R_axes @ t_marker
        offset = tcp_pos - cam_transformed

        print(f"\n[Sample #{len(self.robot_positions)}]")
        print(f"  Camera: [{t_marker[0]*100:+6.2f}, {t_marker[1]*100:+6.2f}, {t_marker[2]*100:+6.2f}] cm")
        print(f"  Robot:  [{tcp_pos[0]*100:+6.2f}, {tcp_pos[1]*100:+6.2f}, {tcp_pos[2]*100:+6.2f}] cm")
        print(f"  Offset: [{offset[0]*100:+6.2f}, {offset[1]*100:+6.2f}, {offset[2]*100:+6.2f}] cm")

        return True

    def calibrate(self) -> bool:
        """
        캘리브레이션 수행 (Simple 방식 - 오프셋 계산)

        원리:
            축 변환은 이미 측정됨 (Robot X=Cam Y, Robot Y=Cam X, Robot Z=-Cam Z)
            위치 오프셋만 계산하면 됨

            P_robot = R_axes @ P_cam + t_offset
            t_offset = mean(P_robot_tcp - R_axes @ P_cam_marker)
        """
        if len(self.cam_positions) < self.config.min_samples:
            print(f"[Calibrator] Not enough samples: {len(self.cam_positions)} < {self.config.min_samples}")
            return False

        print(f"\n[Calibrator] Simple 방식으로 캘리브레이션 ({len(self.cam_positions)}개 샘플)...")

        # 축 변환 행렬
        R_axes = self.get_axes_rotation_matrix()

        # 각 샘플에서 오프셋 계산
        offsets = []
        for cam_pos, robot_pos in zip(self.cam_positions, self.robot_positions):
            cam_transformed = R_axes @ cam_pos
            offset = robot_pos - cam_transformed
            offsets.append(offset)

        offsets = np.array(offsets)

        # 평균 오프셋
        self.t_offset = np.mean(offsets, axis=0)

        # 표준편차 (일관성 확인)
        std = np.std(offsets, axis=0)

        print(f"\n[Result]")
        print(f"  Offset: [{self.t_offset[0]*100:+6.2f}, {self.t_offset[1]*100:+6.2f}, {self.t_offset[2]*100:+6.2f}] cm")
        print(f"  Std:    [{std[0]*100:6.2f}, {std[1]*100:6.2f}, {std[2]*100:6.2f}] cm")

        # 검증: 변환 후 오차
        errors = []
        for cam_pos, robot_pos in zip(self.cam_positions, self.robot_positions):
            predicted = R_axes @ cam_pos + self.t_offset
            error = np.linalg.norm(predicted - robot_pos)
            errors.append(error)

        mean_error = np.mean(errors) * 100  # cm
        max_error = np.max(errors) * 100

        print(f"\n[Validation]")
        print(f"  Mean error: {mean_error:.2f} cm")
        print(f"  Max error:  {max_error:.2f} cm")

        if mean_error < 2:
            print("  → 아주 좋음!")
        elif mean_error < 5:
            print("  → 좋음")
        elif mean_error < 10:
            print("  → 괜찮음")
        else:
            print("  → 오차 큼, 샘플 더 수집하거나 축 변환 확인 권장")

        # T_cam_to_base 4x4 행렬 구성
        self.T_cam_to_base = np.eye(4)
        self.T_cam_to_base[:3, :3] = R_axes
        self.T_cam_to_base[:3, 3] = self.t_offset

        print(f"\nT_cam_to_base:\n{self.T_cam_to_base}")

        return True

    def _compute_reprojection_error_eye_to_hand(self, R_cam2base, t_cam2base,
                                                  robot_poses, marker_poses) -> float:
        """Eye-to-hand용 재투영 에러 계산

        Note: 축 변환이 이미 calibrateHandEye에서 적용되었으므로
              결과 행렬은 이미 로봇 좌표계 기준임
        """
        errors = []

        # 축 변환 행렬
        R_axes = self.get_axes_rotation_matrix()

        # T_cam_to_base 변환 행렬
        T_cam2base = np.eye(4)
        T_cam2base[:3, :3] = R_cam2base
        T_cam2base[:3, 3] = t_cam2base.flatten()

        for (R_tcp, t_tcp), (R_marker, t_marker) in zip(robot_poses, marker_poses):
            # 마커 위치 (카메라 좌표계) → 축 변환 적용
            marker_in_cam = t_marker.reshape(3)
            marker_in_cam_transformed = R_axes @ marker_in_cam

            # 마커 위치를 로봇 베이스 좌표계로 변환
            marker_in_cam_h = np.append(marker_in_cam_transformed, 1)
            marker_in_base = (T_cam2base @ marker_in_cam_h)[:3]

            # 실제 TCP 위치 (로봇 베이스 좌표계)
            tcp_in_base = t_tcp.reshape(3)

            # 위치 차이 (마커는 TCP에 붙어있으므로 비슷해야 함)
            pos_error = np.linalg.norm(marker_in_base - tcp_in_base)
            errors.append(pos_error)

        return np.mean(errors)

    def _compute_reprojection_error(self, R_cam2base, t_cam2base,
                                     R_gripper2base, t_gripper2base,
                                     R_target2cam, t_target2cam) -> float:
        """재투영 에러 계산 (레거시)"""
        errors = []

        for i in range(len(R_gripper2base)):
            # 마커 → 카메라 → 베이스
            T_marker_in_cam = np.eye(4)
            T_marker_in_cam[:3, :3] = R_target2cam[i]
            T_marker_in_cam[:3, 3] = t_target2cam[i].flatten()

            T_cam_base = np.eye(4)
            T_cam_base[:3, :3] = R_cam2base
            T_cam_base[:3, 3] = t_cam2base.flatten()

            # 마커 위치 (베이스 기준, 카메라 경유)
            marker_in_base_via_cam = T_cam_base @ T_marker_in_cam

            # 마커 위치 (베이스 기준, 로봇 TCP = 마커 위치 가정)
            T_tcp_in_base = np.eye(4)
            T_tcp_in_base[:3, :3] = R_gripper2base[i]
            T_tcp_in_base[:3, 3] = t_gripper2base[i].flatten()

            # 위치 차이
            pos_error = np.linalg.norm(
                marker_in_base_via_cam[:3, 3] - T_tcp_in_base[:3, 3]
            )
            errors.append(pos_error)

        return np.mean(errors)

    def save_calibration(self, path: str = None):
        """캘리브레이션 결과 저장 (Simple 방식)"""
        if self.t_offset is None:
            print("[Calibrator] No calibration to save")
            return

        path = path or self.config.output_path

        # 디렉토리 생성
        os.makedirs(os.path.dirname(path), exist_ok=True)

        R_axes = self.get_axes_rotation_matrix()

        np.savez(
            path,
            # Simple 방식 결과
            T_cam_to_base=self.T_cam_to_base,
            R_axes=R_axes,
            t_offset=self.t_offset,
            # 카메라 파라미터
            camera_matrix=self.camera_matrix,
            dist_coeffs=self.dist_coeffs,
            marker_size=self.config.marker_size,
            marker_id=self.config.marker_id,
            num_samples=len(self.cam_positions),
            # Raw 데이터 (재계산용)
            cam_positions=np.array(self.cam_positions) if self.cam_positions else np.array([]),
            robot_positions=np.array(self.robot_positions) if self.robot_positions else np.array([]),
        )

        print(f"\n[Saved] {path}")
        print(f"  T_cam_to_base:\n{self.T_cam_to_base}")

    def load_calibration(self, path: str = None) -> bool:
        """캘리브레이션 결과 로드 (Simple 방식)"""
        path = path or self.config.output_path

        if not os.path.exists(path):
            print(f"[Calibrator] File not found: {path}")
            return False

        data = np.load(path)
        self.T_cam_to_base = data['T_cam_to_base']
        self.camera_matrix = data['camera_matrix']
        self.dist_coeffs = data['dist_coeffs']

        # Simple 방식 데이터 로드
        if 't_offset' in data:
            self.t_offset = data['t_offset']
            print(f"[Calibrator] t_offset loaded: {self.t_offset}")

        # Raw 데이터 로드 (있으면)
        if 'cam_positions' in data and len(data['cam_positions']) > 0:
            self.cam_positions = list(data['cam_positions'])
            self.robot_positions = list(data['robot_positions'])
            print(f"[Calibrator] Raw data loaded: {len(self.cam_positions)} samples")

        print(f"[Calibrator] Loaded from {path}")
        print(f"T_cam_to_base:\n{self.T_cam_to_base}")

        return True

    def recalibrate_from_file(self, path: str = None) -> bool:
        """저장된 raw 데이터로 캘리브레이션 재계산 (Simple 방식)"""
        path = path or self.config.output_path

        if not os.path.exists(path):
            print(f"[Calibrator] File not found: {path}")
            return False

        data = np.load(path)

        # Simple 방식 raw 데이터 확인
        if 'cam_positions' not in data or len(data['cam_positions']) == 0:
            print("[Calibrator] No raw data in file - cannot recalculate")
            return False

        # Raw 데이터 로드
        self.cam_positions = list(data['cam_positions'])
        self.robot_positions = list(data['robot_positions'])

        print(f"[Calibrator] Loaded {len(self.cam_positions)} samples from file")

        # 카메라 파라미터 로드
        self.camera_matrix = data['camera_matrix']
        self.dist_coeffs = data['dist_coeffs']

        # 재계산
        return self.calibrate()

    def transform_to_robot(self, point_cam: np.ndarray) -> np.ndarray:
        """
        카메라 좌표 → 로봇 베이스 좌표 변환 (Simple 방식)

        Args:
            point_cam: (3,) 카메라 좌표계 점 [x, y, z]

        Returns:
            point_base: (3,) 로봇 베이스 좌표계 점 [x, y, z]

        공식:
            P_robot = R_axes @ P_cam + t_offset
        """
        if self.t_offset is None:
            print("[Calibrator] No calibration loaded")
            return point_cam

        R_axes = self.get_axes_rotation_matrix()
        return R_axes @ point_cam + self.t_offset

    @staticmethod
    def camera_to_robot_axes(point_cam: np.ndarray) -> np.ndarray:
        """
        카메라 좌표계 축 → 로봇 좌표계 축 변환 (회전만)

        실측 결과:
        - Robot X+ (앞) → Camera Y+ (아래)
        - Robot Y+ (왼쪽) → Camera X+ (오른쪽)
        - Robot Z+ (위) → Camera Z- (가까워짐)

        역변환 (Camera → Robot):
        - robot_x = cam_y
        - robot_y = cam_x
        - robot_z = -cam_z

        Args:
            point_cam: (3,) 카메라 좌표계 점 [x, y, z]

        Returns:
            point_robot: (3,) 로봇 좌표계 점 (축만 변환, 위치 오프셋 없음)
        """
        cam_x, cam_y, cam_z = point_cam
        robot_x = cam_y
        robot_y = cam_x
        robot_z = -cam_z
        return np.array([robot_x, robot_y, robot_z])

    @staticmethod
    def robot_to_camera_axes(point_robot: np.ndarray) -> np.ndarray:
        """
        로봇 좌표계 축 → 카메라 좌표계 축 변환 (회전만)

        실측 결과:
        - Robot X+ (앞) → Camera Y+ (아래)
        - Robot Y+ (왼쪽) → Camera X+ (오른쪽)
        - Robot Z+ (위) → Camera Z- (가까워짐)

        변환 (Robot → Camera):
        - cam_x = robot_y
        - cam_y = robot_x
        - cam_z = -robot_z

        Args:
            point_robot: (3,) 로봇 좌표계 점 [x, y, z]

        Returns:
            point_cam: (3,) 카메라 좌표계 점 (축만 변환, 위치 오프셋 없음)
        """
        robot_x, robot_y, robot_z = point_robot
        cam_x = robot_y
        cam_y = robot_x
        cam_z = -robot_z
        return np.array([cam_x, cam_y, cam_z])

    @staticmethod
    def get_axes_rotation_matrix() -> np.ndarray:
        """
        카메라 → 로봇 좌표계 축 변환 회전 행렬

        Returns:
            R: (3, 3) 회전 행렬
            robot_point = R @ camera_point

        실측 결과 (2024-12-27):
            로봇 X+ → 카메라 Z-
            로봇 Y+ → 카메라 X+
            로봇 Z+ → 카메라 Y-
        """
        # robot_x = -cam_z, robot_y = cam_x, robot_z = -cam_y
        R = np.array([
            [0,  0, -1],  # robot_x = -cam_z
            [1,  0,  0],  # robot_y = cam_x
            [0, -1,  0],  # robot_z = -cam_y
        ], dtype=np.float64)
        return R

    def interactive_capture(self, scale: float = 2.0):
        """
        인터랙티브 데이터 수집

        조작:
            Space: 현재 자세 캡처
            c: 캘리브레이션 수행
            s: 결과 저장
            r: 데이터 초기화
            q: 종료
        """
        print("\n" + "=" * 60)
        print("Eye-to-Hand 캘리브레이션 - 데이터 수집")
        print("=" * 60)
        print("1. ArUco 마커를 로봇 TCP(그리퍼)에 부착")
        print("2. 로봇을 여러 자세로 이동")
        print("3. Space 키로 각 자세에서 데이터 캡처")
        print("4. 최소 10개 이상의 샘플 수집 권장")
        print("=" * 60)
        print("조작:")
        print("  Space: 현재 자세 캡처")
        print("  c: 캘리브레이션 수행")
        print("  s: 결과 저장")
        print("  r: 데이터 초기화")
        print("  q: 종료")
        print("=" * 60)

        while True:
            # 프레임 가져오기
            image = self.get_frame()
            if image is None:
                continue

            display = image.copy()

            # 마커 감지 및 표시
            result = self.detect_marker(image)

            if result is not None:
                rvec, tvec = result
                cv2.drawFrameAxes(display, self.camera_matrix, self.dist_coeffs,
                                  rvec, tvec, self.config.marker_size * 0.5)

                # 마커 위치 표시
                t = tvec.flatten()
                dict_name = self.aruco_detector.detected_dict_name or "Unknown"
                cv2.putText(display, f"Marker [{dict_name}]: [{t[0]:.3f}, {t[1]:.3f}, {t[2]:.3f}]",
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                status_color = (0, 255, 0)
            else:
                cv2.putText(display, "Marker NOT detected (trying all dictionaries...)",
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                status_color = (0, 0, 255)

            # 샘플 수 표시
            cv2.putText(display, f"Samples: {len(self.cam_positions)}/{self.config.min_samples}",
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)

            # 로봇 TCP 위치 표시
            if self.robot and self.robot.connected:
                tcp_pos, _ = self.robot.get_tcp_pose()
                cv2.putText(display, f"TCP: [{tcp_pos[0]:.3f}, {tcp_pos[1]:.3f}, {tcp_pos[2]:.3f}]",
                           (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

            # 화면 확대
            if scale != 1.0:
                new_w = int(display.shape[1] * scale)
                new_h = int(display.shape[0] * scale)
                display = cv2.resize(display, (new_w, new_h))

            cv2.imshow("Eye-to-Hand Calibration", display)

            key = cv2.waitKey(30) & 0xFF

            if key == ord('q'):
                break
            elif key == ord(' '):  # 캡처
                if self.capture_sample():
                    print(f"  Total samples: {len(self.robot_poses)}")
            elif key == ord('c'):  # 캘리브레이션
                if self.calibrate():
                    print("Calibration successful!")
                else:
                    print("Calibration failed")
            elif key == ord('s'):  # 저장
                self.save_calibration()
            elif key == ord('r'):  # 초기화
                self.robot_poses.clear()
                self.marker_poses.clear()
                self.cam_positions.clear()
                self.robot_positions.clear()
                self.t_offset = None
                self.T_cam_to_base = None
                print("\n[Reset] 데이터 초기화됨")

        cv2.destroyAllWindows()


def print_aruco_marker(marker_id: int = 0, size_pixels: int = 200):
    """ArUco 마커 이미지 생성 및 저장"""
    if not ARUCO_AVAILABLE:
        print("ArUco not available")
        return

    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
    marker_image = aruco.generateImageMarker(aruco_dict, marker_id, size_pixels)

    filename = f"aruco_marker_{marker_id}.png"
    cv2.imwrite(filename, marker_image)
    print(f"Saved marker image: {filename}")
    print(f"  ID: {marker_id}")
    print(f"  Size: {size_pixels}x{size_pixels} pixels")
    print("\n이 이미지를 인쇄하여 로봇 그리퍼에 부착하세요.")
    print("인쇄 시 실제 크기(미터)를 측정하여 --marker-size 옵션으로 지정하세요.")


def main():
    parser = argparse.ArgumentParser(description="Eye-to-Hand 캘리브레이션")
    parser.add_argument("--robot-ip", type=str, default="192.168.137.100",
                       help="로봇 IP 주소")
    parser.add_argument("--marker-id", type=int, default=0,
                       help="ArUco 마커 ID")
    parser.add_argument("--marker-size", type=float, default=0.05,
                       help="마커 크기 (미터)")
    parser.add_argument("--output", type=str, default="config/calibration_eye_to_hand.npz",
                       help="결과 저장 경로")
    parser.add_argument("--scale", type=float, default=2.0,
                       help="화면 확대 비율")
    parser.add_argument("--test", action="store_true",
                       help="기존 캘리브레이션 테스트")
    parser.add_argument("--print-marker", action="store_true",
                       help="ArUco 마커 이미지 생성")
    parser.add_argument("--simulation", action="store_true",
                       help="시뮬레이션 모드 (로봇 없이)")

    args = parser.parse_args()

    # 마커 이미지 생성
    if args.print_marker:
        print_aruco_marker(args.marker_id)
        return

    # 설정
    config = CalibrationConfig(
        robot_ip=args.robot_ip,
        marker_id=args.marker_id,
        marker_size=args.marker_size,
        output_path=args.output
    )

    calibrator = EyeToHandCalibrator(config)

    # 테스트 모드
    if args.test:
        if not calibrator.load_calibration():
            return

        if not calibrator.start_camera():
            return

        # 로봇 연결 (실제 TCP 비교용)
        if not calibrator.connect_robot():
            print("[Warning] 로봇 연결 실패 - 실제 TCP 비교 불가")

        print("\n테스트 모드 - 마커 위치를 로봇 좌표계로 변환")
        print("변환된 Robot 좌표와 실제 TCP를 비교하세요")
        print("q 키로 종료")

        while True:
            image = calibrator.get_frame()
            if image is None:
                continue

            display = image.copy()
            result = calibrator.detect_marker(image)

            # 실제 TCP 위치 가져오기
            actual_tcp = None
            if calibrator.robot and calibrator.robot.connected:
                try:
                    actual_tcp, _ = calibrator.robot.get_tcp_pose()
                except:
                    pass

            if result is not None:
                rvec, tvec = result
                t_cam = tvec.flatten()

                # 로봇 좌표계로 변환
                t_robot = calibrator.transform_to_robot(t_cam)

                cv2.drawFrameAxes(display, calibrator.camera_matrix,
                                  calibrator.dist_coeffs, rvec, tvec, 0.03)

                cv2.putText(display, f"Cam: [{t_cam[0]:.3f}, {t_cam[1]:.3f}, {t_cam[2]:.3f}]",
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(display, f"Transformed: [{t_robot[0]:.3f}, {t_robot[1]:.3f}, {t_robot[2]:.3f}]",
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

                # 실제 TCP 표시 및 오차 계산
                if actual_tcp is not None:
                    cv2.putText(display, f"Actual TCP: [{actual_tcp[0]:.3f}, {actual_tcp[1]:.3f}, {actual_tcp[2]:.3f}]",
                               (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    # 오차 계산 (마커가 TCP에 붙어있다고 가정)
                    error = np.linalg.norm(t_robot - actual_tcp) * 1000  # mm
                    color = (0, 255, 0) if error < 50 else (0, 165, 255) if error < 100 else (0, 0, 255)
                    cv2.putText(display, f"Error: {error:.1f} mm",
                               (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            else:
                cv2.putText(display, "Marker NOT detected",
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                if actual_tcp is not None:
                    cv2.putText(display, f"Actual TCP: [{actual_tcp[0]:.3f}, {actual_tcp[1]:.3f}, {actual_tcp[2]:.3f}]",
                               (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            if args.scale != 1.0:
                new_w = int(display.shape[1] * args.scale)
                new_h = int(display.shape[0] * args.scale)
                display = cv2.resize(display, (new_w, new_h))

            cv2.imshow("Calibration Test", display)

            if cv2.waitKey(30) & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()
        calibrator.stop_camera()
        calibrator.disconnect_robot()
        return

    # 캘리브레이션 모드
    print("\n" + "=" * 60)
    print("Eye-to-Hand 캘리브레이션")
    print("=" * 60)
    print(f"로봇 IP: {args.robot_ip}")
    print(f"마커 ID: {args.marker_id}")
    print(f"마커 크기: {args.marker_size}m")
    print("=" * 60)

    # 카메라 시작
    if not calibrator.start_camera():
        return

    # 로봇 연결
    if not args.simulation:
        if not calibrator.connect_robot():
            calibrator.stop_camera()
            return
    else:
        print("[Calibrator] Simulation mode - robot not connected")
        calibrator.robot = DoosanRobot(simulation=True)
        calibrator.robot.connect()

    try:
        calibrator.interactive_capture(scale=args.scale)
    finally:
        calibrator.stop_camera()
        calibrator.disconnect_robot()

    print("\n캘리브레이션 종료")


if __name__ == "__main__":
    main()
