#!/usr/bin/env python3
"""
간단한 Eye-to-Hand 캘리브레이션

calibrateHandEye 대신 직접 계산:
1. 축 변환은 이미 측정함 (Robot X=Cam Y, Robot Y=Cam X, Robot Z=-Cam Z)
2. 위치 오프셋만 계산하면 됨

사용법:
    python simple_calibration.py --marker-size 0.04

원리:
    P_robot = R_axes @ P_cam + t_offset

    여러 샘플에서:
    t_offset = mean(P_robot_tcp - R_axes @ P_cam_marker)
"""

import numpy as np
import cv2
import time
import argparse
import os

# RealSense
try:
    import pyrealsense2 as rs
    REALSENSE_AVAILABLE = True
except ImportError:
    REALSENSE_AVAILABLE = False
    print("[Error] pyrealsense2 not found")

# 로봇 인터페이스
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from robot_interface import DoosanRobot


# 축 변환 행렬 (실측 결과)
# Robot X+ → Camera Y+
# Robot Y+ → Camera X+
# Robot Z+ → Camera Z-
R_AXES = np.array([
    [0, 1, 0],   # robot_x = cam_y
    [1, 0, 0],   # robot_y = cam_x
    [0, 0, -1],  # robot_z = -cam_z
], dtype=np.float64)


class SimpleCalibrator:
    def __init__(self, marker_size=0.04, marker_id=0, robot_ip="192.168.137.100"):
        self.marker_size = marker_size
        self.marker_id = marker_id
        self.robot_ip = robot_ip

        # 카메라
        self.pipeline = None
        self.camera_matrix = None
        self.dist_coeffs = None

        # ArUco
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.aruco_params = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)

        # 로봇
        self.robot = None

        # 샘플 데이터
        self.cam_positions = []   # 카메라 좌표 (marker)
        self.robot_positions = [] # 로봇 좌표 (TCP)

        # 결과
        self.t_offset = None  # 위치 오프셋

    def start_camera(self):
        if not REALSENSE_AVAILABLE:
            return False

        try:
            self.pipeline = rs.pipeline()
            config = rs.config()
            config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

            profile = self.pipeline.start(config)

            color_stream = profile.get_stream(rs.stream.color)
            intrinsics = color_stream.as_video_stream_profile().get_intrinsics()

            self.camera_matrix = np.array([
                [intrinsics.fx, 0, intrinsics.ppx],
                [0, intrinsics.fy, intrinsics.ppy],
                [0, 0, 1]
            ])
            self.dist_coeffs = np.array(intrinsics.coeffs)

            print("[Camera] Started")
            time.sleep(0.5)
            return True
        except Exception as e:
            print(f"[Camera] Start failed: {e}")
            return False

    def stop_camera(self):
        if self.pipeline:
            self.pipeline.stop()

    def connect_robot(self):
        self.robot = DoosanRobot(self.robot_ip)
        return self.robot.connect()

    def disconnect_robot(self):
        if self.robot:
            self.robot.disconnect()

    def get_frame(self):
        if self.pipeline is None:
            return None
        frames = self.pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            return None
        return np.asanyarray(color_frame.get_data())

    def detect_marker(self, image):
        """마커 감지 → 3D 위치 (카메라 좌표계)"""
        corners, ids, _ = self.detector.detectMarkers(image)

        if ids is None:
            return None, None

        # 타겟 마커 찾기
        target_idx = None
        for i, mid in enumerate(ids.flatten()):
            if mid == self.marker_id:
                target_idx = i
                break

        if target_idx is None:
            return None, None

        # solvePnP로 3D 위치 계산
        marker_corners = corners[target_idx].reshape(4, 2)
        half = self.marker_size / 2
        obj_points = np.array([
            [-half,  half, 0],
            [ half,  half, 0],
            [ half, -half, 0],
            [-half, -half, 0]
        ], dtype=np.float32)

        success, rvec, tvec = cv2.solvePnP(
            obj_points, marker_corners,
            self.camera_matrix, self.dist_coeffs
        )

        if not success:
            return None, None

        return tvec.flatten(), (rvec, corners[target_idx])

    def capture_sample(self):
        """현재 위치 샘플 캡처"""
        image = self.get_frame()
        if image is None:
            print("[Error] 카메라 프레임 없음")
            return False

        cam_pos, _ = self.detect_marker(image)
        if cam_pos is None:
            print("[Error] 마커 감지 안됨")
            return False

        # 로봇 TCP
        print("[Robot] TCP 읽는 중...")
        tcp_pos, _ = self.robot.get_tcp_pose_accurate()

        self.cam_positions.append(cam_pos)
        self.robot_positions.append(tcp_pos)

        print(f"\n[Sample #{len(self.cam_positions)}]")
        print(f"  Camera: [{cam_pos[0]*100:+6.2f}, {cam_pos[1]*100:+6.2f}, {cam_pos[2]*100:+6.2f}] cm")
        print(f"  Robot:  [{tcp_pos[0]*100:+6.2f}, {tcp_pos[1]*100:+6.2f}, {tcp_pos[2]*100:+6.2f}] cm")

        # 변환 후 차이 확인
        cam_transformed = R_AXES @ cam_pos
        diff = tcp_pos - cam_transformed
        print(f"  Offset: [{diff[0]*100:+6.2f}, {diff[1]*100:+6.2f}, {diff[2]*100:+6.2f}] cm")

        return True

    def calibrate(self):
        """캘리브레이션 수행 (오프셋 계산)"""
        if len(self.cam_positions) < 3:
            print(f"[Error] 최소 3개 샘플 필요 (현재: {len(self.cam_positions)})")
            return False

        print(f"\n[Calibration] {len(self.cam_positions)}개 샘플로 계산 중...")

        # 각 샘플에서 오프셋 계산
        offsets = []
        for cam_pos, robot_pos in zip(self.cam_positions, self.robot_positions):
            cam_transformed = R_AXES @ cam_pos
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
            predicted = R_AXES @ cam_pos + self.t_offset
            error = np.linalg.norm(predicted - robot_pos)
            errors.append(error)

        mean_error = np.mean(errors) * 100  # cm
        max_error = np.max(errors) * 100

        print(f"\n[Validation]")
        print(f"  Mean error: {mean_error:.2f} cm")
        print(f"  Max error:  {max_error:.2f} cm")

        if mean_error < 5:
            print("  → 좋음!")
        elif mean_error < 10:
            print("  → 괜찮음")
        else:
            print("  → 오차 큼, 샘플 더 수집 권장")

        return True

    def save(self, path="config/simple_calibration.npz"):
        """결과 저장"""
        if self.t_offset is None:
            print("[Error] 캘리브레이션 먼저 수행하세요")
            return

        os.makedirs(os.path.dirname(path), exist_ok=True)

        # T_cam_to_base 4x4 행렬 구성
        T_cam_to_base = np.eye(4)
        T_cam_to_base[:3, :3] = R_AXES
        T_cam_to_base[:3, 3] = self.t_offset

        np.savez(
            path,
            T_cam_to_base=T_cam_to_base,
            R_axes=R_AXES,
            t_offset=self.t_offset,
            camera_matrix=self.camera_matrix,
            dist_coeffs=self.dist_coeffs,
            marker_size=self.marker_size,
            num_samples=len(self.cam_positions),
            cam_positions=np.array(self.cam_positions),
            robot_positions=np.array(self.robot_positions),
        )

        print(f"\n[Saved] {path}")
        print(f"  T_cam_to_base:\n{T_cam_to_base}")

    def transform_cam_to_robot(self, point_cam):
        """카메라 좌표 → 로봇 좌표"""
        if self.t_offset is None:
            print("[Error] 캘리브레이션 필요")
            return point_cam
        return R_AXES @ point_cam + self.t_offset

    def run(self):
        """인터랙티브 캘리브레이션"""
        print("\n" + "=" * 60)
        print("  간단한 Eye-to-Hand 캘리브레이션")
        print("=" * 60)
        print("  축 변환 (실측): Robot X=Cam Y, Robot Y=Cam X, Robot Z=-Cam Z")
        print("  → 위치 오프셋만 계산하면 됨")
        print("=" * 60)
        print("  조작:")
        print("    SPACE = 샘플 캡처")
        print("    c = 캘리브레이션 계산")
        print("    s = 결과 저장")
        print("    r = 샘플 초기화")
        print("    t = 테스트 (현재 위치 변환)")
        print("    q = 종료")
        print("=" * 60)

        while True:
            image = self.get_frame()
            if image is None:
                continue

            display = image.copy()

            # 마커 감지
            cam_pos, marker_info = self.detect_marker(image)

            if cam_pos is not None:
                rvec, corners = marker_info
                cv2.drawFrameAxes(display, self.camera_matrix, self.dist_coeffs,
                                 rvec, np.array([[cam_pos[0]], [cam_pos[1]], [cam_pos[2]]]),
                                 self.marker_size * 0.5)
                cv2.polylines(display, [corners.astype(np.int32)], True, (0, 255, 0), 2)

                cv2.putText(display,
                    f"Cam: [{cam_pos[0]*100:+.1f}, {cam_pos[1]*100:+.1f}, {cam_pos[2]*100:+.1f}] cm",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                # 변환된 좌표
                if self.t_offset is not None:
                    robot_pred = self.transform_cam_to_robot(cam_pos)
                    cv2.putText(display,
                        f"Pred: [{robot_pred[0]*100:+.1f}, {robot_pred[1]*100:+.1f}, {robot_pred[2]*100:+.1f}] cm",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            else:
                cv2.putText(display, "Marker not detected",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            # 샘플 수
            cv2.putText(display, f"Samples: {len(self.cam_positions)}",
                (10, display.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # 화면 확대
            display = cv2.resize(display, (960, 720))
            cv2.imshow("Simple Calibration", display)

            key = cv2.waitKey(30) & 0xFF

            if key == ord('q'):
                break
            elif key == ord(' '):
                self.capture_sample()
            elif key == ord('c'):
                self.calibrate()
            elif key == ord('s'):
                self.save()
            elif key == ord('r'):
                self.cam_positions.clear()
                self.robot_positions.clear()
                self.t_offset = None
                print("\n[Reset] 샘플 초기화")
            elif key == ord('t'):
                if cam_pos is not None:
                    robot_pred = self.transform_cam_to_robot(cam_pos)
                    tcp_pos, _ = self.robot.get_tcp_pose()
                    error = np.linalg.norm(robot_pred - tcp_pos) * 100
                    print(f"\n[Test]")
                    print(f"  Predicted: [{robot_pred[0]*100:+.1f}, {robot_pred[1]*100:+.1f}, {robot_pred[2]*100:+.1f}] cm")
                    print(f"  Actual:    [{tcp_pos[0]*100:+.1f}, {tcp_pos[1]*100:+.1f}, {tcp_pos[2]*100:+.1f}] cm")
                    print(f"  Error:     {error:.1f} cm")

        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--marker-size", type=float, default=0.04)
    parser.add_argument("--marker-id", type=int, default=0)
    parser.add_argument("--robot-ip", type=str, default="192.168.137.100")
    args = parser.parse_args()

    cal = SimpleCalibrator(
        marker_size=args.marker_size,
        marker_id=args.marker_id,
        robot_ip=args.robot_ip
    )

    if not cal.start_camera():
        return

    if not cal.connect_robot():
        cal.stop_camera()
        return

    try:
        cal.run()
    finally:
        cal.stop_camera()
        cal.disconnect_robot()


if __name__ == "__main__":
    main()
