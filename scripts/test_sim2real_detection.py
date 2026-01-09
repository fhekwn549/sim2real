#!/usr/bin/env python3
"""
Sim2Real 펜 감지 테스트

pen_detector_advanced.py + simple_calibration.npz 연동 테스트
카메라 좌표 → 로봇 좌표 변환 확인

Usage:
    python test_sim2real_detection.py

조작:
    q: 종료
    t: 현재 펜 위치를 로봇 TCP와 비교 (로봇 연결 시)
    m: ArUco 마커 위치로 TCP 예측 vs 실제 TCP 비교 (오프셋 검증)
    s: 스크린샷 저장
"""

import numpy as np
import cv2
import time
import os
import sys

# 경로 추가
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pen_detector_advanced import AdvancedPenDetector, DetectionState
from robot_interface import DoosanRobot


class Sim2RealTester:
    """Sim2Real 테스트 - 펜 감지 + 좌표 변환"""

    def __init__(self, calibration_path: str = "config/simple_calibration.npz",
                 robot_ip: str = "192.168.137.100", connect_robot: bool = True):
        # 캘리브레이션 로드
        self.calibration_path = calibration_path
        self.T_cam_to_base = None
        self.R_axes = None
        self.t_offset = None
        self._load_calibration()

        # 펜 감지기
        self.detector = AdvancedPenDetector()

        # ArUco 마커 감지기 (오프셋 검증용)
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.aruco_params = cv2.aruco.DetectorParameters()
        self.aruco_detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
        self.marker_size = 0.04  # 4cm
        self.marker_id = 0

        # 로봇 (선택적)
        self.robot = None
        self.robot_connected = False
        if connect_robot:
            self.robot = DoosanRobot(robot_ip)
            self.robot_connected = self.robot.connect()
            if not self.robot_connected:
                print("[Warning] 로봇 연결 실패 - 좌표 변환만 테스트")

    def _load_calibration(self):
        """캘리브레이션 로드"""
        if not os.path.exists(self.calibration_path):
            print(f"[Error] 캘리브레이션 파일 없음: {self.calibration_path}")
            print("        먼저 simple_calibration.py를 실행하세요.")
            return False

        data = np.load(self.calibration_path)
        self.T_cam_to_base = data['T_cam_to_base']
        self.R_axes = self.T_cam_to_base[:3, :3]
        self.t_offset = self.T_cam_to_base[:3, 3].copy()

        # 카메라 내부 파라미터 (마커 감지용)
        if 'camera_matrix' in data:
            self.camera_matrix = data['camera_matrix']
            self.dist_coeffs = data['dist_coeffs']
        else:
            # 기본값 (RealSense D455 640x480)
            self.camera_matrix = np.array([
                [382.0, 0, 320.0],
                [0, 382.0, 240.0],
                [0, 0, 1]
            ])
            self.dist_coeffs = np.zeros(5)

        # 마커-TCP 오프셋 (그리퍼 로컬 좌표계 기준)
        # 그리퍼 로컬 좌표:
        #   X: -2.6 cm (마커가 앞면에 있으므로 앞쪽 방향)
        #   Y: 0 cm
        #   Z: +4.2 cm (손가락 방향으로)
        self.marker_offset_local = np.array([-0.026, 0.0, 0.042])

        # 기본 TCP 회전 (그리퍼가 아래를 향할 때)
        # 그리퍼 Z → 로봇 -Z, 그리퍼 X → 로봇 +X, 그리퍼 Y → 로봇 -Y
        self.R_tcp_ref = np.array([
            [1,  0,  0],
            [0, -1,  0],
            [0,  0, -1]
        ], dtype=np.float64)

        print(f"[Calibration] 로드됨: {self.calibration_path}")
        print(f"  Camera offset: [{self.t_offset[0]*100:.1f}, {self.t_offset[1]*100:.1f}, {self.t_offset[2]*100:.1f}] cm")
        print(f"  Marker offset (local): [{self.marker_offset_local[0]*100:.1f}, {self.marker_offset_local[1]*100:.1f}, {self.marker_offset_local[2]*100:.1f}] cm")

        return True

    def cam_to_robot(self, point_cam: np.ndarray, R_tcp: np.ndarray = None) -> np.ndarray:
        """카메라 좌표 → 로봇 좌표 (마커 → TCP 보정 포함)

        Args:
            point_cam: 카메라 좌표계 위치
            R_tcp: TCP 회전 행렬 (None이면 기준 회전 사용)

        Returns:
            로봇 좌표계 위치 (TCP 보정 적용됨)
        """
        if self.R_axes is None:
            return point_cam

        # 1. 카메라 → 로봇 좌표 변환 (마커 위치)
        marker_pos_robot = self.R_axes @ point_cam + self.t_offset

        # 2. 마커 → TCP 오프셋 보정 (TCP 회전 반영)
        if R_tcp is None:
            R_tcp = self.R_tcp_ref  # 기준 회전 사용

        # 오프셋 보정 (현재 비활성화 - 캘리브레이션에 이미 포함되어 있을 수 있음)
        # marker_offset_robot = R_tcp.T @ self.marker_offset_local
        # tcp_pos_robot = marker_pos_robot - marker_offset_robot

        # 보정 없이 반환 (캘리브레이션이 TCP 기준으로 되어있으면 이게 맞음)
        return marker_pos_robot

    def cam_to_robot_raw(self, point_cam: np.ndarray) -> np.ndarray:
        """카메라 좌표 → 로봇 좌표 (마커 보정 없음, 디버그용)"""
        if self.R_axes is None:
            return point_cam
        return self.R_axes @ point_cam + self.t_offset

    def detect_aruco_marker(self, image: np.ndarray) -> tuple:
        """ArUco 마커 감지 → 카메라 좌표계 3D 위치

        Returns:
            (marker_pos_cam, corners) or (None, None)
        """
        corners, ids, _ = self.aruco_detector.detectMarkers(image)

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

        return tvec.flatten(), corners[target_idx]

    def start(self) -> bool:
        """시작"""
        return self.detector.start()

    def stop(self):
        """정지"""
        self.detector.stop()
        if self.robot:
            self.robot.disconnect()

    def run(self):
        """메인 루프"""
        print("\n" + "=" * 60)
        print("  Sim2Real 펜 감지 테스트")
        print("=" * 60)
        print("  조작:")
        print("    q: 종료")
        print("    t: 펜 위치 vs TCP 비교 (로봇 연결 시)")
        print("    m: 마커 위치 → TCP 예측 vs 실제 TCP (오프셋 검증)")
        print("    s: 스크린샷 저장")
        print("=" * 60)

        frame_count = 0
        robot_tcp = None
        robot_tcp_rot = None  # TCP 회전 행렬

        while True:
            # 펜 감지
            result = self.detector.detect()

            # 로봇 TCP 읽기 (10프레임마다)
            frame_count += 1
            if self.robot_connected and frame_count % 10 == 0:
                try:
                    robot_tcp, robot_tcp_rot = self.robot.get_tcp_pose()
                except:
                    robot_tcp = None
                    robot_tcp_rot = None

            # 시각화
            color_image, _ = self.detector.get_frames()
            if color_image is None:
                continue

            display = color_image.copy()
            h, w = display.shape[:2]

            if result and result.state != DetectionState.LOST:
                # 카메라 좌표
                pos_cam = result.position
                grasp_cam = result.grasp_point

                # 로봇 좌표로 변환 (TCP 회전 반영)
                pos_robot = self.cam_to_robot(pos_cam, robot_tcp_rot)
                grasp_robot = self.cam_to_robot(grasp_cam, robot_tcp_rot)

                # 상태 색상
                if result.state == DetectionState.DETECTED:
                    color = (0, 255, 0)
                    status = "DETECTED"
                else:
                    color = (0, 165, 255)
                    status = "PREDICTED"

                # 펜 중심 표시
                cx, cy = result.pixel
                cv2.circle(display, (cx, cy), 8, color, 2)

                # 바운딩 박스
                if result.bbox:
                    x, y, bw, bh = result.bbox
                    cv2.rectangle(display, (x, y), (x+bw, y+bh), color, 2)

                # Grasp point 표시
                if hasattr(self.detector, '_pen_endpoint'):
                    ex, ey = self.detector._pen_endpoint
                    cv2.circle(display, (ex, ey), 10, (255, 100, 0), 3)

                # 정보 표시
                y_offset = 30
                cv2.putText(display, f"Status: {status}", (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                y_offset += 30
                cv2.putText(display, f"Camera: [{pos_cam[0]*100:+6.1f}, {pos_cam[1]*100:+6.1f}, {pos_cam[2]*100:+6.1f}] cm",
                           (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

                y_offset += 30
                cv2.putText(display, f"Robot:  [{pos_robot[0]*100:+6.1f}, {pos_robot[1]*100:+6.1f}, {pos_robot[2]*100:+6.1f}] cm",
                           (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                y_offset += 30
                cv2.putText(display, f"Grasp:  [{grasp_robot[0]*100:+6.1f}, {grasp_robot[1]*100:+6.1f}, {grasp_robot[2]*100:+6.1f}] cm",
                           (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 100, 0), 2)

                # 로봇 TCP 표시
                if robot_tcp is not None:
                    y_offset += 30
                    cv2.putText(display, f"TCP:    [{robot_tcp[0]*100:+6.1f}, {robot_tcp[1]*100:+6.1f}, {robot_tcp[2]*100:+6.1f}] cm",
                               (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

                # 손 가림 표시
                if result.hand_occluding:
                    cv2.putText(display, "HAND OCCLUDING", (w - 200, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                # 방향 표시
                dir_3d = result.direction
                y_offset += 30
                cv2.putText(display, f"Dir: [{dir_3d[0]:.2f}, {dir_3d[1]:.2f}, {dir_3d[2]:.2f}]",
                           (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

            else:
                cv2.putText(display, "LOST - No pen detected", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                if robot_tcp is not None:
                    cv2.putText(display, f"TCP: [{robot_tcp[0]*100:+6.1f}, {robot_tcp[1]*100:+6.1f}, {robot_tcp[2]*100:+6.1f}] cm",
                               (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

            # 화면 확대
            display = cv2.resize(display, (w * 2, h * 2))
            cv2.imshow("Sim2Real Test", display)

            # 키 입력
            key = cv2.waitKey(30) & 0xFF

            if key == ord('q'):
                break

            elif key == ord('t'):
                # TCP와 펜 위치 비교
                if result and result.state != DetectionState.LOST and self.robot_connected:
                    try:
                        tcp_pos, _ = self.robot.get_tcp_pose_accurate()
                        grasp_robot = self.cam_to_robot(result.grasp_point)

                        error = np.linalg.norm(grasp_robot - tcp_pos) * 100
                        print(f"\n[Compare]")
                        print(f"  Grasp (Robot): [{grasp_robot[0]*100:+.1f}, {grasp_robot[1]*100:+.1f}, {grasp_robot[2]*100:+.1f}] cm")
                        print(f"  TCP (Actual):  [{tcp_pos[0]*100:+.1f}, {tcp_pos[1]*100:+.1f}, {tcp_pos[2]*100:+.1f}] cm")
                        print(f"  Error: {error:.1f} cm")
                    except Exception as e:
                        print(f"[Error] TCP 읽기 실패: {e}")

            elif key == ord('m'):
                # 마커 위치로 TCP 예측 (오프셋 검증)
                if self.robot_connected:
                    try:
                        # ArUco 마커 감지
                        marker_pos_cam, marker_corners = self.detect_aruco_marker(color_image)
                        if marker_pos_cam is None:
                            print("\n[Error] ArUco 마커가 감지되지 않음")
                        else:
                            # 실제 TCP 읽기
                            tcp_actual, R_tcp = self.robot.get_tcp_pose_accurate()

                            # 마커 → 로봇 좌표 (보정 없음)
                            marker_pos_robot = self.cam_to_robot_raw(marker_pos_cam)

                            # 마커 → TCP 예측 (오프셋 보정)
                            tcp_predicted = self.cam_to_robot(marker_pos_cam, R_tcp)

                            error = np.linalg.norm(tcp_predicted - tcp_actual) * 100

                            # 디버그: 오프셋 변환 확인
                            offset_robot = R_tcp.T @ self.marker_offset_local
                            actual_offset = marker_pos_robot - tcp_actual

                            print(f"\n[Marker → TCP 검증]")
                            print(f"  Marker (Cam):   [{marker_pos_cam[0]*100:+.1f}, {marker_pos_cam[1]*100:+.1f}, {marker_pos_cam[2]*100:+.1f}] cm")
                            print(f"  Marker (Robot): [{marker_pos_robot[0]*100:+.1f}, {marker_pos_robot[1]*100:+.1f}, {marker_pos_robot[2]*100:+.1f}] cm")
                            print(f"  TCP (Predicted):[{tcp_predicted[0]*100:+.1f}, {tcp_predicted[1]*100:+.1f}, {tcp_predicted[2]*100:+.1f}] cm")
                            print(f"  TCP (Actual):   [{tcp_actual[0]*100:+.1f}, {tcp_actual[1]*100:+.1f}, {tcp_actual[2]*100:+.1f}] cm")
                            print(f"  Error: {error:.1f} cm")
                            print(f"  ---")
                            print(f"  Offset (Local): [{self.marker_offset_local[0]*100:+.1f}, {self.marker_offset_local[1]*100:+.1f}, {self.marker_offset_local[2]*100:+.1f}] cm")
                            print(f"  Offset (Robot): [{offset_robot[0]*100:+.1f}, {offset_robot[1]*100:+.1f}, {offset_robot[2]*100:+.1f}] cm")
                            print(f"  Actual Offset:  [{actual_offset[0]*100:+.1f}, {actual_offset[1]*100:+.1f}, {actual_offset[2]*100:+.1f}] cm")
                    except Exception as e:
                        print(f"[Error] 마커 검증 실패: {e}")

            elif key == ord('s'):
                # 스크린샷 저장
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filename = f"sim2real_test_{timestamp}.png"
                cv2.imwrite(filename, display)
                print(f"[Saved] {filename}")

        cv2.destroyAllWindows()


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Sim2Real 펜 감지 테스트")
    parser.add_argument("--calibration", type=str, default="config/simple_calibration.npz",
                       help="캘리브레이션 파일 경로")
    parser.add_argument("--robot-ip", type=str, default="192.168.137.100",
                       help="로봇 IP")
    parser.add_argument("--no-robot", action="store_true",
                       help="로봇 연결 없이 테스트")

    args = parser.parse_args()

    tester = Sim2RealTester(
        calibration_path=args.calibration,
        robot_ip=args.robot_ip,
        connect_robot=not args.no_robot
    )

    if not tester.start():
        print("카메라 시작 실패")
        return

    try:
        tester.run()
    finally:
        tester.stop()


if __name__ == "__main__":
    main()
