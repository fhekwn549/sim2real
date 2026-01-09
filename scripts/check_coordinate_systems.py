#!/usr/bin/env python3
"""
좌표계 확인 스크립트

카메라 좌표계와 로봇 좌표계의 축 방향을 확인합니다.
ArUco 마커를 그리퍼에 부착하고 로봇을 이동시키면서 확인하세요.

사용법:
    python check_coordinate_systems.py --marker-size 0.04

조작:
    - 로봇을 X+, X-, Y+, Y-, Z+, Z- 방향으로 이동
    - 카메라 좌표와 로봇 좌표가 어떻게 변하는지 관찰
    - 축 매핑 관계 파악

예상 결과:
    로봇 X+ (앞으로) → 카메라 Z+ (깊이 증가) 또는 Z- (깊이 감소)
    로봇 Y+ (왼쪽)  → 카메라 X- (왼쪽) 또는 X+ (오른쪽)
    로봇 Z+ (위로)  → 카메라 Y- (위로) 또는 Y+ (아래로)
"""

import numpy as np
import cv2
import argparse
import time

# RealSense
try:
    import pyrealsense2 as rs
    REALSENSE_AVAILABLE = True
except ImportError:
    REALSENSE_AVAILABLE = False
    print("[Error] pyrealsense2 not found")

# ROS2 서비스 호출용
import subprocess
import re


class CoordinateChecker:
    """좌표계 확인 도구"""

    def __init__(self, marker_size: float = 0.04, marker_id: int = 0, robot_namespace: str = "dsr01"):
        self.marker_size = marker_size
        self.marker_id = marker_id
        self.robot_namespace = robot_namespace

        # 카메라
        self.pipeline = None
        self.camera_matrix = None
        self.dist_coeffs = None

        # ArUco
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.aruco_params = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)

        # 기록용
        self.history_cam = []  # 카메라 좌표 기록
        self.history_robot = []  # 로봇 좌표 기록
        self.last_cam_pos = None
        self.last_robot_pos = None

    def start_camera(self) -> bool:
        """카메라 시작"""
        if not REALSENSE_AVAILABLE:
            return False

        try:
            self.pipeline = rs.pipeline()
            config = rs.config()
            config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

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

            print("[Camera] Started")
            print(f"  Resolution: 640x480")
            print(f"  fx={intrinsics.fx:.1f}, fy={intrinsics.fy:.1f}")
            time.sleep(0.5)
            return True

        except Exception as e:
            print(f"[Camera] Start failed: {e}")
            return False

    def stop_camera(self):
        """카메라 정지"""
        if self.pipeline:
            self.pipeline.stop()

    def get_frame(self) -> np.ndarray:
        """프레임 획득"""
        if self.pipeline is None:
            return None
        frames = self.pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            return None
        return np.asanyarray(color_frame.get_data())

    def detect_marker(self, image: np.ndarray) -> tuple:
        """ArUco 마커 감지 및 3D 위치 반환

        Returns:
            (position_cam, annotated_image) or (None, annotated_image)
            position_cam: [x, y, z] in camera frame (meters)
        """
        corners, ids, rejected = self.detector.detectMarkers(image)
        annotated = image.copy()

        if ids is None or len(ids) == 0:
            return None, annotated

        # 타겟 마커 찾기
        target_idx = None
        for i, mid in enumerate(ids.flatten()):
            if mid == self.marker_id:
                target_idx = i
                break

        if target_idx is None:
            cv2.aruco.drawDetectedMarkers(annotated, corners, ids)
            return None, annotated

        # 마커 그리기
        cv2.aruco.drawDetectedMarkers(annotated, corners, ids)

        # 3D 위치 계산 (solvePnP)
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
            return None, annotated

        # 좌표축 그리기
        cv2.drawFrameAxes(annotated, self.camera_matrix, self.dist_coeffs,
                         rvec, tvec, self.marker_size * 0.5)

        position_cam = tvec.flatten()  # [x, y, z] in camera frame
        return position_cam, annotated

    def get_robot_tcp(self) -> np.ndarray:
        """로봇 TCP 위치 읽기 (meters)"""
        try:
            bash_cmd = (
                f"source /opt/ros/humble/setup.bash && "
                f"source ~/doosan_ws/install/setup.bash && "
                f"ros2 service call /{self.robot_namespace}/aux_control/get_current_posx "
                f"dsr_msgs2/srv/GetCurrentPosx '{{ref: 0}}'"
            )

            result = subprocess.run(
                ['bash', '-c', bash_cmd],
                capture_output=True, text=True, timeout=3.0
            )

            if result.returncode == 0 and result.stdout:
                match = re.search(r'data=\[([^\]]+)\]', result.stdout)
                if match:
                    values = [float(x.strip()) for x in match.group(1).split(',')]
                    if len(values) >= 3:
                        # mm -> m
                        return np.array(values[:3]) / 1000.0
        except:
            pass

        return None

    def record_point(self, cam_pos: np.ndarray, robot_pos: np.ndarray):
        """현재 위치 기록"""
        self.history_cam.append(cam_pos.copy())
        self.history_robot.append(robot_pos.copy())
        print(f"\n[Record #{len(self.history_cam)}]")
        print(f"  Camera: X={cam_pos[0]*100:+6.1f}, Y={cam_pos[1]*100:+6.1f}, Z={cam_pos[2]*100:+6.1f} cm")
        print(f"  Robot:  X={robot_pos[0]*100:+6.1f}, Y={robot_pos[1]*100:+6.1f}, Z={robot_pos[2]*100:+6.1f} cm")

    def analyze_axis_mapping(self):
        """축 매핑 분석"""
        if len(self.history_cam) < 2:
            print("\n[분석] 최소 2개 이상의 포인트가 필요합니다.")
            return

        print("\n" + "=" * 60)
        print("  축 매핑 분석 결과")
        print("=" * 60)

        # 각 포인트 간의 변화량 분석
        for i in range(1, len(self.history_cam)):
            delta_cam = self.history_cam[i] - self.history_cam[i-1]
            delta_robot = self.history_robot[i] - self.history_robot[i-1]

            print(f"\n[이동 {i-1} → {i}]")
            print(f"  카메라 변화: dX={delta_cam[0]*100:+5.1f}, dY={delta_cam[1]*100:+5.1f}, dZ={delta_cam[2]*100:+5.1f} cm")
            print(f"  로봇 변화:   dX={delta_robot[0]*100:+5.1f}, dY={delta_robot[1]*100:+5.1f}, dZ={delta_robot[2]*100:+5.1f} cm")

            # 주요 변화 축 찾기
            cam_main_axis = np.argmax(np.abs(delta_cam))
            robot_main_axis = np.argmax(np.abs(delta_robot))

            axis_names = ['X', 'Y', 'Z']
            cam_sign = '+' if delta_cam[cam_main_axis] > 0 else '-'
            robot_sign = '+' if delta_robot[robot_main_axis] > 0 else '-'

            print(f"  → 로봇 {axis_names[robot_main_axis]}{robot_sign} 이동 시 "
                  f"카메라 {axis_names[cam_main_axis]}{cam_sign} 변화")

        print("\n" + "=" * 60)
        print("  좌표계 참고")
        print("=" * 60)
        print("  RealSense 카메라: X=오른쪽, Y=아래, Z=앞(깊이)")
        print("  Doosan 로봇:      X=앞, Y=왼쪽, Z=위")
        print("=" * 60)

    def run(self):
        """메인 루프"""
        print("\n" + "=" * 60)
        print("  좌표계 확인 도구")
        print("=" * 60)
        print(f"  마커 ID: {self.marker_id}")
        print(f"  마커 크기: {self.marker_size * 100:.1f} cm")
        print("=" * 60)
        print("  조작:")
        print("    SPACE - 현재 위치 기록")
        print("    A     - 축 매핑 분석")
        print("    R     - 기록 초기화")
        print("    Q     - 종료")
        print("=" * 60)
        print("\n로봇을 X+, X-, Y+, Y-, Z+, Z- 방향으로 이동시키면서")
        print("SPACE를 눌러 각 위치를 기록하세요.\n")

        frame_count = 0
        robot_pos = None

        while True:
            frame = self.get_frame()
            if frame is None:
                continue

            # 마커 감지
            cam_pos, display = self.detect_marker(frame)

            # 로봇 TCP 읽기 (5프레임마다 - 성능 개선)
            frame_count += 1
            if frame_count % 5 == 0 or robot_pos is None:
                robot_pos = self.get_robot_tcp()

            # 화면 표시
            y_offset = 30

            if cam_pos is not None:
                self.last_cam_pos = cam_pos
                cv2.putText(display,
                    f"Camera: X={cam_pos[0]*100:+6.1f} Y={cam_pos[1]*100:+6.1f} Z={cam_pos[2]*100:+6.1f} cm",
                    (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            else:
                cv2.putText(display, "Camera: Marker not detected",
                    (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            y_offset += 30
            if robot_pos is not None:
                self.last_robot_pos = robot_pos
                cv2.putText(display,
                    f"Robot:  X={robot_pos[0]*100:+6.1f} Y={robot_pos[1]*100:+6.1f} Z={robot_pos[2]*100:+6.1f} cm",
                    (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            else:
                cv2.putText(display, "Robot: TCP read failed",
                    (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            # 이전 위치와의 변화량 표시
            y_offset += 30
            if self.last_cam_pos is not None and cam_pos is not None and len(self.history_cam) > 0:
                delta = cam_pos - self.history_cam[-1]
                cv2.putText(display,
                    f"Delta(Cam): dX={delta[0]*100:+5.1f} dY={delta[1]*100:+5.1f} dZ={delta[2]*100:+5.1f} cm",
                    (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            y_offset += 25
            if self.last_robot_pos is not None and robot_pos is not None and len(self.history_robot) > 0:
                delta = robot_pos - self.history_robot[-1]
                cv2.putText(display,
                    f"Delta(Rob): dX={delta[0]*100:+5.1f} dY={delta[1]*100:+5.1f} dZ={delta[2]*100:+5.1f} cm",
                    (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            # 기록 수 표시
            y_offset += 30
            cv2.putText(display, f"Records: {len(self.history_cam)}",
                (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # 화면 확대
            display = cv2.resize(display, (960, 720))
            cv2.imshow("Coordinate System Check", display)

            key = cv2.waitKey(30) & 0xFF

            if key == ord('q'):
                break
            elif key == ord(' '):
                if cam_pos is not None and robot_pos is not None:
                    self.record_point(cam_pos, robot_pos)
                else:
                    print("[Error] 마커 또는 로봇 위치를 읽을 수 없습니다.")
            elif key == ord('a'):
                self.analyze_axis_mapping()
            elif key == ord('r'):
                self.history_cam.clear()
                self.history_robot.clear()
                print("\n[Reset] 기록 초기화됨")

        cv2.destroyAllWindows()
        self.stop_camera()

        # 최종 분석
        if len(self.history_cam) >= 2:
            self.analyze_axis_mapping()


def main():
    parser = argparse.ArgumentParser(description="좌표계 확인 도구")
    parser.add_argument("--marker-size", type=float, default=0.04,
                       help="마커 크기 (미터). Default: 0.04")
    parser.add_argument("--marker-id", type=int, default=0,
                       help="마커 ID. Default: 0")
    parser.add_argument("--namespace", type=str, default="dsr01",
                       help="로봇 namespace. Default: dsr01")

    args = parser.parse_args()

    checker = CoordinateChecker(
        marker_size=args.marker_size,
        marker_id=args.marker_id,
        robot_namespace=args.namespace
    )

    if not checker.start_camera():
        print("카메라 시작 실패")
        return

    checker.run()


if __name__ == "__main__":
    main()
