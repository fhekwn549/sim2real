#!/usr/bin/env python3
"""
수동 Eye-to-Hand 캘리브레이션 스크립트

측정된 카메라 위치와 방향을 직접 입력하여 변환 행렬을 생성합니다.
ArUco 마커 기반 자동 캘리브레이션이 실패할 때 사용합니다.

사용법:
    # 기본 값 사용 (X=0.65m, Z=0.915m, 아래를 보는 카메라)
    python manual_calibration.py

    # 위치 직접 지정
    python manual_calibration.py --x 0.65 --y 0.0 --z 0.915

    # 테스트 모드 (펜 감지 확인)
    python manual_calibration.py --test

출력:
    config/calibration_eye_to_hand.npz
"""

import numpy as np
import cv2
import argparse
import os
import time
from typing import Optional, Tuple

# RealSense
try:
    import pyrealsense2 as rs
    REALSENSE_AVAILABLE = True
except ImportError:
    REALSENSE_AVAILABLE = False
    print("[Warning] pyrealsense2 not found")


class ManualCalibration:
    """수동 캘리브레이션"""

    def __init__(self,
                 camera_pos: np.ndarray,
                 camera_orientation: str = "down_forward"):
        """
        Args:
            camera_pos: 카메라 위치 [x, y, z] 로봇 베이스 기준 (미터)
            camera_orientation: 카메라 방향
                - "down_forward": 아래를 보고, 카메라 위쪽이 로봇 뒤쪽
                - "down_backward": 아래를 보고, 카메라 위쪽이 로봇 앞쪽
                - "down_left": 아래를 보고, 카메라 위쪽이 로봇 왼쪽
                - "down_right": 아래를 보고, 카메라 위쪽이 로봇 오른쪽
        """
        self.camera_pos = camera_pos
        self.camera_orientation = camera_orientation

        # 변환 행렬 생성
        self.T_cam_to_base = self._build_transform()

        # 카메라
        self.pipeline = None
        self.camera_matrix = None
        self.dist_coeffs = None

    def _build_transform(self) -> np.ndarray:
        """카메라→로봇 변환 행렬 생성"""

        # 회전 행렬 계산
        # 카메라 좌표계: X=오른쪽, Y=아래, Z=전방(광학축)
        # 로봇 좌표계: X=전방, Y=왼쪽, Z=위쪽

        # 카메라가 아래를 보고 있으므로 카메라 Z축 = 로봇 -Z

        if self.camera_orientation == "down_forward":
            # 카메라 위쪽(-Y)이 로봇 뒤쪽(-X)
            # 카메라 X축 = 로봇 -Y (왼쪽에서 오른쪽)
            # 카메라 Y축 = 로봇 -X (뒤쪽, 아래 방향)
            # 카메라 Z축 = 로봇 -Z (아래)
            R = np.array([
                [0, -1, 0],   # cam X → robot -Y
                [-1, 0, 0],   # cam Y → robot -X
                [0, 0, -1]    # cam Z → robot -Z
            ])
        elif self.camera_orientation == "down_backward":
            # 카메라 위쪽(-Y)이 로봇 앞쪽(+X)
            # 카메라 X축 = 로봇 +Y
            # 카메라 Y축 = 로봇 +X
            # 카메라 Z축 = 로봇 -Z
            R = np.array([
                [0, 1, 0],    # cam X → robot +Y
                [1, 0, 0],    # cam Y → robot +X
                [0, 0, -1]    # cam Z → robot -Z
            ])
        elif self.camera_orientation == "down_left":
            # 카메라 위쪽(-Y)이 로봇 왼쪽(+Y)
            # 카메라 X축 = 로봇 -X
            # 카메라 Y축 = 로봇 +Y
            # 카메라 Z축 = 로봇 -Z
            R = np.array([
                [-1, 0, 0],   # cam X → robot -X
                [0, 1, 0],    # cam Y → robot +Y
                [0, 0, -1]    # cam Z → robot -Z
            ])
        elif self.camera_orientation == "down_right":
            # 카메라 위쪽(-Y)이 로봇 오른쪽(-Y)
            # 카메라 X축 = 로봇 +X
            # 카메라 Y축 = 로봇 -Y
            # 카메라 Z축 = 로봇 -Z
            R = np.array([
                [1, 0, 0],    # cam X → robot +X
                [0, -1, 0],   # cam Y → robot -Y
                [0, 0, -1]    # cam Z → robot -Z
            ])
        else:
            raise ValueError(f"Unknown camera orientation: {self.camera_orientation}")

        # 4x4 변환 행렬
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = self.camera_pos

        return T

    def transform_to_robot(self, point_cam: np.ndarray) -> np.ndarray:
        """
        카메라 좌표 → 로봇 베이스 좌표 변환

        Args:
            point_cam: (3,) 카메라 좌표계 점 [x, y, z] (미터)

        Returns:
            point_base: (3,) 로봇 베이스 좌표계 점 [x, y, z] (미터)
        """
        point_cam_h = np.append(point_cam, 1)
        point_base_h = self.T_cam_to_base @ point_cam_h
        return point_base_h[:3]

    def start_camera(self) -> bool:
        """카메라 시작"""
        if not REALSENSE_AVAILABLE:
            print("[Calibration] RealSense not available")
            return False

        try:
            self.pipeline = rs.pipeline()
            config = rs.config()

            config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
            config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

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

            # Depth scale
            depth_sensor = profile.get_device().first_depth_sensor()
            self.depth_scale = depth_sensor.get_depth_scale()

            self.align = rs.align(rs.stream.color)

            print("[Calibration] Camera started")
            time.sleep(0.5)
            return True

        except Exception as e:
            print(f"[Calibration] Camera start failed: {e}")
            return False

    def stop_camera(self):
        """카메라 정지"""
        if self.pipeline:
            self.pipeline.stop()
        print("[Calibration] Camera stopped")

    def get_frames(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """카메라 프레임 가져오기"""
        if self.pipeline is None:
            return None, None

        try:
            frames = self.pipeline.wait_for_frames(timeout_ms=1000)
            aligned = self.align.process(frames)

            color_frame = aligned.get_color_frame()
            depth_frame = aligned.get_depth_frame()

            if not color_frame or not depth_frame:
                return None, None

            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())

            return color_image, depth_image

        except Exception as e:
            print(f"[Calibration] Get frame failed: {e}")
            return None, None

    def pixel_to_camera(self, u: int, v: int, depth_image: np.ndarray) -> Optional[np.ndarray]:
        """
        픽셀 좌표 → 카메라 3D 좌표

        Args:
            u, v: 픽셀 좌표
            depth_image: 깊이 이미지

        Returns:
            (3,) 카메라 좌표계 점 또는 None
        """
        if self.camera_matrix is None:
            return None

        # 깊이 값 (미터)
        depth = depth_image[v, u] * self.depth_scale
        if depth <= 0 or depth > 3.0:  # 유효 범위
            return None

        # 역투영
        fx = self.camera_matrix[0, 0]
        fy = self.camera_matrix[1, 1]
        cx = self.camera_matrix[0, 2]
        cy = self.camera_matrix[1, 2]

        x = (u - cx) * depth / fx
        y = (v - cy) * depth / fy
        z = depth

        return np.array([x, y, z])

    def save(self, path: str = "config/calibration_eye_to_hand.npz"):
        """캘리브레이션 저장"""
        os.makedirs(os.path.dirname(path), exist_ok=True)

        np.savez(
            path,
            T_cam_to_base=self.T_cam_to_base,
            camera_matrix=self.camera_matrix,
            dist_coeffs=self.dist_coeffs,
            camera_pos=self.camera_pos,
            camera_orientation=self.camera_orientation,
            manual_calibration=True
        )

        print(f"[Calibration] Saved to {path}")

    def interactive_test(self):
        """
        인터랙티브 테스트

        클릭한 지점의 카메라 좌표와 로봇 좌표를 표시합니다.
        """
        print("\n" + "=" * 60)
        print("수동 캘리브레이션 테스트")
        print("=" * 60)
        print(f"카메라 위치 (로봇 기준): {self.camera_pos}")
        print(f"카메라 방향: {self.camera_orientation}")
        print("=" * 60)
        print("조작:")
        print("  마우스 클릭: 해당 지점의 3D 좌표 표시")
        print("  1-4: 카메라 방향 변경")
        print("       1=down_forward, 2=down_backward")
        print("       3=down_left, 4=down_right")
        print("  s: 현재 설정 저장")
        print("  q: 종료")
        print("=" * 60)

        self.click_point = None

        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                self.click_point = (x, y)

        cv2.namedWindow("Manual Calibration Test")
        cv2.setMouseCallback("Manual Calibration Test", mouse_callback)

        last_robot_pos = None

        while True:
            color_image, depth_image = self.get_frames()
            if color_image is None:
                continue

            display = color_image.copy()

            # 클릭된 지점 처리
            if self.click_point is not None:
                u, v = self.click_point

                # 카메라 좌표
                cam_pos = self.pixel_to_camera(u, v, depth_image)

                if cam_pos is not None:
                    # 로봇 좌표
                    robot_pos = self.transform_to_robot(cam_pos)
                    last_robot_pos = robot_pos

                    # 표시
                    cv2.circle(display, (u, v), 5, (0, 255, 0), -1)
                    cv2.putText(display, f"Pixel: ({u}, {v})",
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    cv2.putText(display, f"Cam: [{cam_pos[0]:.3f}, {cam_pos[1]:.3f}, {cam_pos[2]:.3f}]",
                               (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    cv2.putText(display, f"Robot: [{robot_pos[0]:.3f}, {robot_pos[1]:.3f}, {robot_pos[2]:.3f}]",
                               (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                    cv2.putText(display, f"Robot (mm): [{robot_pos[0]*1000:.0f}, {robot_pos[1]*1000:.0f}, {robot_pos[2]*1000:.0f}]",
                               (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            elif last_robot_pos is not None:
                # 마지막 클릭 정보 유지
                cv2.putText(display, f"Last Robot: [{last_robot_pos[0]:.3f}, {last_robot_pos[1]:.3f}, {last_robot_pos[2]:.3f}]",
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

            # 현재 설정 표시
            cv2.putText(display, f"Orientation: {self.camera_orientation}",
                       (10, display.shape[0] - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            cv2.putText(display, f"Cam pos: [{self.camera_pos[0]:.2f}, {self.camera_pos[1]:.2f}, {self.camera_pos[2]:.2f}]",
                       (10, display.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            # 중심점 표시
            h, w = display.shape[:2]
            cv2.drawMarker(display, (w//2, h//2), (0, 0, 255), cv2.MARKER_CROSS, 20, 1)

            cv2.imshow("Manual Calibration Test", display)

            key = cv2.waitKey(30) & 0xFF

            if key == ord('q'):
                break
            elif key == ord('s'):
                self.save()
                print("저장 완료!")
            elif key == ord('1'):
                self.camera_orientation = "down_forward"
                self.T_cam_to_base = self._build_transform()
                print(f"방향 변경: {self.camera_orientation}")
                self.click_point = None
            elif key == ord('2'):
                self.camera_orientation = "down_backward"
                self.T_cam_to_base = self._build_transform()
                print(f"방향 변경: {self.camera_orientation}")
                self.click_point = None
            elif key == ord('3'):
                self.camera_orientation = "down_left"
                self.T_cam_to_base = self._build_transform()
                print(f"방향 변경: {self.camera_orientation}")
                self.click_point = None
            elif key == ord('4'):
                self.camera_orientation = "down_right"
                self.T_cam_to_base = self._build_transform()
                print(f"방향 변경: {self.camera_orientation}")
                self.click_point = None

        cv2.destroyAllWindows()


def print_transform_matrix(T: np.ndarray, name: str = "Transform"):
    """변환 행렬 출력"""
    print(f"\n{name}:")
    print("-" * 50)

    R = T[:3, :3]
    t = T[:3, 3]

    print(f"Translation (m): [{t[0]:.4f}, {t[1]:.4f}, {t[2]:.4f}]")
    print(f"Translation (mm): [{t[0]*1000:.1f}, {t[1]*1000:.1f}, {t[2]*1000:.1f}]")
    print(f"\nRotation matrix:")
    for i in range(3):
        print(f"  [{R[i,0]:7.4f}, {R[i,1]:7.4f}, {R[i,2]:7.4f}]")

    print(f"\n4x4 Matrix:")
    for i in range(4):
        print(f"  [{T[i,0]:8.4f}, {T[i,1]:8.4f}, {T[i,2]:8.4f}, {T[i,3]:8.4f}]")


def main():
    parser = argparse.ArgumentParser(description="수동 Eye-to-Hand 캘리브레이션")
    parser.add_argument("--x", type=float, default=0.65,
                       help="카메라 X 위치 (로봇 앞 방향, 미터)")
    parser.add_argument("--y", type=float, default=0.0,
                       help="카메라 Y 위치 (로봇 왼쪽 방향, 미터)")
    parser.add_argument("--z", type=float, default=0.915,
                       help="카메라 Z 위치 (위쪽, 미터)")
    parser.add_argument("--orientation", type=str, default="down_forward",
                       choices=["down_forward", "down_backward", "down_left", "down_right"],
                       help="카메라 방향")
    parser.add_argument("--output", type=str, default="config/calibration_eye_to_hand.npz",
                       help="출력 경로")
    parser.add_argument("--test", action="store_true",
                       help="인터랙티브 테스트 모드")
    parser.add_argument("--no-save", action="store_true",
                       help="저장하지 않음 (테스트만)")

    args = parser.parse_args()

    # 카메라 위치
    camera_pos = np.array([args.x, args.y, args.z])

    print("\n" + "=" * 60)
    print("수동 Eye-to-Hand 캘리브레이션")
    print("=" * 60)
    print(f"카메라 위치 (로봇 베이스 기준):")
    print(f"  X (앞): {args.x:.3f} m ({args.x*1000:.0f} mm)")
    print(f"  Y (왼쪽): {args.y:.3f} m ({args.y*1000:.0f} mm)")
    print(f"  Z (위): {args.z:.3f} m ({args.z*1000:.0f} mm)")
    print(f"카메라 방향: {args.orientation}")
    print("=" * 60)

    # 캘리브레이션 생성
    calib = ManualCalibration(camera_pos, args.orientation)

    # 변환 행렬 출력
    print_transform_matrix(calib.T_cam_to_base, "T_cam_to_base (Camera → Robot Base)")

    # 테스트용 변환 예시
    print("\n" + "=" * 60)
    print("좌표 변환 예시")
    print("=" * 60)

    test_points_cam = [
        [0, 0, 0.5],      # 카메라 앞 50cm
        [0, 0, 0.915],    # 카메라 앞 91.5cm (바닥)
        [0.1, 0, 0.5],    # 오른쪽 10cm
        [-0.1, 0, 0.5],   # 왼쪽 10cm
    ]

    print("카메라 좌표 → 로봇 좌표:")
    for pt in test_points_cam:
        pt_cam = np.array(pt)
        pt_robot = calib.transform_to_robot(pt_cam)
        print(f"  Cam {pt} → Robot [{pt_robot[0]:.3f}, {pt_robot[1]:.3f}, {pt_robot[2]:.3f}]")

    # 인터랙티브 테스트
    if args.test:
        if not calib.start_camera():
            return

        try:
            calib.interactive_test()
        finally:
            calib.stop_camera()

    # 저장
    if not args.no_save and not args.test:
        calib.save(args.output)
        print(f"\n저장 완료: {args.output}")

    print("\n완료!")


if __name__ == "__main__":
    main()
