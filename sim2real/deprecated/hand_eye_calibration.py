#!/usr/bin/env python3
"""
Hand-Eye Calibration for Eye-to-Hand Setup

외부 고정 카메라(RealSense D455F)와 Doosan E0509 로봇 간의
좌표계 변환 행렬을 계산합니다.

Setup:
    - 카메라: 외부 고정 (Eye-to-Hand)
    - 체커보드: 로봇 그리퍼에 부착
    - 로봇을 여러 자세로 움직이며 데이터 수집

Usage:
    1. 체커보드를 로봇 그리퍼에 부착
    2. 로봇과 카메라 연결 확인
    3. python hand_eye_calibration.py
    4. 's' 키로 현재 자세 저장 (최소 15개 이상 권장)
    5. 'c' 키로 캘리브레이션 수행
    6. 결과는 calibration_result.npz로 저장됨
"""

import numpy as np
import cv2
import time
import os
from typing import Optional, Tuple, List
from scipy.spatial.transform import Rotation as R

# RealSense
try:
    import pyrealsense2 as rs
    HAS_REALSENSE = True
except ImportError:
    print("Warning: pyrealsense2 not installed")
    HAS_REALSENSE = False

# ROS2
try:
    import rclpy
    from rclpy.node import Node
    from dsr_msgs2.srv import GetCurrentPosx
    HAS_ROS2 = True
except ImportError:
    print("Warning: ROS2 not available")
    HAS_ROS2 = False


# ==================== 설정 ====================
CHECKERBOARD_SIZE = (6, 9)  # 내부 코너 개수 (가로, 세로)
SQUARE_SIZE = 0.025  # 각 사각형 크기 (meter) = 25mm

OUTPUT_DIR = "/home/fhekwn549/doosan_ws/src/e0509_gripper_description/scripts/sim2real"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "calibration_result.npz")
DATA_FILE = os.path.join(OUTPUT_DIR, "calibration_data.npz")


class RealSenseCamera:
    """RealSense D455F 카메라 인터페이스"""

    def __init__(self):
        self.pipeline = rs.pipeline()
        self.config = rs.config()

        # 컬러 스트림 설정
        self.config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

        # 시작
        self.profile = self.pipeline.start(self.config)

        # 카메라 내부 파라미터 가져오기
        color_stream = self.profile.get_stream(rs.stream.color)
        intrinsics = color_stream.as_video_stream_profile().get_intrinsics()

        self.camera_matrix = np.array([
            [intrinsics.fx, 0, intrinsics.ppx],
            [0, intrinsics.fy, intrinsics.ppy],
            [0, 0, 1]
        ])
        self.dist_coeffs = np.array(intrinsics.coeffs)

        print(f"Camera Matrix:\n{self.camera_matrix}")
        print(f"Distortion Coefficients: {self.dist_coeffs}")

    def get_frame(self) -> np.ndarray:
        """컬러 프레임 획득"""
        frames = self.pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        return np.asanyarray(color_frame.get_data())

    def stop(self):
        self.pipeline.stop()


class MockCamera:
    """테스트용 가짜 카메라 (웹캠 사용)"""

    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise RuntimeError("카메라를 열 수 없습니다")

        # 기본 카메라 파라미터 (나중에 캘리브레이션 필요)
        self.camera_matrix = np.array([
            [800, 0, 640],
            [0, 800, 360],
            [0, 0, 1]
        ], dtype=np.float64)
        self.dist_coeffs = np.zeros(5)

    def get_frame(self) -> np.ndarray:
        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError("프레임을 읽을 수 없습니다")
        return frame

    def stop(self):
        self.cap.release()


class RobotInterface:
    """Doosan 로봇 인터페이스 (ROS2)"""

    def __init__(self, namespace='dsr01'):
        self.namespace = namespace
        self.node = rclpy.create_node('hand_eye_calibration')

        prefix = f'/{namespace}/aux_control'
        self.cli_get_posx = self.node.create_client(
            GetCurrentPosx, f'{prefix}/get_current_posx')

        print(f"서비스 연결 대기 중: {prefix}/get_current_posx")
        if not self.cli_get_posx.wait_for_service(timeout_sec=10.0):
            raise RuntimeError("로봇 서비스 연결 실패")
        print("서비스 연결됨")

    def get_tcp_pose(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        TCP pose 획득

        Returns:
            position: (3,) array [x, y, z] in meters
            rotation_matrix: (3, 3) rotation matrix
        """
        req = GetCurrentPosx.Request()
        req.ref = 0  # DR_BASE

        future = self.cli_get_posx.call_async(req)
        rclpy.spin_until_future_complete(self.node, future, timeout_sec=2.0)
        result = future.result()

        if result and result.success and len(result.task_pos_info) > 0:
            pos_data = result.task_pos_info[0].data
            # [x, y, z, rx, ry, rz] - mm, degree
            x, y, z = pos_data[0] / 1000.0, pos_data[1] / 1000.0, pos_data[2] / 1000.0
            rx, ry, rz = np.deg2rad(pos_data[3]), np.deg2rad(pos_data[4]), np.deg2rad(pos_data[5])

            position = np.array([x, y, z])

            # Doosan uses ZYX Euler angles
            rotation = R.from_euler('ZYX', [rz, ry, rx])
            rotation_matrix = rotation.as_matrix()

            return position, rotation_matrix

        raise RuntimeError("TCP pose 획득 실패")

    def shutdown(self):
        self.node.destroy_node()


class MockRobot:
    """테스트용 가짜 로봇"""

    def __init__(self):
        self._pose_idx = 0

    def get_tcp_pose(self) -> Tuple[np.ndarray, np.ndarray]:
        # 테스트용 랜덤 포즈
        position = np.array([0.3, 0.0, 0.5]) + np.random.randn(3) * 0.1
        rotation = R.from_euler('xyz', np.random.randn(3) * 0.3)
        return position, rotation.as_matrix()

    def shutdown(self):
        pass


def detect_checkerboard(image: np.ndarray, board_size: Tuple[int, int]) -> Optional[np.ndarray]:
    """
    체커보드 코너 검출

    Returns:
        corners: (N, 2) array of corner positions, or None if not found
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 체커보드 찾기
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
    ret, corners = cv2.findChessboardCorners(gray, board_size, flags)

    if ret:
        # 코너 정밀화
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        return corners.reshape(-1, 2)

    return None


def estimate_board_pose(
    corners: np.ndarray,
    board_size: Tuple[int, int],
    square_size: float,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    체커보드 pose 추정

    Returns:
        position: (3,) array - 카메라 좌표계에서 체커보드 위치
        rotation_matrix: (3, 3) - 회전 행렬
    """
    # 3D 오브젝트 포인트 생성
    objp = np.zeros((board_size[0] * board_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:board_size[0], 0:board_size[1]].T.reshape(-1, 2)
    objp *= square_size

    # PnP 풀이
    corners_2d = corners.reshape(-1, 1, 2).astype(np.float32)
    ret, rvec, tvec = cv2.solvePnP(objp, corners_2d, camera_matrix, dist_coeffs)

    if not ret:
        raise RuntimeError("PnP 풀이 실패")

    position = tvec.flatten()
    rotation_matrix, _ = cv2.Rodrigues(rvec)

    return position, rotation_matrix


def calibrate_hand_eye(
    R_gripper2base_list: List[np.ndarray],
    t_gripper2base_list: List[np.ndarray],
    R_target2cam_list: List[np.ndarray],
    t_target2cam_list: List[np.ndarray]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Eye-to-Hand 캘리브레이션

    Args:
        R_gripper2base_list: 로봇 베이스 → 그리퍼 회전 행렬들
        t_gripper2base_list: 로봇 베이스 → 그리퍼 위치들
        R_target2cam_list: 카메라 → 타겟(체커보드) 회전 행렬들
        t_target2cam_list: 카메라 → 타겟 위치들

    Returns:
        R_cam2base: 로봇 베이스 → 카메라 회전 행렬
        t_cam2base: 로봇 베이스 → 카메라 위치
    """
    # Eye-to-Hand: AX = XB 문제
    # A = gripper2base 변환 사이의 관계
    # B = target2cam 변환 사이의 관계
    # X = cam2base (우리가 구하려는 것)

    R_cam2base, t_cam2base = cv2.calibrateHandEye(
        R_gripper2base_list,
        t_gripper2base_list,
        R_target2cam_list,
        t_target2cam_list,
        method=cv2.CALIB_HAND_EYE_TSAI
    )

    return R_cam2base, t_cam2base.flatten()


def main():
    print("=" * 60)
    print("Hand-Eye Calibration (Eye-to-Hand)")
    print("=" * 60)

    # 초기화
    if HAS_REALSENSE:
        print("\nRealSense 카메라 초기화...")
        camera = RealSenseCamera()
    else:
        print("\nMock 카메라 사용 (테스트 모드)")
        camera = MockCamera()

    if HAS_ROS2:
        rclpy.init()
        print("로봇 연결 중...")
        robot = RobotInterface()
    else:
        print("Mock 로봇 사용 (테스트 모드)")
        robot = MockRobot()

    # 데이터 저장용 리스트
    R_gripper2base_list = []
    t_gripper2base_list = []
    R_target2cam_list = []
    t_target2cam_list = []

    # 기존 데이터 로드 시도
    if os.path.exists(DATA_FILE):
        print(f"\n기존 데이터 발견: {DATA_FILE}")
        data = np.load(DATA_FILE)
        R_gripper2base_list = list(data['R_gripper2base'])
        t_gripper2base_list = list(data['t_gripper2base'])
        R_target2cam_list = list(data['R_target2cam'])
        t_target2cam_list = list(data['t_target2cam'])
        print(f"  로드된 데이터: {len(R_gripper2base_list)}개")

    print("\n" + "=" * 60)
    print("조작 방법:")
    print("  's' - 현재 자세 저장")
    print("  'c' - 캘리브레이션 수행")
    print("  'd' - 마지막 데이터 삭제")
    print("  'r' - 모든 데이터 초기화")
    print("  'q' - 종료")
    print("=" * 60)
    print(f"\n최소 15개 이상의 다양한 자세가 필요합니다.")
    print("현재 저장된 데이터:", len(R_gripper2base_list), "개\n")

    cv2.namedWindow("Hand-Eye Calibration", cv2.WINDOW_NORMAL)

    try:
        while True:
            # 프레임 획득
            frame = camera.get_frame()
            display = frame.copy()

            # 체커보드 검출
            corners = detect_checkerboard(frame, CHECKERBOARD_SIZE)

            if corners is not None:
                # 검출 성공 - 초록색 표시
                cv2.drawChessboardCorners(display, CHECKERBOARD_SIZE,
                                         corners.reshape(-1, 1, 2), True)

                # 체커보드 pose 추정
                try:
                    t_board, R_board = estimate_board_pose(
                        corners, CHECKERBOARD_SIZE, SQUARE_SIZE,
                        camera.camera_matrix, camera.dist_coeffs
                    )

                    # 거리 표시
                    dist = np.linalg.norm(t_board)
                    cv2.putText(display, f"Distance: {dist:.3f}m",
                               (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                except Exception as e:
                    cv2.putText(display, f"Pose estimation failed: {e}",
                               (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            else:
                # 검출 실패 - 빨간색 표시
                cv2.putText(display, "Checkerboard not found",
                           (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # 상태 표시
            cv2.putText(display, f"Saved poses: {len(R_gripper2base_list)}",
                       (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            cv2.imshow("Hand-Eye Calibration", display)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('s'):
                # 현재 자세 저장
                if corners is not None:
                    try:
                        # 체커보드 pose (카메라 좌표계)
                        t_board, R_board = estimate_board_pose(
                            corners, CHECKERBOARD_SIZE, SQUARE_SIZE,
                            camera.camera_matrix, camera.dist_coeffs
                        )

                        # 로봇 TCP pose (로봇 베이스 좌표계)
                        t_robot, R_robot = robot.get_tcp_pose()

                        # 저장
                        R_gripper2base_list.append(R_robot)
                        t_gripper2base_list.append(t_robot)
                        R_target2cam_list.append(R_board)
                        t_target2cam_list.append(t_board)

                        # 중간 저장
                        np.savez(DATA_FILE,
                                R_gripper2base=R_gripper2base_list,
                                t_gripper2base=t_gripper2base_list,
                                R_target2cam=R_target2cam_list,
                                t_target2cam=t_target2cam_list)

                        print(f"\n[저장됨] 총 {len(R_gripper2base_list)}개")
                        print(f"  Robot: pos={t_robot}")
                        print(f"  Board: pos={t_board}")

                    except Exception as e:
                        print(f"\n[오류] 저장 실패: {e}")
                else:
                    print("\n[경고] 체커보드가 보이지 않습니다!")

            elif key == ord('d'):
                # 마지막 데이터 삭제
                if len(R_gripper2base_list) > 0:
                    R_gripper2base_list.pop()
                    t_gripper2base_list.pop()
                    R_target2cam_list.pop()
                    t_target2cam_list.pop()
                    print(f"\n[삭제됨] 남은 데이터: {len(R_gripper2base_list)}개")

            elif key == ord('r'):
                # 모든 데이터 초기화
                R_gripper2base_list = []
                t_gripper2base_list = []
                R_target2cam_list = []
                t_target2cam_list = []
                if os.path.exists(DATA_FILE):
                    os.remove(DATA_FILE)
                print("\n[초기화됨] 모든 데이터 삭제됨")

            elif key == ord('c'):
                # 캘리브레이션 수행
                if len(R_gripper2base_list) < 3:
                    print("\n[오류] 최소 3개 이상의 데이터가 필요합니다")
                    continue

                print("\n" + "=" * 60)
                print("캘리브레이션 수행 중...")

                try:
                    R_cam2base, t_cam2base = calibrate_hand_eye(
                        R_gripper2base_list,
                        t_gripper2base_list,
                        R_target2cam_list,
                        t_target2cam_list
                    )

                    # 4x4 변환 행렬 구성
                    T_cam2base = np.eye(4)
                    T_cam2base[:3, :3] = R_cam2base
                    T_cam2base[:3, 3] = t_cam2base

                    print("\n캘리브레이션 결과:")
                    print("-" * 40)
                    print("Camera -> Robot Base 변환 행렬:")
                    print(T_cam2base)

                    print(f"\n카메라 위치 (로봇 베이스 기준):")
                    print(f"  x: {t_cam2base[0]:.4f} m")
                    print(f"  y: {t_cam2base[1]:.4f} m")
                    print(f"  z: {t_cam2base[2]:.4f} m")

                    # 결과 저장
                    np.savez(OUTPUT_FILE,
                            T_cam2base=T_cam2base,
                            R_cam2base=R_cam2base,
                            t_cam2base=t_cam2base,
                            camera_matrix=camera.camera_matrix,
                            dist_coeffs=camera.dist_coeffs,
                            checkerboard_size=CHECKERBOARD_SIZE,
                            square_size=SQUARE_SIZE,
                            num_samples=len(R_gripper2base_list))

                    print(f"\n결과 저장됨: {OUTPUT_FILE}")
                    print("=" * 60)

                except Exception as e:
                    print(f"\n[오류] 캘리브레이션 실패: {e}")
                    import traceback
                    traceback.print_exc()

            elif key == ord('q'):
                break

    finally:
        cv2.destroyAllWindows()
        camera.stop()
        robot.shutdown()
        if HAS_ROS2:
            rclpy.shutdown()


if __name__ == "__main__":
    main()
