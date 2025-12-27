#!/usr/bin/env python3
"""
수동 Hand-Eye Calibration for Eye-in-Hand Setup

카메라가 그리퍼에 부착된 상태에서 캘리브레이션 수행
직접 로봇을 움직이고 's' 키로 데이터를 저장하는 방식

사용법:
    1. 로봇 bringup 실행 (다른 터미널)
    2. 체커보드를 고정된 위치에 배치 (테이블 위 등)
    3. python manual_hand_eye_calibration.py
    4. 로봇을 다양한 자세로 움직이면서 체커보드를 촬영 (최소 15개)
    5. 'c' 키로 캘리브레이션 수행

Eye-in-Hand 구조:
    - 카메라: 그리퍼 옆에 부착 (로봇과 함께 움직임)
    - 체커보드: 고정 위치
    - 결과: T_cam2tcp (카메라 → TCP 변환 행렬)

조건:
    - 각 축에서 최소 ±15° 이상 회전 변화 필요
    - 체커보드가 항상 완전히 보여야 함
    - 다양한 위치와 각도에서 촬영
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
IMAGE_DIR = os.path.join(OUTPUT_DIR, "calibration_images")


class RealSenseCamera:
    """RealSense D455F 카메라 인터페이스"""

    def __init__(self):
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        # USB 2.0 호환을 위해 낮은 해상도/프레임레이트 시도
        try:
            self.config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
            self.profile = self.pipeline.start(self.config)
        except RuntimeError:
            print("  1280x720 실패, 640x480으로 재시도...")
            self.config = rs.config()
            self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
            self.profile = self.pipeline.start(self.config)

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
        frames = self.pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        return np.asanyarray(color_frame.get_data())

    def stop(self):
        self.pipeline.stop()


class RobotInterface:
    """Doosan 로봇 인터페이스 (ROS2)"""

    def __init__(self, namespace='dsr01'):
        self.namespace = namespace
        self.node = rclpy.create_node('manual_hand_eye_calibration')

        self.cli_get_posx = self.node.create_client(
            GetCurrentPosx, f'/{namespace}/aux_control/get_current_posx')

        print("서비스 연결 대기 중...")
        if not self.cli_get_posx.wait_for_service(timeout_sec=10.0):
            raise RuntimeError("get_current_posx 서비스 연결 실패")
        print("서비스 연결 완료")

    def get_tcp_pose(self) -> Tuple[np.ndarray, np.ndarray]:
        """TCP pose 획득"""
        req = GetCurrentPosx.Request()
        req.ref = 0  # DR_BASE

        future = self.cli_get_posx.call_async(req)

        # 짧은 timeout으로 빠르게 응답 받기
        timeout_end = time.time() + 1.0
        while not future.done() and time.time() < timeout_end:
            rclpy.spin_once(self.node, timeout_sec=0.05)

        result = future.result()

        if result and result.success and len(result.task_pos_info) > 0:
            pos_data = result.task_pos_info[0].data
            x, y, z = pos_data[0] / 1000.0, pos_data[1] / 1000.0, pos_data[2] / 1000.0
            rx, ry, rz = np.deg2rad(pos_data[3]), np.deg2rad(pos_data[4]), np.deg2rad(pos_data[5])

            position = np.array([x, y, z])
            rotation = R.from_euler('ZYX', [rz, ry, rx])
            rotation_matrix = rotation.as_matrix()

            return position, rotation_matrix

        raise RuntimeError("TCP pose 획득 실패")

    def shutdown(self):
        self.node.destroy_node()


def detect_checkerboard(image: np.ndarray, board_size: Tuple[int, int]) -> Optional[np.ndarray]:
    """체커보드 코너 검출"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
    ret, corners = cv2.findChessboardCorners(gray, board_size, flags)

    if ret:
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
    """체커보드 pose 추정"""
    objp = np.zeros((board_size[0] * board_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:board_size[0], 0:board_size[1]].T.reshape(-1, 2)
    objp *= square_size

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
    """Eye-in-Hand 캘리브레이션

    Eye-in-Hand의 경우:
    - 카메라가 그리퍼에 부착되어 함께 움직임
    - 체커보드(타겟)는 고정
    - cv2.calibrateHandEye에 직접 입력
    - 결과: T_cam2gripper (카메라 → 그리퍼/TCP 변환)
    """
    # Eye-in-Hand: 역변환 없이 직접 사용
    R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(
        R_gripper2base_list,
        t_gripper2base_list,
        R_target2cam_list,
        t_target2cam_list,
        method=cv2.CALIB_HAND_EYE_TSAI
    )
    return R_cam2gripper, t_cam2gripper.flatten()


def compute_rotation_stats(R_list: List[np.ndarray]) -> dict:
    """회전 변화량 통계 계산"""
    if len(R_list) < 2:
        return {'total_rotation': 0, 'max_single': 0}

    angles = []
    for i in range(len(R_list)):
        for j in range(i+1, len(R_list)):
            R_diff = R_list[i].T @ R_list[j]
            angle = np.arccos(np.clip((np.trace(R_diff) - 1) / 2, -1, 1))
            angles.append(np.rad2deg(angle))

    return {
        'total_rotation': max(angles) if angles else 0,
        'max_single': max(angles) if angles else 0,
        'min_single': min(angles) if angles else 0,
        'avg': np.mean(angles) if angles else 0
    }


def main():
    print("=" * 60)
    print("  수동 Hand-Eye Calibration (Eye-in-Hand)")
    print("  카메라가 그리퍼에 부착된 구조")
    print("=" * 60)

    os.makedirs(IMAGE_DIR, exist_ok=True)

    if not HAS_REALSENSE:
        print("Error: pyrealsense2가 필요합니다")
        return

    if not HAS_ROS2:
        print("Error: ROS2가 필요합니다")
        return

    print("\nRealSense 카메라 초기화...")
    camera = RealSenseCamera()

    rclpy.init()
    print("로봇 연결 중...")
    robot = RobotInterface()

    # 데이터 저장용 리스트
    R_gripper2base_list = []
    t_gripper2base_list = []
    R_target2cam_list = []
    t_target2cam_list = []

    # 기존 데이터 로드
    if os.path.exists(DATA_FILE):
        print(f"\n기존 데이터 발견: {DATA_FILE}")
        response = input("기존 데이터를 이어서 사용할까요? (y/n): ")
        if response.lower() == 'y':
            data = np.load(DATA_FILE)
            R_gripper2base_list = list(data['R_gripper2base'])
            t_gripper2base_list = list(data['t_gripper2base'])
            R_target2cam_list = list(data['R_target2cam'])
            t_target2cam_list = list(data['t_target2cam'])
            print(f"  로드된 데이터: {len(R_gripper2base_list)}개")
        else:
            os.remove(DATA_FILE)
            for f in os.listdir(IMAGE_DIR):
                os.remove(os.path.join(IMAGE_DIR, f))
            print("기존 데이터 삭제됨")

    print("\n" + "=" * 60)
    print("조작 방법:")
    print("  's' - 현재 자세 저장")
    print("  'c' - 캘리브레이션 수행")
    print("  'd' - 마지막 데이터 삭제")
    print("  'r' - 모든 데이터 초기화")
    print("  'i' - 현재 회전 변화량 확인")
    print("  'q' - 종료")
    print("=" * 60)
    print(f"\n[Eye-in-Hand] 체커보드를 고정하고 로봇을 움직여 촬영하세요!")
    print(f"최소 15개 이상, 회전 변화 30° 이상 필요")
    print(f"현재 저장된 데이터: {len(R_gripper2base_list)}개\n")

    cv2.namedWindow("Manual Hand-Eye Calibration", cv2.WINDOW_NORMAL)

    try:
        while True:
            frame = camera.get_frame()
            display = frame.copy()

            corners = detect_checkerboard(frame, CHECKERBOARD_SIZE)

            if corners is not None:
                cv2.drawChessboardCorners(display, CHECKERBOARD_SIZE,
                                         corners.reshape(-1, 1, 2), True)
                try:
                    t_board, R_board = estimate_board_pose(
                        corners, CHECKERBOARD_SIZE, SQUARE_SIZE,
                        camera.camera_matrix, camera.dist_coeffs
                    )
                    dist = np.linalg.norm(t_board)
                    cv2.putText(display, f"Distance: {dist:.3f}m",
                               (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(display, "Checkerboard OK - Press 's' to save",
                               (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                except:
                    cv2.putText(display, "Pose estimation failed",
                               (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
            else:
                cv2.putText(display, "Checkerboard NOT FOUND",
                           (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # 상태 표시
            cv2.putText(display, f"Saved: {len(R_gripper2base_list)}/15+ poses",
                       (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            # 회전 변화량 표시
            if len(R_gripper2base_list) >= 2:
                stats = compute_rotation_stats(R_gripper2base_list)
                color = (0, 255, 0) if stats['total_rotation'] >= 30 else (0, 165, 255)
                cv2.putText(display, f"Max rotation: {stats['total_rotation']:.1f} deg (need 30+)",
                           (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            cv2.imshow("Manual Hand-Eye Calibration", display)

            key = cv2.waitKey(30) & 0xFF

            if key == ord('s'):
                if corners is not None:
                    try:
                        t_board, R_board = estimate_board_pose(
                            corners, CHECKERBOARD_SIZE, SQUARE_SIZE,
                            camera.camera_matrix, camera.dist_coeffs
                        )
                        t_robot, R_robot = robot.get_tcp_pose()

                        R_gripper2base_list.append(R_robot)
                        t_gripper2base_list.append(t_robot)
                        R_target2cam_list.append(R_board)
                        t_target2cam_list.append(t_board)

                        # 이미지 저장
                        img_path = os.path.join(IMAGE_DIR, f"pose_{len(R_gripper2base_list):02d}.jpg")
                        cv2.imwrite(img_path, frame)

                        # 중간 저장
                        np.savez(DATA_FILE,
                                R_gripper2base=R_gripper2base_list,
                                t_gripper2base=t_gripper2base_list,
                                R_target2cam=R_target2cam_list,
                                t_target2cam=t_target2cam_list)

                        print(f"\n[저장됨] 총 {len(R_gripper2base_list)}개")
                        print(f"  Robot TCP: [{t_robot[0]*1000:.1f}, {t_robot[1]*1000:.1f}, {t_robot[2]*1000:.1f}] mm")

                        if len(R_gripper2base_list) >= 2:
                            stats = compute_rotation_stats(R_gripper2base_list)
                            print(f"  회전 변화량: {stats['total_rotation']:.1f}° (최소 30° 필요)")

                    except Exception as e:
                        print(f"\n[오류] 저장 실패: {e}")
                else:
                    print("\n[경고] 체커보드가 보이지 않습니다!")

            elif key == ord('d'):
                if len(R_gripper2base_list) > 0:
                    R_gripper2base_list.pop()
                    t_gripper2base_list.pop()
                    R_target2cam_list.pop()
                    t_target2cam_list.pop()
                    print(f"\n[삭제됨] 남은 데이터: {len(R_gripper2base_list)}개")

            elif key == ord('r'):
                R_gripper2base_list = []
                t_gripper2base_list = []
                R_target2cam_list = []
                t_target2cam_list = []
                if os.path.exists(DATA_FILE):
                    os.remove(DATA_FILE)
                for f in os.listdir(IMAGE_DIR):
                    os.remove(os.path.join(IMAGE_DIR, f))
                print("\n[초기화됨] 모든 데이터 삭제됨")

            elif key == ord('i'):
                if len(R_gripper2base_list) >= 2:
                    stats = compute_rotation_stats(R_gripper2base_list)
                    print(f"\n[회전 변화량 통계]")
                    print(f"  최대: {stats['total_rotation']:.1f}°")
                    print(f"  최소: {stats['min_single']:.1f}°")
                    print(f"  평균: {stats['avg']:.1f}°")
                    if stats['total_rotation'] >= 30:
                        print("  -> 충분한 회전 변화량!")
                    else:
                        print(f"  -> {30 - stats['total_rotation']:.1f}° 더 필요")
                else:
                    print("\n[정보] 최소 2개 이상의 데이터가 필요합니다")

            elif key == ord('c'):
                if len(R_gripper2base_list) < 3:
                    print("\n[오류] 최소 3개 이상의 데이터가 필요합니다")
                    continue

                stats = compute_rotation_stats(R_gripper2base_list)
                if stats['total_rotation'] < 30:
                    print(f"\n[경고] 회전 변화량이 부족합니다: {stats['total_rotation']:.1f}° (최소 30° 필요)")
                    response = input("그래도 진행할까요? (y/n): ")
                    if response.lower() != 'y':
                        continue

                print("\n" + "=" * 60)
                print("캘리브레이션 수행 중...")

                try:
                    R_cam2tcp, t_cam2tcp = calibrate_hand_eye(
                        R_gripper2base_list,
                        t_gripper2base_list,
                        R_target2cam_list,
                        t_target2cam_list
                    )

                    T_cam2tcp = np.eye(4)
                    T_cam2tcp[:3, :3] = R_cam2tcp
                    T_cam2tcp[:3, 3] = t_cam2tcp

                    print("\n캘리브레이션 결과 (Eye-in-Hand):")
                    print("-" * 40)
                    print("Camera -> TCP(그리퍼) 변환 행렬:")
                    print(T_cam2tcp)

                    print(f"\n카메라 위치 (TCP 기준):")
                    print(f"  x: {t_cam2tcp[0]*1000:.1f} mm")
                    print(f"  y: {t_cam2tcp[1]*1000:.1f} mm")
                    print(f"  z: {t_cam2tcp[2]*1000:.1f} mm")

                    # 결과 저장 (Eye-in-Hand용)
                    np.savez(OUTPUT_FILE,
                            T_cam2tcp=T_cam2tcp,
                            R_cam2tcp=R_cam2tcp,
                            t_cam2tcp=t_cam2tcp,
                            # 호환성을 위해 기존 키도 유지 (동일 값)
                            T_cam2base=T_cam2tcp,
                            R_cam2base=R_cam2tcp,
                            t_cam2base=t_cam2tcp,
                            calibration_type='eye_in_hand',
                            camera_matrix=camera.camera_matrix,
                            dist_coeffs=camera.dist_coeffs,
                            checkerboard_size=CHECKERBOARD_SIZE,
                            square_size=SQUARE_SIZE,
                            num_samples=len(R_gripper2base_list))

                    print(f"\n결과 저장됨: {OUTPUT_FILE}")
                    print("  캘리브레이션 타입: Eye-in-Hand")
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
        rclpy.shutdown()


if __name__ == "__main__":
    main()
