#!/usr/bin/env python3
"""
자동 Hand-Eye Calibration for Eye-to-Hand Setup

로봇이 자동으로 여러 포즈로 이동하면서 체커보드 이미지와 TCP 포즈를 수집합니다.

사용법:
    1. 로봇 bringup 실행 (다른 터미널)
       ros2 launch e0509_gripper_description bringup.launch.py mode:=real host:=<IP>

    2. 그리퍼로 체커보드 잡기

    3. 이 스크립트 실행
       python auto_hand_eye_calibration.py

    4. 's' 키로 현재 위치를 시작 포즈로 저장
    5. Enter 키로 자동 캘리브레이션 시작
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
    from dsr_msgs2.srv import GetCurrentPosx, MoveJoint
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

# 로봇 이동 설정
MOVE_VEL = 10.0  # 이동 속도 (deg/s) - 천천히
MOVE_ACC = 10.0  # 가속도 (deg/s^2) - 부드럽게
SETTLE_TIME = 5.0  # 이동 후 안정화 대기 시간 (초)


class RealSenseCamera:
    """RealSense D455F 카메라 인터페이스"""

    def __init__(self):
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
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


class RobotController:
    """Doosan 로봇 제어 인터페이스 (ROS2 서비스)"""

    def __init__(self, namespace='dsr01'):
        self.namespace = namespace
        self.node = rclpy.create_node('auto_hand_eye_calibration')

        # 서비스 클라이언트 생성
        self.cli_get_posx = self.node.create_client(
            GetCurrentPosx, f'/{namespace}/aux_control/get_current_posx')
        self.cli_move_joint = self.node.create_client(
            MoveJoint, f'/{namespace}/motion/move_joint')

        print("서비스 연결 대기 중...")
        if not self.cli_get_posx.wait_for_service(timeout_sec=10.0):
            raise RuntimeError("get_current_posx 서비스 연결 실패")
        if not self.cli_move_joint.wait_for_service(timeout_sec=10.0):
            raise RuntimeError("move_joint 서비스 연결 실패")
        print("서비스 연결 완료")

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
        rclpy.spin_until_future_complete(self.node, future, timeout_sec=5.0)
        result = future.result()

        if result and result.success and len(result.task_pos_info) > 0:
            pos_data = result.task_pos_info[0].data
            # [x, y, z, rx, ry, rz] - mm, degree
            x, y, z = pos_data[0] / 1000.0, pos_data[1] / 1000.0, pos_data[2] / 1000.0
            rx, ry, rz = np.deg2rad(pos_data[3]), np.deg2rad(pos_data[4]), np.deg2rad(pos_data[5])

            position = np.array([x, y, z])
            rotation = R.from_euler('ZYX', [rz, ry, rx])
            rotation_matrix = rotation.as_matrix()

            return position, rotation_matrix

        raise RuntimeError("TCP pose 획득 실패")

    def get_joint_positions(self) -> List[float]:
        """현재 조인트 각도 획득 (ROS2 topic에서)"""
        from sensor_msgs.msg import JointState

        joint_pos = None
        def callback(msg):
            nonlocal joint_pos
            # joint_1 ~ joint_6 추출
            joint_names = ['joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'joint_6']
            positions = []
            for name in joint_names:
                if name in msg.name:
                    idx = msg.name.index(name)
                    positions.append(np.rad2deg(msg.position[idx]))
            if len(positions) == 6:
                joint_pos = positions

        sub = self.node.create_subscription(JointState, f'/{self.namespace}/joint_states', callback, 10)

        # 메시지 대기
        timeout = time.time() + 2.0
        while joint_pos is None and time.time() < timeout:
            rclpy.spin_once(self.node, timeout_sec=0.1)

        self.node.destroy_subscription(sub)

        if joint_pos is None:
            raise RuntimeError("조인트 위치 획득 실패")

        return joint_pos

    def move_joint(self, joint_positions: List[float], vel: float = MOVE_VEL, acc: float = MOVE_ACC) -> bool:
        """
        조인트 이동
        Args:
            joint_positions: [j1, j2, j3, j4, j5, j6] in degrees
        """
        req = MoveJoint.Request()
        req.pos = [float(p) for p in joint_positions]
        req.vel = vel
        req.acc = acc

        future = self.cli_move_joint.call_async(req)
        rclpy.spin_until_future_complete(self.node, future, timeout_sec=30.0)
        result = future.result()

        return result is not None and result.success

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
    """Eye-to-Hand 캘리브레이션"""
    R_cam2base, t_cam2base = cv2.calibrateHandEye(
        R_gripper2base_list,
        t_gripper2base_list,
        R_target2cam_list,
        t_target2cam_list,
        method=cv2.CALIB_HAND_EYE_TSAI
    )
    return R_cam2base, t_cam2base.flatten()


def generate_calibration_poses(base_joints: List[float], num_poses: int = 20) -> List[List[float]]:
    """
    캘리브레이션용 포즈 생성

    기준 포즈에서 각 조인트를 조금씩 변경하여 다양한 포즈 생성
    충분한 회전 변화 (±15° 이상) 필요
    """
    poses = []

    # 기준 포즈 추가
    poses.append(base_joints.copy())

    # Joint 1 (베이스 회전) 변화 - 화면 안쪽 방향으로만
    for j1_offset in [5, 10, 15, 20]:
        pose = base_joints.copy()
        pose[0] += j1_offset
        poses.append(pose)

    # Joint 5 (손목 기울기) 변화 - 체커보드 기울임
    for j5_offset in [-15, -8, 8, 15]:
        pose = base_joints.copy()
        pose[4] += j5_offset
        poses.append(pose)

    # Joint 6 (손목 회전) 변화 - 체커보드 회전
    for j6_offset in [-20, -10, 10, 20]:
        pose = base_joints.copy()
        pose[5] += j6_offset
        poses.append(pose)

    # 복합 변화 (위치 + 기울기)
    combo_offsets = [
        [-10, 0, 0, 0, -10, 0],
        [10, 0, 0, 0, -10, 0],
        [-10, 0, 0, 0, 10, 0],
        [10, 0, 0, 0, 10, 0],
        [0, 0, 0, 0, -10, -15],
        [0, 0, 0, 0, -10, 15],
        [0, 0, 0, 0, 10, -15],
        [0, 0, 0, 0, 10, 15],
    ]

    for offset in combo_offsets:
        pose = [base_joints[i] + offset[i] for i in range(6)]
        poses.append(pose)

    return poses[:num_poses]


def main():
    print("=" * 60)
    print("  자동 Hand-Eye Calibration (Eye-to-Hand)")
    print("=" * 60)

    # 이미지 저장 디렉토리 생성
    os.makedirs(IMAGE_DIR, exist_ok=True)

    # 초기화
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
    robot = RobotController()

    # 데이터 저장용 리스트
    R_gripper2base_list = []
    t_gripper2base_list = []
    R_target2cam_list = []
    t_target2cam_list = []

    # 기존 데이터 확인
    if os.path.exists(DATA_FILE):
        print(f"\n기존 데이터 발견: {DATA_FILE}")
        response = input("기존 데이터를 삭제하고 새로 시작할까요? (y/n): ")
        if response.lower() == 'y':
            os.remove(DATA_FILE)
            # 이미지 폴더 비우기
            for f in os.listdir(IMAGE_DIR):
                os.remove(os.path.join(IMAGE_DIR, f))
            print("기존 데이터 삭제됨")

    print("\n" + "=" * 60)
    print("조작 방법:")
    print("  1. 그리퍼로 체커보드를 잡으세요")
    print("  2. 체커보드가 카메라에 잘 보이는 위치로 로봇을 이동하세요")
    print("  3. 's' 키를 눌러 현재 위치를 시작 포즈로 저장")
    print("  4. Enter 키를 눌러 자동 캘리브레이션 시작")
    print("  5. 'q' 키로 종료")
    print("=" * 60)

    cv2.namedWindow("Auto Hand-Eye Calibration", cv2.WINDOW_NORMAL)

    base_joints = None
    calibration_poses = None
    auto_mode = False
    current_pose_idx = 0

    try:
        while True:
            # 프레임 획득
            frame = camera.get_frame()
            display = frame.copy()

            # 체커보드 검출
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
                except:
                    pass
                status_color = (0, 255, 0)
                status_text = "Checkerboard OK"
            else:
                status_color = (0, 0, 255)
                status_text = "Checkerboard NOT FOUND"

            cv2.putText(display, status_text, (20, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

            # 상태 표시
            if base_joints is not None:
                cv2.putText(display, "Base pose SET", (20, 90),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                cv2.putText(display, "Press 's' to set base pose", (20, 90),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            cv2.putText(display, f"Saved: {len(R_gripper2base_list)} poses", (20, 120),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            if auto_mode and calibration_poses:
                cv2.putText(display, f"AUTO MODE: {current_pose_idx}/{len(calibration_poses)}",
                           (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            cv2.imshow("Auto Hand-Eye Calibration", display)

            # 자동 모드 처리
            if auto_mode and calibration_poses and current_pose_idx < len(calibration_poses):
                target_pose = calibration_poses[current_pose_idx]
                print(f"\n[{current_pose_idx + 1}/{len(calibration_poses)}] 포즈로 이동 중...")
                print(f"  Target: {[f'{p:.1f}' for p in target_pose]}")

                # 로봇 이동
                success = robot.move_joint(target_pose)
                if not success:
                    print("  [경고] 이동 실패, 다음 포즈로 건너뜀")
                    current_pose_idx += 1
                    continue

                # 안정화 대기
                print(f"  안정화 대기 ({SETTLE_TIME}초)...")
                time.sleep(SETTLE_TIME)

                # 프레임 다시 획득
                frame = camera.get_frame()
                corners = detect_checkerboard(frame, CHECKERBOARD_SIZE)

                if corners is not None:
                    try:
                        # 체커보드 pose
                        t_board, R_board = estimate_board_pose(
                            corners, CHECKERBOARD_SIZE, SQUARE_SIZE,
                            camera.camera_matrix, camera.dist_coeffs
                        )

                        # 로봇 TCP pose
                        t_robot, R_robot = robot.get_tcp_pose()

                        # 저장
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

                        print(f"  [저장됨] 총 {len(R_gripper2base_list)}개")

                    except Exception as e:
                        print(f"  [오류] 데이터 저장 실패: {e}")
                else:
                    print("  [경고] 체커보드 검출 실패, 다음 포즈로")

                current_pose_idx += 1

                # 모든 포즈 완료
                if current_pose_idx >= len(calibration_poses):
                    auto_mode = False
                    print("\n" + "=" * 60)
                    print("자동 데이터 수집 완료!")
                    print(f"총 {len(R_gripper2base_list)}개의 데이터 수집됨")
                    print("'c' 키를 눌러 캘리브레이션을 수행하세요")
                    print("=" * 60)

                continue

            key = cv2.waitKey(100) & 0xFF

            if key == ord('s'):
                # 현재 위치를 기준 포즈로 저장
                try:
                    base_joints = robot.get_joint_positions()
                    print(f"\n[기준 포즈 저장됨]")
                    print(f"  Joints: {[f'{p:.1f}' for p in base_joints]}")

                    # 캘리브레이션 포즈 생성
                    calibration_poses = generate_calibration_poses(base_joints, num_poses=20)
                    print(f"  {len(calibration_poses)}개의 캘리브레이션 포즈 생성됨")
                    print("\nEnter 키를 눌러 자동 캘리브레이션을 시작하세요")
                except Exception as e:
                    print(f"\n[오류] 기준 포즈 저장 실패: {e}")

            elif key == 13:  # Enter
                if base_joints is not None and calibration_poses is not None:
                    print("\n자동 캘리브레이션 시작!")
                    auto_mode = True
                    current_pose_idx = 0
                else:
                    print("\n먼저 's' 키로 기준 포즈를 저장하세요")

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

                    T_cam2base = np.eye(4)
                    T_cam2base[:3, :3] = R_cam2base
                    T_cam2base[:3, 3] = t_cam2base

                    print("\n캘리브레이션 결과:")
                    print("-" * 40)
                    print("Camera -> Robot Base 변환 행렬:")
                    print(T_cam2base)

                    print(f"\n카메라 위치 (로봇 베이스 기준):")
                    print(f"  x: {t_cam2base[0]*1000:.1f} mm")
                    print(f"  y: {t_cam2base[1]*1000:.1f} mm")
                    print(f"  z: {t_cam2base[2]*1000:.1f} mm")

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

            elif key == ord('h'):
                # 홈 위치로 이동
                if base_joints is not None:
                    print("\n기준 포즈로 복귀 중...")
                    robot.move_joint(base_joints)
                    time.sleep(SETTLE_TIME)

            elif key == ord('q'):
                break

    finally:
        cv2.destroyAllWindows()
        camera.stop()
        robot.shutdown()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
