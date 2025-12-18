#!/usr/bin/env python3
"""
Hand-Eye Calibration 검증 스크립트 (Eye-in-Hand)

Eye-in-Hand 캘리브레이션 결과를 검증합니다:
1. 여러 로봇 자세에서 고정된 체커보드 촬영
2. 각 자세에서 카메라 좌표 → 로봇 좌표 변환
3. 변환 결과의 일관성 확인 (고정 체커보드 = 동일 로봇 좌표)

검증 원리:
    - 체커보드가 고정되어 있으므로 로봇 좌표계에서 항상 같은 위치
    - 여러 자세에서 변환한 결과가 일치해야 캘리브레이션 정확함

사용법:
    1. 로봇 bringup 실행
    2. 체커보드를 고정 위치에 배치
    3. python verify_calibration.py
    4. 여러 자세에서 's' 키로 검증
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
CHECKERBOARD_SIZE = (6, 9)
SQUARE_SIZE = 0.025  # 25mm

CALIBRATION_FILE = "/home/fhekwn549/doosan_ws/src/e0509_gripper_description/scripts/sim2real/calibration_result.npz"


class RealSenseCamera:
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

    def get_frame(self) -> np.ndarray:
        frames = self.pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        return np.asanyarray(color_frame.get_data())

    def stop(self):
        self.pipeline.stop()


class RobotInterface:
    def __init__(self, namespace='dsr01'):
        self.namespace = namespace
        self.node = rclpy.create_node('verify_calibration')

        self.cli_get_posx = self.node.create_client(
            GetCurrentPosx, f'/{namespace}/aux_control/get_current_posx')

        print("서비스 연결 대기 중...")
        if not self.cli_get_posx.wait_for_service(timeout_sec=10.0):
            raise RuntimeError("get_current_posx 서비스 연결 실패")
        print("서비스 연결 완료")

    def get_tcp_pose(self) -> Tuple[np.ndarray, np.ndarray]:
        req = GetCurrentPosx.Request()
        req.ref = 0

        future = self.cli_get_posx.call_async(req)

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


def main():
    print("=" * 60)
    print("  Hand-Eye Calibration 검증 (Eye-in-Hand)")
    print("  체커보드 고정, 카메라가 그리퍼에 부착된 구조")
    print("=" * 60)

    # 캘리브레이션 결과 로드
    if not os.path.exists(CALIBRATION_FILE):
        print(f"Error: 캘리브레이션 파일이 없습니다: {CALIBRATION_FILE}")
        return

    calib = np.load(CALIBRATION_FILE)

    # Eye-in-Hand: T_cam2tcp 로드
    if 'T_cam2tcp' in calib:
        T_cam2tcp = calib['T_cam2tcp']
        calib_type = calib.get('calibration_type', 'eye_in_hand')
    else:
        T_cam2tcp = calib['T_cam2base']
        calib_type = calib.get('calibration_type', 'eye_to_hand')
        print("[주의] Eye-to-Hand 캘리브레이션 파일 감지됨")

    print(f"\n캘리브레이션 결과 로드됨:")
    print(f"  타입: {calib_type}")
    print(f"  카메라 위치 (TCP 기준): x={T_cam2tcp[0,3]*1000:.1f}, y={T_cam2tcp[1,3]*1000:.1f}, z={T_cam2tcp[2,3]*1000:.1f} mm")

    if not HAS_REALSENSE or not HAS_ROS2:
        print("Error: RealSense와 ROS2가 필요합니다")
        return

    print("\nRealSense 카메라 초기화...")
    camera = RealSenseCamera()

    rclpy.init()
    print("로봇 연결 중...")
    robot = RobotInterface()

    # 검증 결과 저장
    board_positions_in_robot: List[np.ndarray] = []  # 로봇 좌표계에서 체커보드 위치들

    print("\n" + "=" * 60)
    print("조작 방법:")
    print("  's' - 현재 자세에서 검증 (체커보드 위치 기록)")
    print("  'r' - 검증 결과 초기화")
    print("  'q' - 종료")
    print("=" * 60)
    print("\n[Eye-in-Hand 검증 원리]")
    print("  체커보드가 고정되어 있으므로, 여러 자세에서 변환한")
    print("  로봇 좌표가 일치해야 캘리브레이션이 정확합니다.\n")

    cv2.namedWindow("Calibration Verification (Eye-in-Hand)", cv2.WINDOW_NORMAL)

    try:
        while True:
            frame = camera.get_frame()
            display = frame.copy()

            corners = detect_checkerboard(frame, CHECKERBOARD_SIZE)

            if corners is not None:
                cv2.drawChessboardCorners(display, CHECKERBOARD_SIZE,
                                         corners.reshape(-1, 1, 2), True)

                try:
                    # 카메라 좌표계에서 체커보드 위치
                    t_board_cam, R_board_cam = estimate_board_pose(
                        corners, CHECKERBOARD_SIZE, SQUARE_SIZE,
                        camera.camera_matrix, camera.dist_coeffs
                    )

                    # 현재 TCP pose 획득
                    t_tcp, R_tcp = robot.get_tcp_pose()

                    # Eye-in-Hand 변환: 카메라 → TCP → 로봇 베이스
                    # Step 1: 카메라 → TCP
                    t_board_cam_h = np.append(t_board_cam, 1)
                    t_board_tcp_h = T_cam2tcp @ t_board_cam_h
                    t_board_tcp = t_board_tcp_h[:3]

                    # Step 2: TCP → 로봇 베이스
                    T_tcp2base = np.eye(4)
                    T_tcp2base[:3, :3] = R_tcp
                    T_tcp2base[:3, 3] = t_tcp

                    t_board_tcp_h2 = np.append(t_board_tcp, 1)
                    t_board_base_h = T_tcp2base @ t_board_tcp_h2
                    t_board_base = t_board_base_h[:3]

                    cv2.putText(display, f"Board in cam: [{t_board_cam[0]*1000:.0f}, {t_board_cam[1]*1000:.0f}, {t_board_cam[2]*1000:.0f}] mm",
                               (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    cv2.putText(display, f"Board in TCP: [{t_board_tcp[0]*1000:.0f}, {t_board_tcp[1]*1000:.0f}, {t_board_tcp[2]*1000:.0f}] mm",
                               (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    cv2.putText(display, f"Board in base: [{t_board_base[0]*1000:.0f}, {t_board_base[1]*1000:.0f}, {t_board_base[2]*1000:.0f}] mm",
                               (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    cv2.putText(display, "Press 's' to record",
                               (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                except Exception as e:
                    cv2.putText(display, f"Error: {e}",
                               (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            else:
                cv2.putText(display, "Checkerboard NOT FOUND",
                           (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # 일관성 통계 표시
            if len(board_positions_in_robot) >= 2:
                positions = np.array(board_positions_in_robot)
                mean_pos = np.mean(positions, axis=0)
                std_pos = np.std(positions, axis=0)
                max_deviation = np.max(np.linalg.norm(positions - mean_pos, axis=1)) * 1000

                cv2.putText(display, f"Samples: {len(board_positions_in_robot)}, Max deviation: {max_deviation:.1f} mm",
                           (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                cv2.putText(display, f"Mean: [{mean_pos[0]*1000:.0f}, {mean_pos[1]*1000:.0f}, {mean_pos[2]*1000:.0f}] mm",
                           (20, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            elif len(board_positions_in_robot) == 1:
                cv2.putText(display, f"Samples: 1 (need 2+ for consistency check)",
                           (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

            cv2.imshow("Calibration Verification (Eye-in-Hand)", display)

            key = cv2.waitKey(30) & 0xFF

            if key == ord('s'):
                if corners is not None:
                    try:
                        # 카메라 좌표계에서 체커보드 위치
                        t_board_cam, R_board_cam = estimate_board_pose(
                            corners, CHECKERBOARD_SIZE, SQUARE_SIZE,
                            camera.camera_matrix, camera.dist_coeffs
                        )

                        # 현재 TCP pose 획득
                        t_tcp, R_tcp = robot.get_tcp_pose()

                        # Eye-in-Hand 변환
                        t_board_cam_h = np.append(t_board_cam, 1)
                        t_board_tcp_h = T_cam2tcp @ t_board_cam_h
                        t_board_tcp = t_board_tcp_h[:3]

                        T_tcp2base = np.eye(4)
                        T_tcp2base[:3, :3] = R_tcp
                        T_tcp2base[:3, 3] = t_tcp

                        t_board_tcp_h2 = np.append(t_board_tcp, 1)
                        t_board_base_h = T_tcp2base @ t_board_tcp_h2
                        t_board_base = t_board_base_h[:3]

                        board_positions_in_robot.append(t_board_base.copy())

                        print(f"\n[기록 #{len(board_positions_in_robot)}]")
                        print(f"  TCP 위치: [{t_tcp[0]*1000:.1f}, {t_tcp[1]*1000:.1f}, {t_tcp[2]*1000:.1f}] mm")
                        print(f"  체커보드 (카메라 좌표): [{t_board_cam[0]*1000:.1f}, {t_board_cam[1]*1000:.1f}, {t_board_cam[2]*1000:.1f}] mm")
                        print(f"  체커보드 (로봇 좌표):   [{t_board_base[0]*1000:.1f}, {t_board_base[1]*1000:.1f}, {t_board_base[2]*1000:.1f}] mm")

                        if len(board_positions_in_robot) >= 2:
                            positions = np.array(board_positions_in_robot)
                            mean_pos = np.mean(positions, axis=0)
                            std_pos = np.std(positions, axis=0)
                            deviations = np.linalg.norm(positions - mean_pos, axis=1) * 1000

                            print(f"\n  [일관성 검사]")
                            print(f"    평균 위치: [{mean_pos[0]*1000:.1f}, {mean_pos[1]*1000:.1f}, {mean_pos[2]*1000:.1f}] mm")
                            print(f"    표준편차:  [{std_pos[0]*1000:.1f}, {std_pos[1]*1000:.1f}, {std_pos[2]*1000:.1f}] mm")
                            print(f"    최대 편차: {np.max(deviations):.1f} mm")
                            print(f"    평균 편차: {np.mean(deviations):.1f} mm")

                            if np.max(deviations) < 10:
                                print(f"    -> 우수! 캘리브레이션 정확도가 높습니다.")
                            elif np.max(deviations) < 30:
                                print(f"    -> 양호. 캘리브레이션이 적당합니다.")
                            else:
                                print(f"    -> 경고! 편차가 큽니다. 재캘리브레이션 권장.")

                    except Exception as e:
                        print(f"\n[오류] 기록 실패: {e}")
                else:
                    print("\n[경고] 체커보드가 보이지 않습니다!")

            elif key == ord('r'):
                board_positions_in_robot = []
                print("\n[초기화됨] 검증 결과 초기화")

            elif key == ord('q'):
                break

    finally:
        if len(board_positions_in_robot) >= 2:
            positions = np.array(board_positions_in_robot)
            mean_pos = np.mean(positions, axis=0)
            std_pos = np.std(positions, axis=0)
            deviations = np.linalg.norm(positions - mean_pos, axis=1) * 1000

            print("\n" + "=" * 60)
            print("최종 검증 결과 (Eye-in-Hand):")
            print("-" * 40)
            print(f"  테스트 횟수: {len(board_positions_in_robot)}")
            print(f"  평균 체커보드 위치 (로봇 좌표):")
            print(f"    x: {mean_pos[0]*1000:.1f} mm")
            print(f"    y: {mean_pos[1]*1000:.1f} mm")
            print(f"    z: {mean_pos[2]*1000:.1f} mm")
            print(f"\n  일관성 (편차):")
            print(f"    표준편차: x={std_pos[0]*1000:.1f}, y={std_pos[1]*1000:.1f}, z={std_pos[2]*1000:.1f} mm")
            print(f"    최대 편차: {np.max(deviations):.1f} mm")
            print(f"    평균 편차: {np.mean(deviations):.1f} mm")

            if np.max(deviations) < 10:
                print(f"\n  결론: 우수! 캘리브레이션 정확도가 높습니다.")
            elif np.max(deviations) < 30:
                print(f"\n  결론: 양호. 일반적인 작업에 사용 가능합니다.")
            else:
                print(f"\n  결론: 재캘리브레이션을 권장합니다.")
            print("=" * 60)

        cv2.destroyAllWindows()
        camera.stop()
        robot.shutdown()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
