#!/usr/bin/env python3
"""
자동 Eye-to-Hand 캘리브레이션

다양한 자세로 자동 이동하면서 캘리브레이션 데이터를 수집합니다.
"""

import numpy as np
import cv2
import time
import os
import sys
import itertools

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ROS2
import rclpy
from rclpy.node import Node
from dsr_msgs2.srv import MoveJoint, GetCurrentPosx, GetCurrentPosj

# RealSense & ArUco
import pyrealsense2 as rs
from cv2 import aruco

class AutoCalibrator:
    def __init__(self, namespace='dsr01'):
        rclpy.init()
        self.node = rclpy.create_node('auto_calibrator')

        # ROS2 서비스 클라이언트
        self.cli_movej = self.node.create_client(MoveJoint, f'/{namespace}/motion/move_joint')
        self.cli_get_posx = self.node.create_client(GetCurrentPosx, f'/{namespace}/aux_control/get_current_posx')
        self.cli_get_posj = self.node.create_client(GetCurrentPosj, f'/{namespace}/aux_control/get_current_posj')

        print("ROS2 서비스 연결 대기...")
        self.cli_movej.wait_for_service(timeout_sec=5.0)
        self.cli_get_posx.wait_for_service(timeout_sec=5.0)
        print("ROS2 연결 완료!")

        # 카메라
        self.pipeline = None
        self.camera_matrix = None
        self.dist_coeffs = None

        # 마커 설정
        self.marker_size = 0.04  # 4cm
        self.marker_id = 0

        # 수집된 데이터
        self.robot_poses = []
        self.marker_poses = []

        # 조인트 범위 (사용자 입력 기반 + 확장)
        self.joint_ranges = {
            'j1': [-40, 30],      # 기존 범위
            'j2': [-20, 50],      # 확장 (높이 변화)
            'j3': [50, 120],      # 확장 (높이 변화)
            'j4': [-45, 90],      # 확장 (기존 90 고정 → 다양화)
            'j5': [60, 120],      # 확장 (기존 90 고정 → 다양화)
            'j6': [-70, 70],      # 기존 범위
        }

    def start_camera(self):
        """카메라 시작"""
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

        profile = self.pipeline.start(config)

        # intrinsics
        color_stream = profile.get_stream(rs.stream.color)
        intr = color_stream.as_video_stream_profile().get_intrinsics()

        self.camera_matrix = np.array([
            [intr.fx, 0, intr.ppx],
            [0, intr.fy, intr.ppy],
            [0, 0, 1]
        ])
        self.dist_coeffs = np.array(intr.coeffs)

        print(f"카메라 시작 - fx={intr.fx:.1f}, fy={intr.fy:.1f}")
        time.sleep(0.5)

    def stop_camera(self):
        if self.pipeline:
            self.pipeline.stop()

    def get_frame(self):
        """카메라 프레임"""
        frames = self.pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if color_frame:
            return np.asanyarray(color_frame.get_data())
        return None

    def detect_marker(self, image):
        """ArUco 마커 감지"""
        # 여러 딕셔너리 시도
        dicts = [
            aruco.DICT_4X4_50,
            aruco.DICT_5X5_50,
            aruco.DICT_6X6_250,
        ]

        for dict_type in dicts:
            aruco_dict = aruco.getPredefinedDictionary(dict_type)
            params = aruco.DetectorParameters()
            detector = aruco.ArucoDetector(aruco_dict, params)
            corners, ids, _ = detector.detectMarkers(image)

            if ids is not None and self.marker_id in ids.flatten():
                idx = np.where(ids.flatten() == self.marker_id)[0][0]

                # Pose 추정
                half = self.marker_size / 2
                obj_pts = np.array([
                    [-half, half, 0], [half, half, 0],
                    [half, -half, 0], [-half, -half, 0]
                ], dtype=np.float32)

                marker_corners = corners[idx].reshape(4, 2)
                success, rvec, tvec = cv2.solvePnP(
                    obj_pts, marker_corners,
                    self.camera_matrix, self.dist_coeffs
                )

                if success:
                    return rvec, tvec

        return None, None

    def get_tcp_pose(self):
        """현재 TCP pose"""
        req = GetCurrentPosx.Request()
        req.ref = 0
        future = self.cli_get_posx.call_async(req)
        rclpy.spin_until_future_complete(self.node, future, timeout_sec=2.0)

        if future.result() and future.result().success:
            posx = list(future.result().task_pos_info[0].data)[:6]
            # mm -> m, deg -> rad
            position = np.array(posx[:3]) / 1000.0
            rx, ry, rz = np.radians(posx[3:6])

            # Euler to rotation matrix
            R = self._euler_to_rotation_matrix(rx, ry, rz)
            return R, position
        return None, None

    def _euler_to_rotation_matrix(self, a, b, c):
        """ZYZ Euler (Doosan 표준) -> Rotation matrix

        Doosan posx: [x, y, z, a, b, c] 형식
        ZYZ intrinsic: R = Rz(a) * Ry(b) * Rz(c)
        """
        ca, sa = np.cos(a), np.sin(a)
        cb, sb = np.cos(b), np.sin(b)
        cc, sc = np.cos(c), np.sin(c)

        Rz1 = np.array([[ca, -sa, 0], [sa, ca, 0], [0, 0, 1]])
        Ry = np.array([[cb, 0, sb], [0, 1, 0], [-sb, 0, cb]])
        Rz2 = np.array([[cc, -sc, 0], [sc, cc, 0], [0, 0, 1]])

        return Rz1 @ Ry @ Rz2

    def move_joint(self, joints_deg, vel=30, acc=30):
        """관절 이동"""
        req = MoveJoint.Request()
        req.pos = [float(j) for j in joints_deg]
        req.vel = float(vel)
        req.acc = float(acc)
        req.time = 0.0
        req.radius = 0.0
        req.mode = 0
        req.blend_type = 0
        req.sync_type = 0

        future = self.cli_movej.call_async(req)
        rclpy.spin_until_future_complete(self.node, future, timeout_sec=30.0)

        if future.result():
            return future.result().success
        return False

    def generate_poses(self, num_poses=25):
        """다양한 자세 생성"""
        poses = []

        # 각 조인트의 값들
        j1_vals = np.linspace(self.joint_ranges['j1'][0], self.joint_ranges['j1'][1], 3)
        j2_vals = np.linspace(self.joint_ranges['j2'][0], self.joint_ranges['j2'][1], 3)
        j3_vals = np.linspace(self.joint_ranges['j3'][0], self.joint_ranges['j3'][1], 3)
        j4_vals = np.linspace(self.joint_ranges['j4'][0], self.joint_ranges['j4'][1], 3)
        j5_vals = np.linspace(self.joint_ranges['j5'][0], self.joint_ranges['j5'][1], 3)
        j6_vals = np.linspace(self.joint_ranges['j6'][0], self.joint_ranges['j6'][1], 3)

        # 조합 생성 (전체 조합은 너무 많으므로 샘플링)
        all_combinations = list(itertools.product(j1_vals, j2_vals, j3_vals, j4_vals, j5_vals, j6_vals))

        # 랜덤 샘플링
        if len(all_combinations) > num_poses:
            indices = np.random.choice(len(all_combinations), num_poses, replace=False)
            poses = [all_combinations[i] for i in indices]
        else:
            poses = all_combinations

        return poses

    def capture_sample(self):
        """현재 자세에서 샘플 캡처"""
        # 안정화 대기
        time.sleep(0.5)

        # 여러 프레임 시도
        for _ in range(10):
            image = self.get_frame()
            if image is None:
                continue

            rvec, tvec = self.detect_marker(image)
            if rvec is not None:
                # 마커 pose
                R_marker, _ = cv2.Rodrigues(rvec)
                t_marker = tvec.flatten()

                # TCP pose
                R_tcp, t_tcp = self.get_tcp_pose()
                if R_tcp is None:
                    continue

                self.marker_poses.append((R_marker, t_marker))
                self.robot_poses.append((R_tcp, t_tcp))

                return True

            time.sleep(0.1)

        return False

    def calibrate(self):
        """캘리브레이션 수행"""
        if len(self.robot_poses) < 10:
            print(f"샘플 부족: {len(self.robot_poses)} < 10")
            return None

        print(f"\n캘리브레이션 수행 중 ({len(self.robot_poses)} 샘플)...")

        # 데이터 준비 (Eye-to-Hand)
        R_base2gripper = []
        t_base2gripper = []
        R_target2cam = []
        t_target2cam = []

        for (R_tcp, t_tcp), (R_marker, t_marker) in zip(self.robot_poses, self.marker_poses):
            # gripper2base -> base2gripper
            R_g2b = R_tcp
            t_g2b = t_tcp.reshape(3, 1)

            R_b2g = R_g2b.T
            t_b2g = -R_g2b.T @ t_g2b

            R_base2gripper.append(R_b2g)
            t_base2gripper.append(t_b2g)
            R_target2cam.append(R_marker)
            t_target2cam.append(t_marker.reshape(3, 1))

        # 여러 방법 시도
        methods = [
            (cv2.CALIB_HAND_EYE_TSAI, "Tsai"),
            (cv2.CALIB_HAND_EYE_PARK, "Park"),
            (cv2.CALIB_HAND_EYE_HORAUD, "Horaud"),
            (cv2.CALIB_HAND_EYE_ANDREFF, "Andreff"),
            (cv2.CALIB_HAND_EYE_DANIILIDIS, "Daniilidis"),
        ]

        best_R, best_t = None, None
        best_error = float('inf')
        best_method = ""

        for method, name in methods:
            try:
                R_cam2base, t_cam2base = cv2.calibrateHandEye(
                    R_base2gripper, t_base2gripper,
                    R_target2cam, t_target2cam,
                    method=method
                )

                # 에러 계산
                error = self._compute_error(R_cam2base, t_cam2base)
                print(f"  {name}: error = {error:.4f}m ({error*1000:.1f}mm)")

                if error < best_error:
                    best_error = error
                    best_R = R_cam2base
                    best_t = t_cam2base
                    best_method = name

            except Exception as e:
                print(f"  {name}: 실패 - {e}")

        if best_R is None:
            return None

        print(f"\n최적 방법: {best_method} (error: {best_error*1000:.1f}mm)")

        # 변환 행렬 생성
        T_cam_to_base = np.eye(4)
        T_cam_to_base[:3, :3] = best_R
        T_cam_to_base[:3, 3] = best_t.flatten()

        return T_cam_to_base, best_error

    def _compute_error(self, R_cam2base, t_cam2base):
        """평균 에러 계산"""
        T = np.eye(4)
        T[:3, :3] = R_cam2base
        T[:3, 3] = t_cam2base.flatten()

        errors = []
        for (R_tcp, t_tcp), (R_marker, t_marker) in zip(self.robot_poses, self.marker_poses):
            marker_cam = t_marker.reshape(3)
            marker_cam_h = np.append(marker_cam, 1)
            marker_base = (T @ marker_cam_h)[:3]

            tcp_base = t_tcp.reshape(3)
            error = np.linalg.norm(marker_base - tcp_base)
            errors.append(error)

        return np.mean(errors)

    def save_calibration(self, T_cam_to_base, path="config/calibration_eye_to_hand.npz"):
        """결과 저장"""
        os.makedirs(os.path.dirname(path), exist_ok=True)

        np.savez(
            path,
            T_cam_to_base=T_cam_to_base,
            camera_matrix=self.camera_matrix,
            dist_coeffs=self.dist_coeffs,
            marker_size=self.marker_size,
            marker_id=self.marker_id,
            num_samples=len(self.robot_poses)
        )
        print(f"저장 완료: {path}")

    def run(self, num_poses=25):
        """자동 캘리브레이션 실행"""
        print("=" * 60)
        print("자동 Eye-to-Hand 캘리브레이션")
        print("=" * 60)
        print(f"목표 자세 수: {num_poses}")
        print("마커가 그리퍼에 잘 붙어있는지 확인하세요!")
        print("=" * 60)

        input("Enter 키를 누르면 시작합니다...")

        # 카메라 시작
        self.start_camera()

        # 자세 생성
        poses = self.generate_poses(num_poses)
        print(f"\n{len(poses)}개 자세 생성 완료")

        # 자세별로 이동 및 캡처
        success_count = 0
        for i, pose in enumerate(poses):
            print(f"\n[{i+1}/{len(poses)}] 이동 중: {[f'{p:.0f}' for p in pose]}")

            # 이동
            if not self.move_joint(pose, vel=30, acc=30):
                print("  이동 실패, 건너뜀")
                continue

            # 캡처
            if self.capture_sample():
                success_count += 1
                print(f"  캡처 성공! (총 {success_count}개)")
            else:
                print("  마커 감지 실패, 건너뜀")

        print(f"\n총 {success_count}개 샘플 수집 완료")

        # 캘리브레이션
        result = self.calibrate()

        if result:
            T_cam_to_base, error = result
            print(f"\n캘리브레이션 성공!")
            print(f"평균 오차: {error*1000:.1f}mm")
            print(f"\nT_cam_to_base:\n{T_cam_to_base}")

            # 저장
            save = input("\n저장하시겠습니까? (y/n): ")
            if save.lower() == 'y':
                self.save_calibration(T_cam_to_base)
        else:
            print("캘리브레이션 실패!")

        # 정리
        self.stop_camera()
        self.node.destroy_node()
        rclpy.shutdown()


def main():
    calibrator = AutoCalibrator()
    calibrator.run(num_poses=30)


if __name__ == "__main__":
    main()
