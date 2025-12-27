#!/usr/bin/env python3
"""
캘리브레이션된 펜 감지 테스트 (YOLO 세그멘테이션)

펜 감지 결과를 로봇 좌표계로 변환하여 표시합니다.
'g' 키를 누르면 로봇이 펜 위치 위로 이동합니다.

Usage:
    python test_pen_detection_calibrated.py
"""

import numpy as np
import cv2
import os
import sys
import time
import threading

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pen_detector_yolo import YOLOPenDetector, YOLODetectorConfig, DetectionState

# ROS2
try:
    import rclpy
    from dsr_msgs2.srv import MoveLine
    HAS_ROS2 = True
except ImportError:
    HAS_ROS2 = False
    print("[Warning] ROS2 not available, robot movement disabled")


class CalibratedPenDetector:
    """캘리브레이션된 펜 감지기 (YOLO 세그멘테이션)"""

    def __init__(self, model_path: str = None, calibration_path: str = None):
        script_dir = os.path.dirname(os.path.abspath(__file__))

        # YOLO 모델 경로
        if model_path is None:
            model_path = os.path.join(os.path.expanduser("~"), "runs/segment/train/weights/best.pt")

        # 펜 감지기 (YOLO 세그멘테이션)
        self.detector = YOLOPenDetector(model_path)

        # 캘리브레이션 로드
        self.R_axes = None
        self.t_offset = None
        self.T_cam_to_base = None

        if calibration_path is None:
            # scripts/config 폴더 (캘리브레이션 스크립트 실행 위치)
            calibration_path = os.path.join(script_dir, "config", "calibration_eye_to_hand.npz")
        self.load_calibration(calibration_path)

    def load_calibration(self, path: str) -> bool:
        """캘리브레이션 결과 로드 (Simple 방식)"""
        if not os.path.exists(path):
            print(f"[CalibratedPen] Calibration file not found: {path}")
            return False

        data = np.load(path)
        self.T_cam_to_base = data['T_cam_to_base']

        # Simple 방식 데이터
        if 'R_axes' in data and 't_offset' in data:
            self.R_axes = data['R_axes']
            self.t_offset = data['t_offset']
            print(f"[CalibratedPen] Loaded (Simple method)")
            print(f"  R_axes:\n{self.R_axes}")
            print(f"  t_offset: {self.t_offset}")
        else:
            # 기존 방식 호환
            self.R_axes = self.T_cam_to_base[:3, :3]
            self.t_offset = self.T_cam_to_base[:3, 3]

        print(f"[CalibratedPen] Loaded from {path}")
        return True

    def transform_to_robot(self, point_cam: np.ndarray) -> np.ndarray:
        """카메라 좌표 → 로봇 베이스 좌표 변환 (Simple 방식)"""
        if self.R_axes is None or self.t_offset is None:
            return point_cam

        return self.R_axes @ point_cam + self.t_offset

    def transform_direction_to_robot(self, direction_cam: np.ndarray) -> np.ndarray:
        """카메라 좌표계 방향 → 로봇 베이스 좌표계 방향 변환"""
        if self.R_axes is None:
            return direction_cam

        direction_base = self.R_axes @ direction_cam
        norm = np.linalg.norm(direction_base)
        if norm > 0:
            direction_base = direction_base / norm
        return direction_base

    def start(self) -> bool:
        return self.detector.start()

    def stop(self):
        self.detector.stop()

    def detect(self):
        """펜 감지 및 로봇 좌표계로 변환"""
        result = self.detector.detect()

        if result is None or result.state != DetectionState.DETECTED:
            return None

        # 카메라 좌표계 위치 (grasp = cap)
        grasp_cam = result.grasp_point
        if grasp_cam is None or np.linalg.norm(grasp_cam) < 0.01:
            return None  # 깊이 데이터 없음

        # 로봇 좌표계로 변환
        grasp_robot = self.transform_to_robot(grasp_cam)

        # 펜 방향 (cap → tip)
        direction_cam = None
        direction_robot = None
        if result.cap_3d is not None and result.tip_3d is not None:
            dir_cam = result.tip_3d - result.cap_3d
            if np.linalg.norm(dir_cam) > 0.01:
                direction_cam = dir_cam / np.linalg.norm(dir_cam)
                direction_robot = self.transform_direction_to_robot(direction_cam)

        return {
            'grasp_cam': grasp_cam,
            'grasp_robot': grasp_robot,
            'direction_cam': direction_cam,
            'direction_robot': direction_robot,
            'confidence': result.confidence,
            'bbox': result.bbox,
            'cap_pixel': result.cap_pixel,
            'tip_pixel': result.tip_pixel,
            'pitch_deg': result.pitch_deg,
            'yaw_deg': result.yaw_deg,
            'mask': result.mask
        }

    def get_frames(self):
        return self.detector.get_frames()

    def get_last_frames(self):
        """마지막 detect()에서 사용한 프레임 반환 (동기화용)"""
        return self.detector.get_last_frames()

    def visualize(self, image, result_dict):
        """결과 시각화"""
        if result_dict is None:
            return image

        # 기존 YOLO 시각화 사용하려면 detector의 _last_result 사용
        if self.detector._last_result is not None:
            return self.detector.visualize(image, self.detector._last_result)
        return image


class RobotMover:
    """로봇 이동 제어"""

    def __init__(self, namespace='dsr01'):
        self.namespace = namespace
        self.node = None
        self.cli_move_line = None
        self.cli_get_posx = None
        self.running = True
        self.spin_thread = None

        if not HAS_ROS2:
            return

        rclpy.init()
        self.node = rclpy.create_node('pen_test_mover')

        from dsr_msgs2.srv import GetCurrentPosx
        self.cli_move_line = self.node.create_client(
            MoveLine, f'/{namespace}/motion/move_line')
        self.cli_get_posx = self.node.create_client(
            GetCurrentPosx, f'/{namespace}/aux_control/get_current_posx')

        print("로봇 서비스 연결 대기...")
        if not self.cli_move_line.wait_for_service(timeout_sec=5.0):
            print("[Warning] move_line 서비스 연결 실패")
            self.cli_move_line = None
        if not self.cli_get_posx.wait_for_service(timeout_sec=5.0):
            print("[Warning] get_current_posx 서비스 연결 실패")
            self.cli_get_posx = None

        if self.cli_move_line and self.cli_get_posx:
            print("로봇 서비스 연결 완료")

        self.spin_thread = threading.Thread(target=self._spin, daemon=True)
        self.spin_thread.start()

    def _spin(self):
        while self.running and rclpy.ok():
            rclpy.spin_once(self.node, timeout_sec=0.1)

    def get_current_tcp(self):
        """현재 TCP 위치/자세 가져오기"""
        if self.cli_get_posx is None:
            return None

        from dsr_msgs2.srv import GetCurrentPosx
        req = GetCurrentPosx.Request()
        req.ref = 0  # DR_BASE

        future = self.cli_get_posx.call_async(req)

        timeout = time.time() + 2.0
        while not future.done() and time.time() < timeout:
            time.sleep(0.05)

        if future.done():
            result = future.result()
            if result and result.success and len(result.task_pos_info) > 0:
                return list(result.task_pos_info[0].data)[:6]
        return None

    def move_to(self, x_mm, y_mm, z_mm, offset_z=100):
        """펜 위치 위로 이동 (offset_z mm 위) - 현재 자세 유지"""
        if self.cli_move_line is None:
            print("로봇 서비스 없음")
            return False

        # 현재 TCP 자세 가져오기
        current_tcp = self.get_current_tcp()
        if current_tcp is None:
            print("현재 TCP 위치를 가져올 수 없음")
            return False

        # 현재 자세(RX, RY, RZ) 유지
        rx, ry, rz = current_tcp[3], current_tcp[4], current_tcp[5]

        # Z를 offset만큼 위로
        target_z = z_mm + offset_z

        print(f"\n로봇 이동: [{x_mm:.1f}, {y_mm:.1f}, {target_z:.1f}] mm")
        print(f"  현재 자세 유지: RX={rx:.1f}, RY={ry:.1f}, RZ={rz:.1f}")
        print(f"  (펜 위치에서 {offset_z}mm 위)")

        req = MoveLine.Request()
        req.pos = [float(x_mm), float(y_mm), float(target_z), rx, ry, rz]
        req.vel = [50.0, 20.0]  # 더 느린 속도
        req.acc = [50.0, 20.0]
        req.time = 0.0
        req.radius = 0.0
        req.ref = 0  # DR_BASE
        req.mode = 0  # DR_MV_MOD_ABS
        req.blend_type = 0
        req.sync_type = 0  # SYNC

        future = self.cli_move_line.call_async(req)

        # 완료 대기
        timeout = time.time() + 30.0
        while not future.done() and time.time() < timeout:
            time.sleep(0.1)

        if future.done() and future.result().success:
            print("이동 완료!")
            return True
        else:
            print("이동 실패 또는 타임아웃")
            return False

    def shutdown(self):
        self.running = False
        if self.node:
            self.node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def main():
    print("=" * 60)
    print("캘리브레이션된 펜 감지 테스트 (YOLO 세그멘테이션)")
    print("=" * 60)
    print("펜 위치가 카메라 좌표(Cam)와 로봇 좌표(Robot)로 표시됩니다.")
    print("")
    print("조작:")
    print("  g: 펜 위치로 로봇 이동 (100mm 위)")
    print("  r: Cap/Tip 트래킹 리셋")
    print("  q: 종료")
    print("=" * 60)

    detector = CalibratedPenDetector()
    robot = RobotMover() if HAS_ROS2 else None

    # 캘리브레이션 적용 여부 확인
    if detector.R_axes is not None:
        print("\n[OK] 캘리브레이션이 적용되었습니다.")
        calibrated = True
    else:
        print("\n[WARNING] 캘리브레이션이 적용되지 않음! 카메라 좌표 그대로 사용됩니다.")
        calibrated = False

    if not detector.start():
        print("카메라 시작 실패!")
        return

    last_robot_pos = None  # 마지막 감지된 로봇 좌표

    try:
        while True:
            # 먼저 감지 수행 (내부적으로 프레임 가져옴)
            result = detector.detect()

            # detect()에서 사용한 프레임 가져오기 (동기화)
            color_image, depth_image = detector.get_last_frames()
            if color_image is None:
                continue

            # YOLO 시각화 적용 (감지에 사용한 동일 프레임)
            display = detector.visualize(color_image, result)

            if result is not None:
                # 로봇 좌표 표시 (화면 오른쪽)
                robot_pos = result['grasp_robot']
                last_robot_pos = robot_pos.copy()

                # 로봇 좌표 (mm 단위) - 오른쪽에 표시
                robot_mm = robot_pos * 1000
                x_offset = display.shape[1] - 280

                cv2.putText(display, "=== ROBOT COORD ===",
                           (x_offset, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                cv2.putText(display, f"X: {robot_mm[0]:+7.1f} mm",
                           (x_offset, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                cv2.putText(display, f"Y: {robot_mm[1]:+7.1f} mm",
                           (x_offset, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                cv2.putText(display, f"Z: {robot_mm[2]:+7.1f} mm",
                           (x_offset, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

                # 방향 표시
                if result['direction_robot'] is not None:
                    dir_r = result['direction_robot']
                    cv2.putText(display, f"Dir: [{dir_r[0]:+.2f}, {dir_r[1]:+.2f}, {dir_r[2]:+.2f}]",
                               (x_offset, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)

                # 안내
                cv2.putText(display, "[G] Move robot",
                           (x_offset, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

            # 캘리브레이션 상태 표시
            status_text = "CALIBRATED" if calibrated else "NOT CALIBRATED!"
            status_color = (0, 255, 0) if calibrated else (0, 0, 255)
            cv2.putText(display, status_text, (display.shape[1] - 150, display.shape[0] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 2)

            # 화면 확대
            scale = 1.5
            new_w = int(display.shape[1] * scale)
            new_h = int(display.shape[0] * scale)
            display = cv2.resize(display, (new_w, new_h))

            cv2.imshow("Calibrated Pen Detection (YOLO)", display)

            key = cv2.waitKey(30) & 0xFF

            if key == ord('q'):
                break
            elif key == ord('r'):
                # Cap/Tip 트래킹 리셋
                detector.detector.reset_tracking()
                print("Cap/Tip 트래킹 리셋 - 펜을 cap이 위로 오게 놓으세요")
            elif key == ord('g'):
                # 펜 위치로 로봇 이동 - 현재 감지된 좌표 사용
                if result is not None and robot is not None:
                    robot_pos = result['grasp_robot']
                    robot_mm = robot_pos * 1000

                    # 안전 범위 확인
                    if robot_mm[2] < 50:
                        print(f"[안전] Z={robot_mm[2]:.1f}mm 너무 낮음! 이동 취소")
                    elif abs(robot_mm[0]) > 800 or abs(robot_mm[1]) > 800:
                        print(f"[안전] X/Y 범위 초과! 이동 취소")
                    else:
                        print(f"\n[이동] 현재 감지 좌표: X={robot_mm[0]:.1f}, Y={robot_mm[1]:.1f}, Z={robot_mm[2]:.1f}")
                        robot.move_to(robot_mm[0], robot_mm[1], robot_mm[2], offset_z=100)
                else:
                    print("펜이 현재 감지되지 않음! 펜을 카메라에 보이게 하고 다시 시도하세요.")

    except KeyboardInterrupt:
        pass

    detector.stop()
    if robot:
        robot.shutdown()
    cv2.destroyAllWindows()
    print("종료")


if __name__ == "__main__":
    main()
