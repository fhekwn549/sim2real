#!/usr/bin/env python3
"""
Z 오프셋 보정 스크립트

1. 펜을 감지해서 카메라 좌표 출력
2. 로봇을 수동으로 펜 위치에 가져간 후 실제 TCP 좌표 출력
3. Z 차이 계산해서 캘리브레이션 업데이트

Usage:
    python calibrate_z_offset.py
"""

import numpy as np
import cv2
import os
import sys
import time
import threading

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pen_detector_yolo import YOLOPenDetector, DetectionState

# ROS2
try:
    import rclpy
    from dsr_msgs2.srv import GetCurrentPosx
    HAS_ROS2 = True
except ImportError:
    HAS_ROS2 = False
    print("[Warning] ROS2 not available")


class ZOffsetCalibrator:
    def __init__(self):
        self.script_dir = os.path.dirname(os.path.abspath(__file__))

        # YOLO 펜 감지기
        model_path = "/home/fhekwn549/runs/segment/train/weights/best.pt"
        self.detector = YOLOPenDetector(model_path)

        # 캘리브레이션 로드
        calib_path = os.path.join(self.script_dir, "config", "calibration_eye_to_hand.npz")
        self.calib_path = calib_path

        data = np.load(calib_path)
        self.R_axes = data['R_axes']
        self.t_offset = data['t_offset'].copy()

        print(f"현재 t_offset: {self.t_offset * 1000} mm")
        print(f"현재 Z offset: {self.t_offset[2] * 1000:.1f} mm")

        # ROS2 노드
        self.node = None
        self.cli_get_posx = None
        self.running = True

        if HAS_ROS2:
            rclpy.init()
            self.node = rclpy.create_node('z_offset_calibrator')
            self.cli_get_posx = self.node.create_client(
                GetCurrentPosx, '/dsr01/aux_control/get_current_posx')

            if not self.cli_get_posx.wait_for_service(timeout_sec=5.0):
                print("[Warning] get_current_posx 서비스 연결 실패")
                self.cli_get_posx = None
            else:
                print("로봇 서비스 연결 완료")

            self.spin_thread = threading.Thread(target=self._spin, daemon=True)
            self.spin_thread.start()

    def _spin(self):
        while self.running and rclpy.ok():
            rclpy.spin_once(self.node, timeout_sec=0.1)

    def transform_to_robot(self, point_cam):
        """카메라 좌표 → 로봇 좌표 (현재 캘리브레이션)"""
        return self.R_axes @ point_cam + self.t_offset

    def get_current_tcp(self):
        """현재 TCP 위치 가져오기 (mm 단위)"""
        if self.cli_get_posx is None:
            return None

        req = GetCurrentPosx.Request()
        req.ref = 0  # DR_BASE

        future = self.cli_get_posx.call_async(req)

        timeout = time.time() + 2.0
        while not future.done() and time.time() < timeout:
            time.sleep(0.05)

        if future.done():
            result = future.result()
            if result and result.success and len(result.task_pos_info) > 0:
                return np.array(list(result.task_pos_info[0].data)[:3])  # X, Y, Z (mm)
        return None

    def run(self):
        print("\n" + "=" * 60)
        print("Z 오프셋 보정")
        print("=" * 60)
        print("\n조작:")
        print("  c: 현재 감지된 펜 좌표 저장 (카메라)")
        print("  r: 현재 로봇 TCP 좌표 저장")
        print("  s: Z 오프셋 계산 및 저장")
        print("  q: 종료")
        print("\n방법:")
        print("  1. 펜을 놓고 'c' 눌러 카메라 좌표 저장")
        print("  2. 로봇을 펜 캡 위치에 수동으로 이동")
        print("  3. 'r' 눌러 로봇 좌표 저장")
        print("  4. 's' 눌러 Z 오프셋 계산 및 저장")
        print("=" * 60)

        if not self.detector.start():
            print("카메라 시작 실패!")
            return

        saved_cam = None  # 카메라 좌표 (미터)
        saved_robot = None  # 로봇 좌표 (mm)

        try:
            while True:
                result = self.detector.detect()
                color_image, _ = self.detector.get_last_frames()

                if color_image is None:
                    continue

                display = self.detector.visualize(color_image, result) if result else color_image.copy()

                # 상태 표시
                y_pos = 30

                # 카메라 좌표
                if result is not None and result.state == DetectionState.DETECTED:
                    cam_pos = result.grasp_point
                    if cam_pos is not None and np.linalg.norm(cam_pos) > 0.01:
                        robot_pos = self.transform_to_robot(cam_pos)

                        cv2.putText(display, f"Cam: [{cam_pos[0]*1000:.1f}, {cam_pos[1]*1000:.1f}, {cam_pos[2]*1000:.1f}] mm",
                                   (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                        y_pos += 25

                        cv2.putText(display, f"-> Robot (calc): [{robot_pos[0]*1000:.1f}, {robot_pos[1]*1000:.1f}, {robot_pos[2]*1000:.1f}] mm",
                                   (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                        y_pos += 25

                # 현재 TCP
                tcp = self.get_current_tcp()
                if tcp is not None:
                    cv2.putText(display, f"Robot TCP: [{tcp[0]:.1f}, {tcp[1]:.1f}, {tcp[2]:.1f}] mm",
                               (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    y_pos += 25

                # 저장된 값
                y_pos += 10
                if saved_cam is not None:
                    cv2.putText(display, f"Saved Cam: [{saved_cam[0]*1000:.1f}, {saved_cam[1]*1000:.1f}, {saved_cam[2]*1000:.1f}] mm",
                               (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 128, 0), 2)
                    y_pos += 25

                if saved_robot is not None:
                    cv2.putText(display, f"Saved Robot: [{saved_robot[0]:.1f}, {saved_robot[1]:.1f}, {saved_robot[2]:.1f}] mm",
                               (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 128, 255), 2)
                    y_pos += 25

                # Z 오프셋 차이
                if saved_cam is not None and saved_robot is not None:
                    calc_robot = self.transform_to_robot(saved_cam) * 1000  # mm
                    z_diff = saved_robot[2] - calc_robot[2]

                    cv2.putText(display, f"Z diff: {z_diff:.1f} mm (actual - calculated)",
                               (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    y_pos += 25

                    new_z_offset = self.t_offset[2] + z_diff / 1000
                    cv2.putText(display, f"New Z offset: {new_z_offset*1000:.1f} mm (was {self.t_offset[2]*1000:.1f})",
                               (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                cv2.imshow("Z Offset Calibration", display)
                key = cv2.waitKey(30) & 0xFF

                if key == ord('q'):
                    break

                elif key == ord('c'):
                    # 카메라 좌표 저장
                    if result is not None and result.state == DetectionState.DETECTED:
                        cam_pos = result.grasp_point
                        if cam_pos is not None and np.linalg.norm(cam_pos) > 0.01:
                            saved_cam = cam_pos.copy()
                            print(f"\n[저장] 카메라 좌표: {saved_cam * 1000} mm")
                        else:
                            print("[오류] 깊이 데이터 없음")
                    else:
                        print("[오류] 펜이 감지되지 않음")

                elif key == ord('r'):
                    # 로봇 좌표 저장
                    tcp = self.get_current_tcp()
                    if tcp is not None:
                        saved_robot = tcp.copy()
                        print(f"\n[저장] 로봇 TCP: {saved_robot} mm")
                    else:
                        print("[오류] TCP 좌표를 가져올 수 없음")

                elif key == ord('s'):
                    # Z 오프셋 계산 및 저장
                    if saved_cam is None or saved_robot is None:
                        print("[오류] 먼저 c와 r로 좌표를 저장하세요")
                        continue

                    calc_robot = self.transform_to_robot(saved_cam) * 1000  # mm
                    z_diff = saved_robot[2] - calc_robot[2]

                    print(f"\n계산된 로봇 좌표: {calc_robot}")
                    print(f"실제 로봇 좌표: {saved_robot}")
                    print(f"Z 차이: {z_diff:.1f} mm")

                    # t_offset 업데이트
                    old_z = self.t_offset[2]
                    self.t_offset[2] += z_diff / 1000

                    print(f"\nZ offset: {old_z*1000:.1f} → {self.t_offset[2]*1000:.1f} mm")

                    # 저장
                    data = dict(np.load(self.calib_path))
                    data['t_offset'] = self.t_offset
                    np.savez(self.calib_path, **data)

                    print(f"[저장 완료] {self.calib_path}")

        except KeyboardInterrupt:
            pass

        self.detector.stop()
        if self.node:
            self.running = False
            self.node.destroy_node()
            rclpy.shutdown()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    calibrator = ZOffsetCalibrator()
    calibrator.run()
