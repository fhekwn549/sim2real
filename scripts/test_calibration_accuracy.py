#!/usr/bin/env python3
"""
캘리브레이션 정확도 테스트

마커를 카메라에 보여주면서 실제 TCP 위치와 변환된 마커 위치를 비교합니다.
차이가 작을수록 캘리브레이션이 정확한 것입니다.
"""

import numpy as np
import cv2
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from calibrate_eye_to_hand import EyeToHandCalibrator, CalibrationConfig

# ROS2
try:
    import rclpy
    from rclpy.node import Node
    HAS_ROS2 = True
except ImportError:
    HAS_ROS2 = False

def get_tcp_from_ros2(namespace='dsr01'):
    """ROS2로 현재 TCP 위치 가져오기"""
    if not HAS_ROS2:
        return None
    
    try:
        if not rclpy.ok():
            rclpy.init()
        
        node = rclpy.create_node('calib_test_node')
        
        from dsr_msgs2.srv import GetCurrentPosx
        client = node.create_client(GetCurrentPosx, f'/{namespace}/aux_control/get_current_posx')
        
        if not client.wait_for_service(timeout_sec=2.0):
            node.destroy_node()
            return None
        
        req = GetCurrentPosx.Request()
        req.ref = 0
        
        future = client.call_async(req)
        
        start = time.time()
        while not future.done() and (time.time() - start) < 2.0:
            rclpy.spin_once(node, timeout_sec=0.1)
        
        if future.done():
            result = future.result()
            if result and result.success and len(result.task_pos_info) > 0:
                posx = list(result.task_pos_info[0].data)[:6]
                node.destroy_node()
                return np.array(posx[:3])  # mm 단위
        
        node.destroy_node()
        return None
    except Exception as e:
        print(f"TCP 읽기 실패: {e}")
        return None

def main():
    print("=" * 60)
    print("캘리브레이션 정확도 테스트")
    print("=" * 60)
    print("마커를 로봇 그리퍼에 붙이고 카메라에 보여주세요.")
    print("변환된 마커 위치와 실제 TCP 위치를 비교합니다.")
    print("차이(Error)가 작을수록 캘리브레이션이 정확합니다.")
    print("")
    print("조작:")
    print("  Space: 현재 위치에서 오차 측정")
    print("  q: 종료")
    print("=" * 60)
    
    # 캘리브레이션 로드
    script_dir = os.path.dirname(os.path.abspath(__file__))
    calib_path = os.path.join(script_dir, "config", "calibration_eye_to_hand.npz")
    
    if not os.path.exists(calib_path):
        print(f"캘리브레이션 파일 없음: {calib_path}")
        return
    
    data = np.load(calib_path)
    T_cam_to_base = data['T_cam_to_base']
    camera_matrix = data['camera_matrix']
    dist_coeffs = data['dist_coeffs']
    
    print(f"캘리브레이션 로드: {calib_path}")
    
    # 카메라 시작
    import pyrealsense2 as rs
    from cv2 import aruco
    
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipeline.start(config)
    
    # ArUco 설정 (자동 탐지)
    aruco_dicts = [
        (aruco.DICT_4X4_50, "4X4_50"),
        (aruco.DICT_5X5_50, "5X5_50"),
        (aruco.DICT_6X6_250, "6X6_250"),
        (aruco.DICT_ARUCO_ORIGINAL, "ORIGINAL"),
    ]
    
    marker_size = float(data['marker_size'])
    print(f"마커 크기: {marker_size}m")
    
    errors = []
    
    try:
        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue
            
            image = np.asanyarray(color_frame.get_data())
            display = image.copy()
            
            # 마커 감지 (여러 딕셔너리 시도)
            rvec, tvec = None, None
            detected_dict = None
            
            for dict_type, dict_name in aruco_dicts:
                aruco_dict = aruco.getPredefinedDictionary(dict_type)
                params = aruco.DetectorParameters()
                detector = aruco.ArucoDetector(aruco_dict, params)
                corners, ids, _ = detector.detectMarkers(image)
                
                if ids is not None and len(ids) > 0:
                    # Pose 추정
                    half = marker_size / 2
                    obj_pts = np.array([
                        [-half, half, 0], [half, half, 0],
                        [half, -half, 0], [-half, -half, 0]
                    ], dtype=np.float32)
                    
                    marker_corners = corners[0].reshape(4, 2)
                    success, rvec, tvec = cv2.solvePnP(obj_pts, marker_corners, camera_matrix, dist_coeffs)
                    
                    if success:
                        detected_dict = dict_name
                        cv2.drawFrameAxes(display, camera_matrix, dist_coeffs, rvec, tvec, marker_size * 0.5)
                        break
            
            if rvec is not None and tvec is not None:
                t_cam = tvec.flatten()  # 미터
                
                # 로봇 좌표계로 변환
                t_cam_h = np.append(t_cam, 1)
                t_robot = (T_cam_to_base @ t_cam_h)[:3]  # 미터
                t_robot_mm = t_robot * 1000  # mm
                
                # 실제 TCP 위치
                tcp_mm = get_tcp_from_ros2()
                
                # 표시
                cv2.putText(display, f"Dict: {detected_dict}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(display, f"Marker(cam): [{t_cam[0]:.3f}, {t_cam[1]:.3f}, {t_cam[2]:.3f}]m",
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(display, f"Marker->Robot: [{t_robot_mm[0]:.1f}, {t_robot_mm[1]:.1f}, {t_robot_mm[2]:.1f}]mm",
                           (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                
                if tcp_mm is not None:
                    cv2.putText(display, f"Actual TCP: [{tcp_mm[0]:.1f}, {tcp_mm[1]:.1f}, {tcp_mm[2]:.1f}]mm",
                               (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    
                    # 오차 계산
                    error = np.linalg.norm(t_robot_mm - tcp_mm)
                    error_xyz = t_robot_mm - tcp_mm
                    
                    color = (0, 255, 0) if error < 50 else (0, 165, 255) if error < 100 else (0, 0, 255)
                    cv2.putText(display, f"Error: {error:.1f}mm  (X:{error_xyz[0]:.1f}, Y:{error_xyz[1]:.1f}, Z:{error_xyz[2]:.1f})",
                               (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                else:
                    cv2.putText(display, "TCP: Not available (ROS2 연결 확인)",
                               (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            else:
                cv2.putText(display, "Marker NOT detected",
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            # 통계 표시
            if errors:
                avg_err = np.mean(errors)
                cv2.putText(display, f"Avg Error ({len(errors)} samples): {avg_err:.1f}mm",
                           (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
            
            # 화면 확대
            display = cv2.resize(display, (1280, 960))
            cv2.imshow("Calibration Accuracy Test", display)
            
            key = cv2.waitKey(30) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' '):
                # 현재 오차 기록
                if rvec is not None and tcp_mm is not None:
                    error = np.linalg.norm(t_robot_mm - tcp_mm)
                    errors.append(error)
                    print(f"Sample {len(errors)}: Error = {error:.1f}mm")
                    print(f"  Marker->Robot: {t_robot_mm}")
                    print(f"  Actual TCP:    {tcp_mm}")
    
    except KeyboardInterrupt:
        pass
    
    pipeline.stop()
    cv2.destroyAllWindows()
    
    if errors:
        print(f"\n=== 결과 ===")
        print(f"샘플 수: {len(errors)}")
        print(f"평균 오차: {np.mean(errors):.1f}mm")
        print(f"최소 오차: {np.min(errors):.1f}mm")
        print(f"최대 오차: {np.max(errors):.1f}mm")
        
        if np.mean(errors) > 50:
            print("\n[주의] 평균 오차가 50mm 이상입니다. 캘리브레이션 재수행을 권장합니다.")

if __name__ == "__main__":
    main()
