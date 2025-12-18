#!/usr/bin/env python3
"""
Robot Observation Collector 테스트 스크립트

실제 로봇에서 observation을 실시간으로 모니터링합니다.

Usage:
    # 로봇 실행 후
    python3 test_observation.py

    # Virtual mode 테스트
    python3 test_observation.py --ns dsr01
"""

import argparse
import time
import numpy as np
import rclpy

from robot_observation import RobotObservationCollector


def print_observation(collector: RobotObservationCollector, clear=True):
    """Observation을 포맷팅해서 출력"""
    if clear:
        print('\033[2J\033[H', end='')  # 화면 클리어

    robot_state = collector.get_robot_only_observation()

    print('=' * 70)
    print('  Robot Observation Monitor (Ctrl+C to exit)')
    print('=' * 70)

    # Joint Position
    print('\n[Joint Position (degree)]')
    arm_pos_deg = np.rad2deg(robot_state['joint_pos'][:6])
    print(f'  J1: {arm_pos_deg[0]:8.2f}°   J2: {arm_pos_deg[1]:8.2f}°   J3: {arm_pos_deg[2]:8.2f}°')
    print(f'  J4: {arm_pos_deg[3]:8.2f}°   J5: {arm_pos_deg[4]:8.2f}°   J6: {arm_pos_deg[5]:8.2f}°')

    gripper_pos = robot_state['joint_pos'][6:10]
    print(f'  Gripper: {gripper_pos[0]:.3f} rad (all 4 joints)')

    # Joint Velocity
    print('\n[Joint Velocity (deg/s)]')
    arm_vel_deg = np.rad2deg(robot_state['joint_vel'][:6])
    print(f'  J1: {arm_vel_deg[0]:8.2f}   J2: {arm_vel_deg[1]:8.2f}   J3: {arm_vel_deg[2]:8.2f}')
    print(f'  J4: {arm_vel_deg[3]:8.2f}   J5: {arm_vel_deg[4]:8.2f}   J6: {arm_vel_deg[5]:8.2f}')

    # TCP Position
    print('\n[TCP Position (mm)]')
    tcp_mm = robot_state['tcp_pos'] * 1000
    print(f'  X: {tcp_mm[0]:8.2f}   Y: {tcp_mm[1]:8.2f}   Z: {tcp_mm[2]:8.2f}')

    # EE Position (Grasp Point)
    print('\n[EE Position / Grasp Point (mm)]')
    ee_mm = robot_state['ee_pos'] * 1000
    print(f'  X: {ee_mm[0]:8.2f}   Y: {ee_mm[1]:8.2f}   Z: {ee_mm[2]:8.2f}')

    # Pen Info (카메라 데이터)
    print('\n[Pen Pose (camera data - placeholder)]')
    pen_pos = collector.get_pen_pos() * 1000
    pen_quat = collector.get_pen_orientation()
    print(f'  Position: X={pen_pos[0]:.1f}, Y={pen_pos[1]:.1f}, Z={pen_pos[2]:.1f} mm')
    print(f'  Orientation: w={pen_quat[0]:.3f}, x={pen_quat[1]:.3f}, y={pen_quat[2]:.3f}, z={pen_quat[3]:.3f}')

    # Full Observation
    obs = collector.get_observation()
    print(f'\n[Full Observation]')
    print(f'  Shape: {obs.shape}, dtype: {obs.dtype}')
    print(f'  Range: [{obs.min():.4f}, {obs.max():.4f}]')

    print('\n' + '=' * 70)
    print(f'  Updated: {time.strftime("%H:%M:%S")}')


def main():
    parser = argparse.ArgumentParser(description='Robot Observation 테스트')
    parser.add_argument('--ns', type=str, default='dsr01', help='로봇 namespace')
    parser.add_argument('--rate', type=float, default=2.0, help='업데이트 주기 (Hz)')
    parser.add_argument('--once', action='store_true', help='한 번만 출력')
    args = parser.parse_args()

    rclpy.init()

    collector = RobotObservationCollector(namespace=args.ns)

    print(f'로봇 서비스 연결 중... (namespace: {args.ns})')

    if not collector.wait_for_services(timeout=10.0):
        print('\n서비스 연결 실패!')
        print('다음을 확인하세요:')
        print('  1. 로봇이 실행 중인가?')
        print('  2. ros2 launch e0509_gripper_description bringup.launch.py ...')
        rclpy.shutdown()
        return

    print('연결 성공!\n')

    if args.once:
        print_observation(collector, clear=False)
    else:
        try:
            while rclpy.ok():
                print_observation(collector)
                time.sleep(1.0 / args.rate)
        except KeyboardInterrupt:
            print('\n\n종료합니다.')

    rclpy.shutdown()


if __name__ == '__main__':
    main()
