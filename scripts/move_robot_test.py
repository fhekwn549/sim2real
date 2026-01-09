#!/usr/bin/env python3
"""
로봇 이동 테스트 스크립트

좌표계 확인용으로 로봇을 X, Y, Z 방향으로 이동시킵니다.

사용법:
    source /opt/ros/humble/setup.bash
    source ~/doosan_ws/install/setup.bash
    python3 move_robot_test.py

주의:
    - 로봇 주변에 장애물이 없는지 확인하세요
    - 비상 정지 버튼을 준비하세요
"""

import subprocess
import time
import sys

def call_ros2_service(cmd: str, timeout: float = 10.0) -> bool:
    """ROS2 서비스 호출"""
    bash_cmd = (
        f"source /opt/ros/humble/setup.bash && "
        f"source ~/doosan_ws/install/setup.bash && "
        f"{cmd}"
    )
    try:
        result = subprocess.run(
            ['bash', '-c', bash_cmd],
            capture_output=True, text=True, timeout=timeout
        )
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        print("  → 타임아웃")
        return False
    except Exception as e:
        print(f"  → 오류: {e}")
        return False


def get_current_pos():
    """현재 TCP 위치 읽기"""
    import re
    bash_cmd = (
        f"source /opt/ros/humble/setup.bash && "
        f"source ~/doosan_ws/install/setup.bash && "
        f"ros2 service call /dsr01/aux_control/get_current_posx "
        f"dsr_msgs2/srv/GetCurrentPosx '{{ref: 0}}'"
    )
    try:
        result = subprocess.run(
            ['bash', '-c', bash_cmd],
            capture_output=True, text=True, timeout=5.0
        )
        if result.returncode == 0:
            match = re.search(r'data=\[([^\]]+)\]', result.stdout)
            if match:
                values = [float(x.strip()) for x in match.group(1).split(',')]
                return values[:6]  # [x, y, z, a, b, c] in mm, deg
    except:
        pass
    return None


def move_relative(dx_mm: float, dy_mm: float, dz_mm: float, vel: float = 50.0):
    """상대 이동 (Base 좌표계 기준)"""
    # 현재 위치 읽기
    current = get_current_pos()
    if current is None:
        print("현재 위치를 읽을 수 없습니다.")
        return False

    # 목표 위치 계산
    target_x = current[0] + dx_mm
    target_y = current[1] + dy_mm
    target_z = current[2] + dz_mm
    target_a = current[3]
    target_b = current[4]
    target_c = current[5]

    print(f"  현재: X={current[0]:.1f}, Y={current[1]:.1f}, Z={current[2]:.1f} mm")
    print(f"  목표: X={target_x:.1f}, Y={target_y:.1f}, Z={target_z:.1f} mm")

    # movel 호출 (절대 위치)
    cmd = (
        f"ros2 service call /dsr01/motion/move_line dsr_msgs2/srv/MoveLine \""
        f"{{pos: [{target_x}, {target_y}, {target_z}, {target_a}, {target_b}, {target_c}], "
        f"vel: [{vel}, 10.0], acc: [{vel}, 10.0], time: 0.0, radius: 0.0, ref: 0, mode: 0}}\""
    )

    return call_ros2_service(cmd, timeout=15.0)


def main():
    print("=" * 60)
    print("  로봇 이동 테스트 (좌표계 확인용)")
    print("=" * 60)
    print("  Doosan 로봇 좌표계:")
    print("    X+ = 앞쪽 (로봇 정면)")
    print("    Y+ = 왼쪽")
    print("    Z+ = 위쪽")
    print("=" * 60)
    print("\n  이동 거리: 50mm (5cm)")
    print("\n  명령어:")
    print("    1 = X+ (앞으로)")
    print("    2 = X- (뒤로)")
    print("    3 = Y+ (왼쪽)")
    print("    4 = Y- (오른쪽)")
    print("    5 = Z+ (위로)")
    print("    6 = Z- (아래로)")
    print("    p = 현재 위치 출력")
    print("    h = Home 위치로 이동")
    print("    q = 종료")
    print("=" * 60)

    move_dist = 50.0  # mm

    while True:
        try:
            cmd = input("\n명령 입력: ").strip().lower()
        except KeyboardInterrupt:
            print("\n종료")
            break

        if cmd == 'q':
            break
        elif cmd == 'p':
            pos = get_current_pos()
            if pos:
                print(f"\n현재 위치:")
                print(f"  X = {pos[0]:.1f} mm")
                print(f"  Y = {pos[1]:.1f} mm")
                print(f"  Z = {pos[2]:.1f} mm")
                print(f"  A = {pos[3]:.1f} deg")
                print(f"  B = {pos[4]:.1f} deg")
                print(f"  C = {pos[5]:.1f} deg")
        elif cmd == 'h':
            print("\nHome 위치로 이동 중...")
            home_cmd = (
                "ros2 service call /dsr01/motion/move_joint dsr_msgs2/srv/MoveJoint \""
                "{pos: [0.0, 0.0, 90.0, 0.0, 0.0, 90.0], vel: 30.0, acc: 30.0, "
                "time: 0.0, radius: 0.0, mode: 0, blend_type: 0, sync_type: 0}\""
            )
            if call_ros2_service(home_cmd, timeout=20.0):
                print("  → 완료")
            else:
                print("  → 실패")
        elif cmd == '1':
            print("\nX+ (앞으로) 50mm 이동 중...")
            if move_relative(move_dist, 0, 0):
                print("  → 완료")
        elif cmd == '2':
            print("\nX- (뒤로) 50mm 이동 중...")
            if move_relative(-move_dist, 0, 0):
                print("  → 완료")
        elif cmd == '3':
            print("\nY+ (왼쪽) 50mm 이동 중...")
            if move_relative(0, move_dist, 0):
                print("  → 완료")
        elif cmd == '4':
            print("\nY- (오른쪽) 50mm 이동 중...")
            if move_relative(0, -move_dist, 0):
                print("  → 완료")
        elif cmd == '5':
            print("\nZ+ (위로) 50mm 이동 중...")
            if move_relative(0, 0, move_dist):
                print("  → 완료")
        elif cmd == '6':
            print("\nZ- (아래로) 50mm 이동 중...")
            if move_relative(0, 0, -move_dist):
                print("  → 완료")
        else:
            print("알 수 없는 명령입니다.")


if __name__ == "__main__":
    main()
