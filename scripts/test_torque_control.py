#!/usr/bin/env python3
"""
Doosan E0509 토크 제어 테스트 스크립트

이 스크립트는 실시간 토크 제어 기능을 테스트합니다.
기본적으로 중력 보상만 수행하여 로봇이 현재 자세를 유지합니다.

사용법:
    # ROS2 환경 설정 후 실행
    source /opt/ros/humble/setup.bash
    source ~/doosan_ws/install/setup.bash
    python3 test_torque_control.py

주의:
    - 실제 로봇에서 실행 시 주의하세요!
    - 비상정지 버튼을 준비하세요.
    - 처음 실행 시 --dry-run 옵션으로 테스트하세요.
"""

import numpy as np
import time
import argparse
import signal
import sys

from robot_interface import DoosanRobot, RobotStateRt


class TorqueControlTest:
    """토크 제어 테스트 클래스"""

    def __init__(self, robot: DoosanRobot, control_freq: float = 100.0):
        """
        Args:
            robot: DoosanRobot 인스턴스
            control_freq: 제어 주파수 (Hz), 실제 권장값은 1000Hz
        """
        self.robot = robot
        self.control_freq = control_freq
        self.dt = 1.0 / control_freq
        self.running = False

        # 통계
        self.loop_count = 0
        self.max_loop_time = 0.0
        self.total_loop_time = 0.0

    def gravity_compensation_test(self, duration: float = 5.0):
        """
        중력 보상 테스트

        로봇이 현재 자세를 유지합니다.
        gravity_torque를 그대로 출력하면 로봇이 제자리에 있어야 합니다.

        Args:
            duration: 테스트 시간 (초)
        """
        print("\n" + "=" * 60)
        print("중력 보상 테스트")
        print("=" * 60)
        print(f"제어 주파수: {self.control_freq} Hz")
        print(f"테스트 시간: {duration} 초")
        print("로봇이 현재 자세를 유지해야 합니다.")
        print("Ctrl+C로 중단할 수 있습니다.")
        print("=" * 60)

        # 실시간 제어 시작
        if not self.robot.start_rt_control():
            print("[ERROR] 실시간 제어 시작 실패")
            return False

        self.running = True
        start_time = time.time()
        self.loop_count = 0
        self.max_loop_time = 0.0
        self.total_loop_time = 0.0

        try:
            while self.running and (time.time() - start_time) < duration:
                loop_start = time.time()

                # 상태 읽기
                state = self.robot.read_rt_state_fast()
                if state is None:
                    # 빠른 읽기 실패 시 느린 방법 시도
                    state = self.robot.read_rt_state()

                if state is not None:
                    # 중력 보상 토크 출력
                    torque = state.gravity_torque.copy()
                    self.robot.set_torque(torque)

                    # 상태 출력 (10Hz)
                    if self.loop_count % int(self.control_freq / 10) == 0:
                        elapsed = time.time() - start_time
                        print(f"\r[{elapsed:.1f}s] 토크: {np.round(torque, 2)}", end="", flush=True)

                # 루프 통계
                loop_time = time.time() - loop_start
                self.max_loop_time = max(self.max_loop_time, loop_time)
                self.total_loop_time += loop_time
                self.loop_count += 1

                # 대기 (목표 주기 맞추기)
                sleep_time = self.dt - loop_time
                if sleep_time > 0:
                    time.sleep(sleep_time)

        except KeyboardInterrupt:
            print("\n\n[INFO] 사용자에 의해 중단됨")

        finally:
            # 실시간 제어 종료
            self.robot.stop_rt_control()
            self.running = False

        # 결과 출력
        print("\n\n" + "=" * 60)
        print("테스트 결과")
        print("=" * 60)
        print(f"총 루프 수: {self.loop_count}")
        if self.loop_count > 0:
            avg_loop_time = self.total_loop_time / self.loop_count
            actual_freq = 1.0 / avg_loop_time if avg_loop_time > 0 else 0
            print(f"평균 루프 시간: {avg_loop_time * 1000:.2f} ms")
            print(f"최대 루프 시간: {self.max_loop_time * 1000:.2f} ms")
            print(f"실제 제어 주파수: {actual_freq:.1f} Hz")
        print("=" * 60)

        return True

    def read_state_test(self, count: int = 10):
        """
        상태 읽기 테스트

        실시간 상태 읽기가 정상 작동하는지 확인합니다.

        Args:
            count: 읽기 횟수
        """
        print("\n" + "=" * 60)
        print("상태 읽기 테스트")
        print("=" * 60)

        for i in range(count):
            print(f"\n[{i+1}/{count}] 상태 읽기...")
            start = time.time()
            state = self.robot.read_rt_state()
            elapsed = time.time() - start

            if state is not None:
                print(f"  읽기 시간: {elapsed * 1000:.1f} ms")
                print(f"  관절 위치 (deg): {np.degrees(state.joint_position).round(1)}")
                print(f"  관절 속도 (deg/s): {np.degrees(state.joint_velocity).round(1)}")
                print(f"  중력 토크 (Nm): {state.gravity_torque.round(2)}")
                print(f"  TCP 위치 (mm): {(state.tcp_position[:3] * 1000).round(1)}")
                print(f"  제어 모드: {state.control_mode}")
            else:
                print(f"  [ERROR] 상태 읽기 실패")

            time.sleep(0.5)

        print("\n" + "=" * 60)
        return True

    def stop(self):
        """테스트 중단"""
        self.running = False


def signal_handler(signum, frame):
    """SIGINT 핸들러"""
    print("\n\n[INFO] 종료 신호 수신...")
    sys.exit(0)


def main():
    parser = argparse.ArgumentParser(description='Doosan E0509 토크 제어 테스트')
    parser.add_argument('--ip', type=str, default='192.168.137.100',
                        help='로봇 IP 주소 (기본값: 192.168.137.100)')
    parser.add_argument('--namespace', type=str, default='dsr01',
                        help='ROS2 네임스페이스 (기본값: dsr01)')
    parser.add_argument('--freq', type=float, default=100.0,
                        help='제어 주파수 Hz (기본값: 100, 실제 권장: 1000)')
    parser.add_argument('--duration', type=float, default=5.0,
                        help='테스트 시간 초 (기본값: 5)')
    parser.add_argument('--test', type=str, default='gravity',
                        choices=['gravity', 'read', 'all'],
                        help='테스트 종류: gravity(중력보상), read(상태읽기), all(전체)')
    parser.add_argument('--dry-run', action='store_true',
                        help='실제 토크 명령 없이 상태 읽기만 테스트')

    args = parser.parse_args()

    # 시그널 핸들러 등록
    signal.signal(signal.SIGINT, signal_handler)

    print("\n" + "=" * 60)
    print("Doosan E0509 토크 제어 테스트")
    print("=" * 60)
    print(f"로봇 IP: {args.ip}")
    print(f"ROS2 네임스페이스: /{args.namespace}")
    print(f"제어 주파수: {args.freq} Hz")
    print(f"테스트: {args.test}")
    if args.dry_run:
        print("[DRY RUN] 토크 명령 없이 상태 읽기만 테스트")
    print("=" * 60)

    # 로봇 연결
    robot = DoosanRobot(
        ip=args.ip,
        use_ros2=True,
        ros2_namespace=args.namespace
    )

    if not robot.connect():
        print("[ERROR] 로봇 연결 실패")
        return 1

    print("[OK] 로봇 연결 성공")

    # 현재 상태 확인
    joint_pos = robot.get_joint_positions()
    print(f"현재 관절 위치 (deg): {np.degrees(joint_pos).round(1)}")

    # 테스트 실행
    tester = TorqueControlTest(robot, control_freq=args.freq)

    try:
        if args.test in ['read', 'all']:
            tester.read_state_test(count=5)

        if args.test in ['gravity', 'all'] and not args.dry_run:
            # 사용자 확인
            print("\n" + "=" * 60)
            print("[WARNING] 중력 보상 테스트를 시작합니다.")
            print("로봇에 토크 명령이 전송됩니다!")
            print("비상정지 버튼을 준비하세요.")
            print("=" * 60)
            response = input("계속하시겠습니까? (y/N): ")
            if response.lower() == 'y':
                tester.gravity_compensation_test(duration=args.duration)
            else:
                print("테스트 취소됨")

    except Exception as e:
        print(f"\n[ERROR] 테스트 중 오류: {e}")
        import traceback
        traceback.print_exc()

    finally:
        robot.disconnect()
        print("[OK] 로봇 연결 해제")

    return 0


if __name__ == "__main__":
    sys.exit(main())
