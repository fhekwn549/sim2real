#!/usr/bin/env python3
"""
Doosan E0509 OSC 제어 테스트 스크립트

OSC(Operational Space Controller)를 사용하여 로봇을 제어합니다.

테스트 모드:
1. hold: 현재 위치 유지 (OSC로 자세 유지)
2. move: 목표 위치로 이동 (작은 움직임)

사용법:
    # 위치 유지 테스트
    python3 test_osc_control.py --test hold --duration 5

    # 작은 움직임 테스트 (주의!)
    python3 test_osc_control.py --test move --delta 0.02

주의:
    - 실제 로봇에서 실행 시 비상정지 버튼을 준비하세요!
    - 처음에는 --dry-run으로 테스트하세요.
"""

import numpy as np
import time
import argparse
import signal
import sys

from robot_interface import DoosanRobot, RobotStateRt
from osc_controller import OperationalSpaceController, euler_zyz_to_quat


class OSCControlTest:
    """OSC 제어 테스트 클래스"""

    def __init__(
        self,
        robot: DoosanRobot,
        control_freq: float = 100.0,
        stiffness: np.ndarray = None,
        damping_ratio: float = 1.0,
    ):
        """
        Args:
            robot: DoosanRobot 인스턴스
            control_freq: 제어 주파수 (Hz)
            stiffness: OSC 강성 [pos_xyz(3), rot_xyz(3)]
            damping_ratio: 댐핑 비율
        """
        self.robot = robot
        self.control_freq = control_freq
        self.dt = 1.0 / control_freq
        self.running = False

        # OSC 컨트롤러 생성
        if stiffness is None:
            stiffness = np.array([150.0, 150.0, 150.0, 50.0, 50.0, 50.0])

        self.osc = OperationalSpaceController(
            stiffness=stiffness,
            damping_ratio=damping_ratio,
            inertial_dynamics_decoupling=True,
            gravity_compensation=True,
        )

        # 통계
        self.loop_count = 0
        self.max_loop_time = 0.0
        self.total_loop_time = 0.0
        self.position_errors = []

    def hold_position_test(self, duration: float = 5.0):
        """
        위치 유지 테스트

        OSC를 사용하여 현재 위치를 유지합니다.
        중력 보상 + 위치 제어.

        Args:
            duration: 테스트 시간 (초)
        """
        print("\n" + "=" * 60)
        print("OSC 위치 유지 테스트")
        print("=" * 60)
        print(f"제어 주파수: {self.control_freq} Hz")
        print(f"테스트 시간: {duration} 초")
        print(f"강성: {self.osc.stiffness}")
        print("Ctrl+C로 중단할 수 있습니다.")
        print("=" * 60)

        # 현재 자세 읽기
        state = self.robot.read_rt_state()
        if state is None:
            print("[ERROR] 로봇 상태를 읽을 수 없습니다.")
            return False

        # 현재 위치를 목표로 설정
        current_pos = state.tcp_position[:3]  # [x, y, z] in meters
        current_rot = state.tcp_position[3:]  # [a, b, c] ZYZ Euler in radians
        current_quat = euler_zyz_to_quat(current_rot[0], current_rot[1], current_rot[2])

        self.osc.set_target_pose(current_pos, current_quat)

        print(f"\n목표 위치: {current_pos * 1000} mm")
        print(f"목표 회전 (ZYZ): {np.degrees(current_rot)} deg")

        # 실시간 제어 시작
        if not self.robot.start_rt_control():
            print("[ERROR] 실시간 제어 시작 실패")
            return False

        self.running = True
        start_time = time.time()
        self._reset_stats()

        try:
            while self.running and (time.time() - start_time) < duration:
                loop_start = time.time()

                # 상태 읽기
                state = self.robot.read_rt_state_fast()
                if state is None:
                    state = self.robot.read_rt_state()

                if state is not None:
                    # 현재 자세
                    ee_pos = state.tcp_position[:3]
                    ee_rot = state.tcp_position[3:]
                    ee_quat = euler_zyz_to_quat(ee_rot[0], ee_rot[1], ee_rot[2])

                    # EE 속도 (간단히 0으로 가정, 실제로는 계산 필요)
                    ee_vel = np.zeros(6)

                    # OSC 토크 계산
                    torque = self.osc.compute(
                        jacobian=state.jacobian_matrix,
                        current_ee_pos=ee_pos,
                        current_ee_quat=ee_quat,
                        current_ee_vel=ee_vel,
                        mass_matrix=state.mass_matrix,
                        gravity=state.gravity_torque,
                    )

                    # 토크 명령 전송
                    self.robot.set_torque(torque)

                    # 위치 에러 기록
                    pos_error = np.linalg.norm(self.osc.target_pos - ee_pos)
                    self.position_errors.append(pos_error)

                    # 상태 출력 (10Hz)
                    if self.loop_count % int(self.control_freq / 10) == 0:
                        elapsed = time.time() - start_time
                        print(f"\r[{elapsed:.1f}s] 위치 에러: {pos_error*1000:.2f} mm, "
                              f"토크: {np.round(torque[:3], 1)}", end="", flush=True)

                self._update_stats(loop_start)

                # 주기 맞추기
                self._sleep_remaining(loop_start)

        except KeyboardInterrupt:
            print("\n\n[INFO] 사용자에 의해 중단됨")

        finally:
            self.robot.stop_rt_control()
            self.running = False

        self._print_results()
        return True

    def move_to_delta_test(self, delta: np.ndarray, move_time: float = 2.0):
        """
        상대 위치 이동 테스트

        현재 위치에서 delta만큼 이동합니다.

        Args:
            delta: (3,) 이동량 [dx, dy, dz] (m)
            move_time: 이동 시간 (초)
        """
        print("\n" + "=" * 60)
        print("OSC 상대 이동 테스트")
        print("=" * 60)
        print(f"이동량: {delta * 1000} mm")
        print(f"이동 시간: {move_time} 초")
        print("Ctrl+C로 중단할 수 있습니다.")
        print("=" * 60)

        # 현재 자세 읽기
        state = self.robot.read_rt_state()
        if state is None:
            print("[ERROR] 로봇 상태를 읽을 수 없습니다.")
            return False

        # 시작 위치
        start_pos = state.tcp_position[:3].copy()
        start_rot = state.tcp_position[3:].copy()
        start_quat = euler_zyz_to_quat(start_rot[0], start_rot[1], start_rot[2])

        # 목표 위치
        target_pos = start_pos + delta
        target_quat = start_quat.copy()  # 회전은 유지

        print(f"\n시작 위치: {start_pos * 1000} mm")
        print(f"목표 위치: {target_pos * 1000} mm")

        # 실시간 제어 시작
        if not self.robot.start_rt_control():
            print("[ERROR] 실시간 제어 시작 실패")
            return False

        self.running = True
        start_time = time.time()
        self._reset_stats()

        try:
            while self.running:
                loop_start = time.time()
                elapsed = time.time() - start_time

                # 보간된 목표 위치 (선형 보간)
                alpha = min(elapsed / move_time, 1.0)
                interp_pos = start_pos + alpha * delta
                self.osc.set_target_pose(interp_pos, target_quat)

                # 상태 읽기
                state = self.robot.read_rt_state_fast()
                if state is None:
                    state = self.robot.read_rt_state()

                if state is not None:
                    ee_pos = state.tcp_position[:3]
                    ee_rot = state.tcp_position[3:]
                    ee_quat = euler_zyz_to_quat(ee_rot[0], ee_rot[1], ee_rot[2])
                    ee_vel = np.zeros(6)

                    torque = self.osc.compute(
                        jacobian=state.jacobian_matrix,
                        current_ee_pos=ee_pos,
                        current_ee_quat=ee_quat,
                        current_ee_vel=ee_vel,
                        mass_matrix=state.mass_matrix,
                        gravity=state.gravity_torque,
                    )

                    self.robot.set_torque(torque)

                    pos_error = np.linalg.norm(interp_pos - ee_pos)
                    self.position_errors.append(pos_error)

                    if self.loop_count % int(self.control_freq / 10) == 0:
                        progress = alpha * 100
                        print(f"\r[{progress:.0f}%] 위치: {ee_pos * 1000}, "
                              f"에러: {pos_error*1000:.2f} mm", end="", flush=True)

                self._update_stats(loop_start)

                # 이동 완료 후 1초 유지
                if elapsed > move_time + 1.0:
                    break

                self._sleep_remaining(loop_start)

        except KeyboardInterrupt:
            print("\n\n[INFO] 사용자에 의해 중단됨")

        finally:
            self.robot.stop_rt_control()
            self.running = False

        self._print_results()
        return True

    def _reset_stats(self):
        """통계 초기화"""
        self.loop_count = 0
        self.max_loop_time = 0.0
        self.total_loop_time = 0.0
        self.position_errors = []

    def _update_stats(self, loop_start: float):
        """루프 통계 업데이트"""
        loop_time = time.time() - loop_start
        self.max_loop_time = max(self.max_loop_time, loop_time)
        self.total_loop_time += loop_time
        self.loop_count += 1

    def _sleep_remaining(self, loop_start: float):
        """남은 시간 대기"""
        loop_time = time.time() - loop_start
        sleep_time = self.dt - loop_time
        if sleep_time > 0:
            time.sleep(sleep_time)

    def _print_results(self):
        """결과 출력"""
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

        if len(self.position_errors) > 0:
            errors = np.array(self.position_errors)
            print(f"\n위치 에러:")
            print(f"  평균: {np.mean(errors) * 1000:.2f} mm")
            print(f"  최대: {np.max(errors) * 1000:.2f} mm")
            print(f"  최종: {errors[-1] * 1000:.2f} mm")

        print("=" * 60)

    def stop(self):
        """테스트 중단"""
        self.running = False


def signal_handler(signum, frame):
    """SIGINT 핸들러"""
    print("\n\n[INFO] 종료 신호 수신...")
    sys.exit(0)


def main():
    parser = argparse.ArgumentParser(description='Doosan E0509 OSC 제어 테스트')
    parser.add_argument('--ip', type=str, default='192.168.137.100',
                        help='로봇 IP 주소')
    parser.add_argument('--namespace', type=str, default='dsr01',
                        help='ROS2 네임스페이스')
    parser.add_argument('--freq', type=float, default=100.0,
                        help='제어 주파수 Hz')
    parser.add_argument('--duration', type=float, default=5.0,
                        help='테스트 시간 초 (hold 모드)')
    parser.add_argument('--test', type=str, default='hold',
                        choices=['hold', 'move'],
                        help='테스트 종류: hold(위치유지), move(이동)')
    parser.add_argument('--delta', type=float, default=0.02,
                        help='이동 거리 m (move 모드, 기본값 2cm)')
    parser.add_argument('--stiffness', type=float, default=150.0,
                        help='OSC 강성')
    parser.add_argument('--damping', type=float, default=1.0,
                        help='OSC 댐핑 비율')
    parser.add_argument('--dry-run', action='store_true',
                        help='토크 명령 없이 상태만 확인')

    args = parser.parse_args()

    signal.signal(signal.SIGINT, signal_handler)

    print("\n" + "=" * 60)
    print("Doosan E0509 OSC 제어 테스트")
    print("=" * 60)
    print(f"로봇 IP: {args.ip}")
    print(f"ROS2 네임스페이스: /{args.namespace}")
    print(f"제어 주파수: {args.freq} Hz")
    print(f"테스트: {args.test}")
    print(f"OSC 강성: {args.stiffness}")
    print(f"OSC 댐핑: {args.damping}")
    if args.dry_run:
        print("[DRY RUN] 토크 명령 없이 상태만 확인")
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

    # 강성 설정
    stiffness = np.array([args.stiffness, args.stiffness, args.stiffness,
                          args.stiffness/3, args.stiffness/3, args.stiffness/3])

    # 테스트 실행
    tester = OSCControlTest(
        robot,
        control_freq=args.freq,
        stiffness=stiffness,
        damping_ratio=args.damping
    )

    try:
        if args.dry_run:
            # Dry run: 상태만 읽기
            print("\n[DRY RUN] 로봇 상태 확인...")
            state = robot.read_rt_state()
            if state is not None:
                print(f"TCP 위치: {state.tcp_position[:3] * 1000} mm")
                print(f"중력 토크: {state.gravity_torque}")
                print(f"자코비안:\n{state.jacobian_matrix}")
                print(f"질량 행렬:\n{state.mass_matrix}")
            else:
                print("[ERROR] 상태 읽기 실패")

        elif args.test == 'hold':
            # 사용자 확인
            print("\n" + "=" * 60)
            print("[WARNING] OSC 위치 유지 테스트를 시작합니다.")
            print("로봇에 토크 명령이 전송됩니다!")
            print("비상정지 버튼을 준비하세요.")
            print("=" * 60)
            response = input("계속하시겠습니까? (y/N): ")
            if response.lower() == 'y':
                tester.hold_position_test(duration=args.duration)
            else:
                print("테스트 취소됨")

        elif args.test == 'move':
            # 사용자 확인
            print("\n" + "=" * 60)
            print("[WARNING] OSC 이동 테스트를 시작합니다.")
            print(f"로봇이 Z축으로 {args.delta * 1000} mm 이동합니다!")
            print("비상정지 버튼을 준비하세요.")
            print("=" * 60)
            response = input("계속하시겠습니까? (y/N): ")
            if response.lower() == 'y':
                delta = np.array([0, 0, args.delta])  # Z축 이동
                tester.move_to_delta_test(delta, move_time=2.0)
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
