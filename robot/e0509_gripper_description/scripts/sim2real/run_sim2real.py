#!/usr/bin/env python3
"""
Sim2Real Policy 실행 스크립트

ROS2 브릿지와 파일 기반 통신으로 실제 로봇을 제어합니다.
Python 버전 충돌을 피하기 위해 ROS2 의존성이 없습니다.

=== 사전 준비 ===
# 터미널 1: ROS2 로봇 연결
source /opt/ros/humble/setup.bash
source ~/doosan_ws/install/setup.bash
ros2 launch e0509_gripper_description bringup.launch.py mode:=real host:=192.168.137.100

# 터미널 2: Sim2Real Bridge (ROS2 환경)
source /opt/ros/humble/setup.bash
source ~/doosan_ws/install/setup.bash
cd ~/doosan_ws/src/e0509_gripper_description/scripts/sim2real
python3 sim2real_bridge.py

# 터미널 3: Policy 실행 (Isaac Sim 환경 또는 일반 환경)
cd ~/doosan_ws/src/e0509_gripper_description/scripts/sim2real
python3 run_sim2real.py --checkpoint /path/to/model.pt

=== 사용법 ===
python3 run_sim2real.py --checkpoint /home/fhekwn549/simple_move/model_1999.pt
python3 run_sim2real.py --checkpoint /path/to/model.pt --fixed_target 0.4 0.0 0.3
"""

import argparse
import numpy as np
import time
import signal
import sys
import json
import os
from typing import Optional, Tuple

from policy_loader import PolicyLoader, ENV_CONFIGS

# 공유 파일 경로 (sim2real_bridge.py와 동일)
STATE_FILE = '/tmp/sim2real_state.json'
COMMAND_FILE = '/tmp/sim2real_command.json'


class RobotStateReader:
    """파일 기반 로봇 상태 읽기"""

    def __init__(self, state_file: str = STATE_FILE):
        self.state_file = state_file
        self._last_state = None
        self._last_mtime = 0

    def read_state(self) -> Optional[dict]:
        """최신 상태 읽기"""
        if not os.path.exists(self.state_file):
            return self._last_state

        try:
            mtime = os.path.getmtime(self.state_file)
            if mtime == self._last_mtime and self._last_state is not None:
                return self._last_state

            with open(self.state_file, 'r') as f:
                state = json.load(f)

            self._last_state = state
            self._last_mtime = mtime
            return state

        except (json.JSONDecodeError, IOError):
            return self._last_state

    def get_joint_positions_rad(self) -> np.ndarray:
        """관절 위치 (라디안)"""
        state = self.read_state()
        if state and 'joint_pos_rad' in state:
            return np.array(state['joint_pos_rad'])
        return np.zeros(6)

    def get_tcp_position_m(self) -> np.ndarray:
        """TCP 위치 (미터)"""
        state = self.read_state()
        if state and 'tcp_pos_m' in state:
            return np.array(state['tcp_pos_m'])
        return np.zeros(3)

    def is_connected(self) -> bool:
        """브릿지 연결 확인"""
        state = self.read_state()
        if state is None:
            return False
        # 5초 이내 업데이트가 있으면 연결된 것으로 간주
        if 'timestamp' in state:
            return (time.time() - state['timestamp']) < 5.0
        return False


class RobotCommandWriter:
    """파일 기반 로봇 명령 전송"""

    def __init__(self, command_file: str = COMMAND_FILE):
        self.command_file = command_file

    def write_command(self, command: dict):
        """명령 전송"""
        command['timestamp'] = time.time()
        try:
            with open(self.command_file, 'w') as f:
                json.dump(command, f)
        except IOError as e:
            print(f"[Error] 명령 전송 실패: {e}")

    def move_joint(self, target_deg: list, vel: float = 30, acc: float = 30):
        """관절 이동 명령"""
        self.write_command({
            'type': 'move_joint',
            'target_deg': target_deg,
            'vel': vel,
            'acc': acc,
        })

    def gripper_open(self):
        """그리퍼 열기"""
        self.write_command({'type': 'gripper_open'})

    def gripper_close(self):
        """그리퍼 닫기"""
        self.write_command({'type': 'gripper_close'})

    def go_home(self):
        """Home 위치 이동"""
        self.write_command({'type': 'home'})


class Sim2RealController:
    """Sim2Real 제어기"""

    # 관절 한계 (도)
    JOINT_LIMITS_DEG = {
        'lower': [-360, -95, -135, -360, -135, -360],
        'upper': [360, 95, 135, 360, 135, 360]
    }

    def __init__(self, args):
        self.args = args
        self.running = False

        # 관절 한계 (라디안)
        self.joint_limits_lower = np.radians(self.JOINT_LIMITS_DEG['lower'])
        self.joint_limits_upper = np.radians(self.JOINT_LIMITS_DEG['upper'])

        # 로봇 인터페이스 (파일 기반)
        self.state_reader = RobotStateReader()
        self.command_writer = RobotCommandWriter()

        # Policy 로드
        print(f'[Policy] 로드 중: {args.checkpoint}')
        self.policy = PolicyLoader(args.checkpoint, args.env_type)

        # 펜 감지기 (선택)
        self.pen_detector = None
        if args.use_camera:
            try:
                from pen_detector import PenDetector
                self.pen_detector = PenDetector()
                print('[Camera] PenDetector 초기화됨')
            except Exception as e:
                print(f'[Camera] PenDetector 로드 실패: {e}')

        print('[Controller] 초기화 완료')

    def wait_for_bridge(self, timeout: float = 30.0) -> bool:
        """브릿지 연결 대기"""
        print('[Bridge] sim2real_bridge.py 연결 대기 중...')
        print(f'        상태 파일: {STATE_FILE}')

        start = time.time()
        while time.time() - start < timeout:
            if self.state_reader.is_connected():
                print('[Bridge] 연결됨!')
                return True
            time.sleep(0.5)
            print('.', end='', flush=True)

        print('\n[Error] 브릿지 연결 타임아웃')
        print('        sim2real_bridge.py가 실행 중인지 확인하세요.')
        return False

    def build_observation(self, target_pos: np.ndarray) -> np.ndarray:
        """Observation 구성"""
        joint_pos = self.state_reader.get_joint_positions_rad()
        joint_vel = np.zeros(6)  # 속도는 현재 0으로
        tcp_pos = self.state_reader.get_tcp_position_m()

        env_type = self.args.env_type

        if env_type == "target_tracking":
            # 18차원: joint_pos(6) + joint_vel(6) + grasp_pos(3) + target_pos(3)
            obs = np.concatenate([
                joint_pos,
                joint_vel,
                tcp_pos,
                target_pos,
            ])
        elif env_type == "pen_grasp":
            # 36차원
            gripper_pos = np.zeros(4)
            gripper_vel = np.zeros(4)
            pen_pos = target_pos
            pen_quat = np.array([1, 0, 0, 0])

            obs = np.concatenate([
                joint_pos, gripper_pos,      # 10
                joint_vel, gripper_vel,      # 10
                tcp_pos,                     # 3
                pen_pos,                     # 3
                pen_quat,                    # 4
                pen_pos - tcp_pos,           # 3
                pen_pos - tcp_pos,           # 3
            ])
        else:
            # 기본: target_tracking
            obs = np.concatenate([joint_pos, joint_vel, tcp_pos, target_pos])

        return obs.astype(np.float32)

    def run(self, fixed_target: np.ndarray = None):
        """메인 제어 루프"""
        print('=' * 60)
        print('  Sim2Real Policy Execution')
        print('=' * 60)
        print(f'  Checkpoint: {self.args.checkpoint}')
        print(f'  Env type: {self.args.env_type}')
        print(f'  Duration: {self.args.duration}초')
        print(f'  Frequency: {self.args.freq}Hz')
        if fixed_target is not None:
            print(f'  Fixed target: {fixed_target}')
        print('=' * 60)

        # 브릿지 연결 확인
        if not self.wait_for_bridge():
            return

        # 카메라 시작
        if self.pen_detector:
            self.pen_detector.start()

        # Home 이동
        print('[Robot] Home 위치로 이동...')
        self.command_writer.go_home()
        time.sleep(3.0)

        # 그리퍼 열기
        print('[Robot] 그리퍼 열기...')
        self.command_writer.gripper_open()
        time.sleep(1.0)

        # simple_move 환경: Home TCP 위치 저장 후 +5cm 위로 목표 설정
        initial_tcp = self.state_reader.get_tcp_position_m()
        print(f'[Info] 초기 TCP 위치: {initial_tcp}')

        # simple_move 학습 태스크용 목표 (Home TCP에서 5cm 위)
        if fixed_target is None and self.args.env_type == 'target_tracking':
            # 학습시 사용된 목표와 동일하게 설정
            # Phase 1: 5cm 위로, Phase 2: 원위치로 (자동 전환은 policy가 처리)
            fixed_target = initial_tcp + np.array([0.0, 0.0, 0.05])
            print(f'[Info] simple_move 목표 (TCP+5cm): {fixed_target}')

        dt = 1.0 / self.args.freq
        max_steps = int(self.args.duration * self.args.freq)
        step = 0

        self.running = True

        print('[Loop] 제어 루프 시작 (Ctrl+C로 종료)')
        print('=' * 60)

        try:
            while self.running and step < max_steps:
                loop_start = time.time()

                # 타겟 위치 결정
                if fixed_target is not None:
                    target_pos = fixed_target
                elif self.pen_detector:
                    # 카메라로 펜 인식
                    cam_pos = self.pen_detector.get_pen_position_camera()
                    if cam_pos is not None:
                        # TODO: 카메라 좌표 → 로봇 좌표 변환 (calibration 사용)
                        target_pos = cam_pos  # 임시: 직접 사용
                    else:
                        target_pos = initial_tcp + np.array([0.0, 0.0, 0.05])
                else:
                    target_pos = initial_tcp + np.array([0.0, 0.0, 0.05])

                # Observation 구성
                obs = self.build_observation(target_pos)

                # Policy 추론
                action = self.policy.get_action(obs, apply_scale=True)

                # 현재 관절 위치
                current_pos_rad = self.state_reader.get_joint_positions_rad()

                # 타겟 관절 계산
                target_joint_rad = current_pos_rad + action[:6]
                target_joint_rad = np.clip(
                    target_joint_rad,
                    self.joint_limits_lower,
                    self.joint_limits_upper
                )

                # 명령 전송
                target_joint_deg = np.degrees(target_joint_rad).tolist()
                self.command_writer.move_joint(target_joint_deg, vel=60, acc=60)

                # 로그 출력
                if step % 30 == 0:
                    tcp_pos = self.state_reader.get_tcp_position_m()
                    dist = np.linalg.norm(tcp_pos - target_pos)
                    print(f'[Step {step:4d}] dist={dist*100:.1f}cm, '
                          f'tcp={tcp_pos.round(3)}, target={target_pos.round(3)}')

                step += 1

                # 제어 주기 유지
                elapsed = time.time() - loop_start
                if elapsed < dt:
                    time.sleep(dt - elapsed)

        except KeyboardInterrupt:
            print('\n[Interrupt] 사용자 중단')

        self.running = False

        # 정리
        print('[Cleanup] 종료 중...')

        if self.pen_detector:
            self.pen_detector.stop()

        self.command_writer.go_home()
        time.sleep(3.0)

        print('[Done] 완료')


def main():
    parser = argparse.ArgumentParser(description='Sim2Real Policy Execution')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='학습된 모델 경로 (.pt)')
    parser.add_argument('--env_type', type=str, default='target_tracking',
                       choices=list(ENV_CONFIGS.keys()),
                       help='환경 타입')
    parser.add_argument('--duration', type=float, default=60.0,
                       help='실행 시간 (초)')
    parser.add_argument('--freq', type=float, default=30.0,
                       help='제어 주파수 (Hz)')
    parser.add_argument('--fixed_target', type=float, nargs=3, default=None,
                       metavar=('X', 'Y', 'Z'),
                       help='고정 타겟 위치 (미터)')
    parser.add_argument('--use_camera', action='store_true',
                       help='RealSense 카메라로 펜 감지')

    args = parser.parse_args()

    # 시그널 핸들러
    controller = None

    def signal_handler(sig, frame):
        print('\n[Signal] 종료 신호 수신')
        if controller:
            controller.running = False

    signal.signal(signal.SIGINT, signal_handler)

    # 컨트롤러 생성 및 실행
    controller = Sim2RealController(args)

    fixed_target = np.array(args.fixed_target) if args.fixed_target else None
    controller.run(fixed_target)


if __name__ == '__main__':
    main()
