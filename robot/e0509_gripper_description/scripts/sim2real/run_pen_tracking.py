#!/usr/bin/env python3
"""
펜 추적 Sim2Real 실행 스크립트

학습된 정책을 실제 로봇에서 실행합니다.

=== 구조 ===
1. RealSense로 펜 위치 감지 (pen_detector.py)
2. 카메라 → 로봇 좌표 변환 (coordinate_transformer.py)
3. 로봇 상태 읽기 (sim2real_bridge.py의 파일)
4. Observation 구성 → Policy 추론 → Action
5. Action → 로봇 명령 (파일로 전송)

=== 사용법 ===
터미널 1: 로봇 bringup
    ros2 launch e0509_gripper_description bringup_dsr.launch.py

터미널 2: Sim2Real 브릿지 (ROS2)
    python3 sim2real_bridge.py

터미널 3: 펜 추적 실행 (이 스크립트)
    python3 run_pen_tracking.py --checkpoint /path/to/model.pt
"""

import argparse
import json
import os
import sys
import time
import glob
import numpy as np
from scipy.spatial.transform import Rotation as R

# 로컬 모듈
from pen_detector import PenDetector, DetectionConfig
from coordinate_transformer import CoordinateTransformer

# PyTorch
import torch
import torch.nn as nn

# 파일 기반 통신 경로
STATE_FILE = '/tmp/sim2real_state.json'
COMMAND_FILE = '/tmp/sim2real_command.json'

# 목표 거리 (학습 환경과 동일)
TARGET_DISTANCE = 0.05  # 5cm


class ActorNetwork(nn.Module):
    """RSL-RL Actor 네트워크"""

    def __init__(self, num_obs, num_actions, hidden_dims=[256, 256, 128]):
        super().__init__()

        layers = []
        in_dim = num_obs
        for out_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.ELU())
            in_dim = out_dim
        layers.append(nn.Linear(in_dim, num_actions))

        self.actor = nn.Sequential(*layers)

    def forward(self, obs):
        return self.actor(obs)


class PenTrackingController:
    """펜 추적 컨트롤러"""

    def __init__(self, checkpoint_path: str = None, device: str = 'cuda:0'):
        self.device = device

        # 컴포넌트 초기화
        print("[초기화] 펜 감지기...")
        self.pen_detector = PenDetector()

        print("[초기화] 좌표 변환기...")
        self.transformer = CoordinateTransformer()

        # 정책 로드
        print("[초기화] 정책 네트워크...")
        self.policy = ActorNetwork(
            num_obs=18,
            num_actions=6,
            hidden_dims=[256, 256, 128]
        ).to(device)

        if checkpoint_path and os.path.exists(checkpoint_path):
            self._load_checkpoint(checkpoint_path)
        else:
            print("[경고] 체크포인트 없음 - 랜덤 정책 사용")

        self.policy.eval()

        # 상태
        self.prev_joint_pos = None
        self.action_scale = 0.05  # 학습 환경과 동일

    def _load_checkpoint(self, path: str):
        """체크포인트 로드"""
        print(f"[로드] {path}")
        checkpoint = torch.load(path, map_location=self.device)

        state_dict = checkpoint.get("model_state_dict", checkpoint)
        actor_dict = {k: v for k, v in state_dict.items() if k.startswith("actor.")}

        self.policy.load_state_dict(actor_dict, strict=False)
        print("[로드] 완료!")

    def read_robot_state(self) -> dict:
        """로봇 상태 읽기 (파일에서)"""
        if not os.path.exists(STATE_FILE):
            return None

        try:
            with open(STATE_FILE, 'r') as f:
                state = json.load(f)
            return state
        except (json.JSONDecodeError, IOError):
            return None

    def send_command(self, target_joint_deg: list, vel: float = 30.0, acc: float = 30.0):
        """로봇 명령 전송 (파일로)"""
        command = {
            'type': 'move_joint',
            'target_deg': target_joint_deg,
            'vel': vel,
            'acc': acc,
            'timestamp': time.time(),
        }

        try:
            with open(COMMAND_FILE, 'w') as f:
                json.dump(command, f)
        except IOError as e:
            print(f"[오류] 명령 전송 실패: {e}")

    def get_tcp_pose_from_state(self, state: dict):
        """상태에서 TCP pose 추출"""
        tcp_pos_m = np.array(state.get('tcp_pos_m', [0.4, 0.0, 0.4]))

        # TCP 회전 (rx, ry, rz in rad → 회전 행렬)
        tcp_rot_rad = state.get('tcp_rot_rad', [0.0, 0.0, 0.0])
        # ZYX 오일러 각도 → 회전 행렬 (두산 로봇 기준)
        tcp_rot = R.from_euler('ZYX', tcp_rot_rad[::-1]).as_matrix()

        return tcp_pos_m, tcp_rot

    def build_observation(self, robot_state: dict, pen_pos_robot: np.ndarray) -> torch.Tensor:
        """Observation 구성 (학습 환경과 동일한 형식)"""
        # 로봇 관절 상태
        joint_pos = np.array(robot_state['joint_pos_rad'])

        # 관절 속도 계산 (간단한 차분)
        if self.prev_joint_pos is None:
            joint_vel = np.zeros(6)
        else:
            joint_vel = (joint_pos - self.prev_joint_pos) * 30  # 30Hz 가정
        self.prev_joint_pos = joint_pos.copy()

        # TCP 위치 (로봇 베이스 기준)
        tcp_pos, _ = self.get_tcp_pose_from_state(robot_state)

        # 상대 위치 (TCP → 펜)
        relative_pos = pen_pos_robot - tcp_pos

        # 목표 오프셋 (현재 방향으로 5cm)
        distance = np.linalg.norm(relative_pos) + 1e-6
        direction = relative_pos / distance
        target_offset = direction * TARGET_DISTANCE

        # Observation 구성 (18차원)
        obs = np.concatenate([
            joint_pos,      # 6
            joint_vel,      # 6
            relative_pos,   # 3
            target_offset,  # 3
        ])

        return torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)

    def run_step(self) -> bool:
        """단일 제어 스텝 실행"""
        # 1. 로봇 상태 읽기
        robot_state = self.read_robot_state()
        if robot_state is None:
            print("[경고] 로봇 상태 없음 - sim2real_bridge.py 실행 중인지 확인")
            return False

        # 2. 펜 위치 감지 (카메라 좌표)
        pen_pos_cam = self.pen_detector.get_pen_position_camera()
        if pen_pos_cam is None:
            print("[경고] 펜 감지 실패")
            return True  # 계속 실행

        # 3. 카메라 → 로봇 좌표 변환
        tcp_pos, tcp_rot = self.get_tcp_pose_from_state(robot_state)
        pen_pos_robot = self.transformer.camera_to_robot(pen_pos_cam, tcp_pos, tcp_rot)

        # 4. Observation 구성
        obs = self.build_observation(robot_state, pen_pos_robot)

        # 5. Policy 추론
        with torch.no_grad():
            action = self.policy(obs).cpu().numpy().flatten()

        # 6. Action → 관절 각도 변환
        current_joint_rad = np.array(robot_state['joint_pos_rad'])
        target_joint_rad = current_joint_rad + action * self.action_scale

        # 관절 한계 클램핑 (도산 E0509 기준)
        joint_limits_deg = [
            (-360, 360),   # joint_1
            (-95, 95),     # joint_2
            (-135, 135),   # joint_3
            (-360, 360),   # joint_4
            (-135, 135),   # joint_5
            (-360, 360),   # joint_6
        ]
        for i, (low, high) in enumerate(joint_limits_deg):
            target_joint_rad[i] = np.clip(
                target_joint_rad[i],
                np.radians(low),
                np.radians(high)
            )

        target_joint_deg = np.degrees(target_joint_rad).tolist()

        # 7. 로봇 명령 전송
        self.send_command(target_joint_deg, vel=50, acc=50)

        return True

    def start(self):
        """카메라 시작"""
        return self.pen_detector.start()

    def stop(self):
        """종료"""
        self.pen_detector.stop()


def find_latest_checkpoint():
    """최신 체크포인트 찾기"""
    log_dir = os.path.expanduser("~/IsaacLab/pen_grasp_rl/logs/pen_tracking")
    patterns = [
        os.path.join(log_dir, "**", "model_*.pt"),
        os.path.join(log_dir, "**", "*final*.pt"),
    ]
    checkpoints = []
    for pattern in patterns:
        checkpoints.extend(glob.glob(pattern, recursive=True))

    if checkpoints:
        return max(checkpoints, key=os.path.getctime)
    return None


def main():
    parser = argparse.ArgumentParser(description='펜 추적 Sim2Real')
    parser.add_argument('--checkpoint', '-c', type=str, default=None,
                        help='체크포인트 경로 (없으면 최신 자동 탐색)')
    parser.add_argument('--rate', type=float, default=30.0, help='제어 주기 (Hz)')
    parser.add_argument('--visualize', '-v', action='store_true', help='카메라 시각화')
    args = parser.parse_args()

    print("=" * 60)
    print("  펜 추적 Sim2Real")
    print("=" * 60)
    print(f"  목표 거리: {TARGET_DISTANCE*100:.1f}cm")
    print(f"  제어 주기: {args.rate}Hz")
    print("=" * 60)

    # 체크포인트 찾기
    checkpoint_path = args.checkpoint
    if checkpoint_path is None:
        checkpoint_path = find_latest_checkpoint()
        if checkpoint_path:
            print(f"[자동 탐색] {checkpoint_path}")

    # 컨트롤러 초기화
    controller = PenTrackingController(checkpoint_path=checkpoint_path)

    if not controller.start():
        print("[오류] 카메라 시작 실패")
        return

    print("\n준비 완료! 실행 중... (Ctrl+C로 종료)")
    print("=" * 60)

    dt = 1.0 / args.rate
    step_count = 0

    try:
        while True:
            loop_start = time.time()

            # 제어 스텝
            if controller.run_step():
                step_count += 1

                # 주기적 상태 출력
                if step_count % 30 == 0:
                    robot_state = controller.read_robot_state()
                    pen_pos = controller.pen_detector.get_pen_position_camera()

                    if robot_state and pen_pos is not None:
                        tcp_pos = robot_state.get('tcp_pos_m', [0, 0, 0])
                        tcp_pos, tcp_rot = controller.get_tcp_pose_from_state(robot_state)
                        pen_robot = controller.transformer.camera_to_robot(pen_pos, tcp_pos, tcp_rot)
                        distance = np.linalg.norm(pen_robot - tcp_pos)

                        print(f"[Step {step_count}]")
                        print(f"  TCP (robot):  [{tcp_pos[0]:.3f}, {tcp_pos[1]:.3f}, {tcp_pos[2]:.3f}]")
                        print(f"  Pen (robot):  [{pen_robot[0]:.3f}, {pen_robot[1]:.3f}, {pen_robot[2]:.3f}]")
                        print(f"  Distance:     {distance*100:.1f}cm (목표: {TARGET_DISTANCE*100:.1f}cm)")
                        print(f"  Error:        {abs(distance - TARGET_DISTANCE)*100:.1f}cm")

            # 시각화 (옵션)
            if args.visualize:
                if not controller.pen_detector.visualize(1):
                    break

            # 주기 유지
            elapsed = time.time() - loop_start
            if elapsed < dt:
                time.sleep(dt - elapsed)

    except KeyboardInterrupt:
        print("\n\n종료 중...")

    controller.stop()
    print(f"총 스텝: {step_count}")


if __name__ == '__main__':
    main()
