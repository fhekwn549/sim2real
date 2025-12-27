#!/usr/bin/env python3
"""
V7 Pen Grasp Sim2Real Controller

V7 학습 모델을 사용하여 실제 로봇으로 펜을 잡는 컨트롤러입니다.

핵심 기능:
1. Hybrid Control: 거리에 따라 Closed-Loop ↔ Open-Loop 자동 전환 (Freeze mode)
2. ActionProcessor: scale-by-dist로 목표 근처에서 부드러운 접근
3. Jacobian IK: Isaac Lab V7과 동일한 DLS 방식 (lambda=0.05)
4. 다단계 시퀀스: RL 접근 → 그리퍼 닫기 → 들어올리기

=== 실행 방법 ===

터미널 1: Doosan 로봇 드라이버
    source /opt/ros/humble/setup.bash
    source ~/doosan_ws/install/setup.bash
    ros2 launch dsr_bringup2 dsr_bringup2_real.launch.py

터미널 2: RealSense 카메라
    ros2 launch realsense2_camera rs_launch.py

터미널 3: 그리퍼 서비스 (필요시)
    source ~/doosan_ws/install/setup.bash
    ros2 run e0509_gripper_description gripper_service_node

터미널 4: 펜 감지 노드 (CoWriteBot)
    source ~/CoWriteBot/install/setup.bash
    ros2 run cowritebot find_marker

터미널 5: Sim2Real 브릿지 (ROS2 환경)
    source /opt/ros/humble/setup.bash
    source ~/doosan_ws/install/setup.bash
    source ~/CoWriteBot/install/setup.bash
    cd ~/sim2real/sim2real
    python3 sim2real_bridge.py

터미널 6: 컨트롤러 실행 (일반 환경)
    cd ~/sim2real/sim2real
    python3 pen_grasp_controller.py --checkpoint /home/fhekwn549/ikv7/model_99999.pt
"""

import argparse
import numpy as np
import time
import signal
import json
import os
from typing import Optional, Tuple
from dataclasses import dataclass

from action_processor import ActionProcessor
from jacobian_ik import JacobianIK


# =============================================================================
# 설정
# =============================================================================
@dataclass
class ControllerConfig:
    """컨트롤러 설정"""
    # 성공 판정 threshold
    success_dist_m: float = 0.03      # 3cm
    success_perp_m: float = 0.01      # 1cm
    success_hold_steps: int = 10      # 10스텝 유지

    # Freeze mode (펜 위치 고정)
    freeze_distance_m: float = 0.10   # 10cm 이내에서 고정 모드

    # ActionProcessor
    scale_by_dist: bool = True
    scale_min: float = 0.1
    scale_range_cm: float = 10.0

    # 그리퍼 오프셋 (link_6 → grasp_point)
    # URDF 기준: base_center(0.048) + finger_dir(0.02) ≈ 0.07m
    gripper_offset_m: float = 0.07    # 약 7cm (gripper open 상태)

    # 제어 설정
    control_freq_hz: float = 30.0
    max_steps: int = 500
    action_scale: float = 0.03        # V7과 동일

    # ===== 안전장치 설정 =====
    # 작업 공간 한계 (로봇 베이스 좌표계, 미터)
    workspace_min: tuple = (-0.2, -0.5, 0.05)   # (X_min, Y_min, Z_min)
    workspace_max: tuple = (0.7, 0.5, 0.6)      # (X_max, Y_max, Z_max)

    # Z 높이 제한 (테이블 충돌 방지)
    min_z_height_m: float = 0.05      # 5cm 이하로 내려가지 않음

    # 진행 상황 모니터링
    no_progress_steps: int = 100      # 100스텝 동안 진전 없으면 중단
    no_progress_threshold_m: float = 0.01  # 1cm 이상 가까워져야 진전으로 인정

    # 비상 정지 조건
    max_dist_from_target_m: float = 0.8  # 타겟에서 80cm 이상 멀어지면 중단

    # 파일 경로
    state_file: str = '/tmp/sim2real_state.json'
    command_file: str = '/tmp/sim2real_command.json'

    # URDF 경로
    urdf_path: str = "/home/fhekwn549/doosan_ws/src/doosan-robot2/dsr_description2/urdf/e0509.urdf"


# =============================================================================
# 로봇 인터페이스 (파일 기반)
# =============================================================================
class RobotInterface:
    """파일 기반 로봇 인터페이스"""

    def __init__(self, state_file: str, command_file: str):
        self.state_file = state_file
        self.command_file = command_file
        self._last_state = None

    def read_state(self) -> Optional[dict]:
        """로봇 상태 읽기"""
        if not os.path.exists(self.state_file):
            return self._last_state

        try:
            with open(self.state_file, 'r') as f:
                state = json.load(f)
            self._last_state = state
            return state
        except (json.JSONDecodeError, IOError):
            return self._last_state

    def get_joint_positions(self) -> np.ndarray:
        """관절 위치 (라디안)"""
        state = self.read_state()
        if state and 'joint_pos_rad' in state:
            return np.array(state['joint_pos_rad'])
        return np.zeros(6)

    def get_joint_velocities(self) -> np.ndarray:
        """관절 속도"""
        # 현재는 0으로 반환 (필요시 구현)
        return np.zeros(6)

    def get_tcp_position(self) -> np.ndarray:
        """TCP 위치 (미터)"""
        state = self.read_state()
        if state and 'tcp_pos_m' in state:
            return np.array(state['tcp_pos_m'])
        return np.zeros(3)

    def is_connected(self) -> bool:
        """브릿지 연결 확인"""
        state = self.read_state()
        if state and 'timestamp' in state:
            return (time.time() - state['timestamp']) < 5.0
        return False

    def write_command(self, command: dict):
        """명령 전송"""
        command['timestamp'] = time.time()
        try:
            with open(self.command_file, 'w') as f:
                json.dump(command, f)
        except IOError as e:
            print(f"[Error] 명령 전송 실패: {e}")

    def move_joint_delta(self, delta_rad: np.ndarray, vel: float = 60, acc: float = 60):
        """관절 델타 이동"""
        current = self.get_joint_positions()
        target_rad = current + delta_rad

        # 관절 한계 적용
        limits_lower = np.radians([-360, -95, -135, -360, -135, -360])
        limits_upper = np.radians([360, 95, 135, 360, 135, 360])
        target_rad = np.clip(target_rad, limits_lower, limits_upper)

        target_deg = np.degrees(target_rad).tolist()
        self.write_command({
            'type': 'move_joint',
            'target_deg': target_deg,
            'vel': vel,
            'acc': acc,
        })

    def gripper_open(self):
        self.write_command({'type': 'gripper_open'})

    def gripper_close(self):
        self.write_command({'type': 'gripper_close'})

    def go_home(self):
        self.write_command({'type': 'home'})


# =============================================================================
# 펜 인식기 Wrapper (파일 기반 - sim2real_bridge와 연동)
# =============================================================================
class PenTracker:
    """펜 위치/방향 추적 (Freeze mode 지원)

    sim2real_bridge.py에서 state 파일에 저장한 펜 위치를 읽어옵니다.
    """

    def __init__(self, state_file: str, freeze_distance: float = 0.10):
        self.state_file = state_file
        self.freeze_distance = freeze_distance
        self.frozen = False

        # 현재/고정 값
        self.current_cap_pos: Optional[np.ndarray] = None
        self.current_pen_z: Optional[np.ndarray] = None
        self.frozen_cap_pos: Optional[np.ndarray] = None
        self.frozen_pen_z: Optional[np.ndarray] = None

        print(f"[PenTracker] 파일 기반 모드 (state: {state_file})")

    def start(self):
        """시작 (호환성 유지)"""
        pass

    def stop(self):
        """종료 (호환성 유지)"""
        pass

    def update(self, gripper_pos: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """펜 위치/방향 업데이트

        Args:
            gripper_pos: 현재 그리퍼 위치 (거리 계산용)

        Returns:
            (cap_pos, pen_z): 사용할 펜 캡 위치와 축 방향
        """
        # state 파일에서 펜 정보 읽기
        pen_info = self._read_pen_from_state()

        if pen_info is not None and pen_info.get('detected', False):
            cap_pos = np.array(pen_info['cap_end_3d'])
            tip_pos = np.array(pen_info['tip_end_3d'])

            # 펜 축 방향 계산 (캡 → 팁 방향)
            pen_dir = tip_pos - cap_pos
            pen_z = pen_dir / (np.linalg.norm(pen_dir) + 1e-8)

            self.current_cap_pos = cap_pos
            self.current_pen_z = pen_z

        # 유효한 값이 없으면 기본값 사용
        if self.current_cap_pos is None:
            self.current_cap_pos = np.array([0.3, 0.0, 0.1])  # 기본 위치
            self.current_pen_z = np.array([0.0, 0.0, 1.0])    # 수직
            print("[PenTracker] 펜 미감지, 기본값 사용")

        # Freeze mode 체크
        if not self.frozen:
            dist = np.linalg.norm(self.current_cap_pos - gripper_pos)

            if dist < self.freeze_distance:
                # 고정 모드 전환
                self.frozen = True
                self.frozen_cap_pos = self.current_cap_pos.copy()
                self.frozen_pen_z = self.current_pen_z.copy()
                print(f"[PenTracker] Freeze mode 활성화 (dist={dist*100:.1f}cm)")

        # 고정 모드면 고정값 반환, 아니면 현재값 반환
        if self.frozen:
            return self.frozen_cap_pos, self.frozen_pen_z
        else:
            return self.current_cap_pos, self.current_pen_z

    def _read_pen_from_state(self) -> Optional[dict]:
        """state 파일에서 펜 정보 읽기"""
        if not os.path.exists(self.state_file):
            return None

        try:
            with open(self.state_file, 'r') as f:
                state = json.load(f)
            return state.get('pen', None)
        except (json.JSONDecodeError, IOError):
            return None

    def reset(self):
        """상태 리셋"""
        self.frozen = False
        self.frozen_cap_pos = None
        self.frozen_pen_z = None


# =============================================================================
# V7 Pen Grasp Controller
# =============================================================================
class PenGraspController:
    """V7 Pen Grasp Sim2Real 컨트롤러"""

    def __init__(self, checkpoint_path: str, config: ControllerConfig = None):
        self.config = config or ControllerConfig()
        self.running = False

        # 로봇 인터페이스
        self.robot = RobotInterface(
            self.config.state_file,
            self.config.command_file
        )

        # Policy 로드
        print(f"[Policy] 로드 중: {checkpoint_path}")
        self.policy = self._load_policy(checkpoint_path)

        # ActionProcessor
        self.processor = ActionProcessor(
            scale_by_dist=self.config.scale_by_dist,
            scale_min=self.config.scale_min,
            scale_range_cm=self.config.scale_range_cm,
        )

        # 펜 추적기 (파일 기반)
        self.pen_tracker = PenTracker(
            state_file=self.config.state_file,
            freeze_distance=self.config.freeze_distance_m
        )

        # Jacobian IK (Isaac Lab V7과 동일한 방식)
        self.ik = JacobianIK(urdf_path=self.config.urdf_path, lambda_val=0.05)

        print(f"[Controller] 초기화 완료")
        print(f"  - ActionProcessor: {self.processor}")
        print(f"  - Freeze distance: {self.config.freeze_distance_m*100:.0f}cm")
        print(f"  - Gripper offset: {self.config.gripper_offset_m*100:.0f}cm")
        print(f"  - IK method: {self.ik.method}")
        print(f"[Safety] 안전장치 활성화:")
        print(f"  - 작업공간 X: {self.config.workspace_min[0]*100:.0f}~{self.config.workspace_max[0]*100:.0f}cm")
        print(f"  - 작업공간 Y: {self.config.workspace_min[1]*100:.0f}~{self.config.workspace_max[1]*100:.0f}cm")
        print(f"  - 작업공간 Z: {self.config.workspace_min[2]*100:.0f}~{self.config.workspace_max[2]*100:.0f}cm")
        print(f"  - 최소 Z 높이: {self.config.min_z_height_m*100:.0f}cm")
        print(f"  - 진행없음 감지: {self.config.no_progress_steps}스텝 / {self.config.no_progress_threshold_m*100:.0f}cm")
        print(f"  - 최대 타겟 거리: {self.config.max_dist_from_target_m*100:.0f}cm")

    def _load_policy(self, checkpoint_path: str):
        """Policy 로드"""
        import torch
        import torch.nn as nn

        class SimpleActor(nn.Module):
            def __init__(self, obs_dim=27, action_dim=3, hidden_dims=[256, 256, 128]):
                super().__init__()
                layers = []
                in_dim = obs_dim
                for h_dim in hidden_dims:
                    layers.append(nn.Linear(in_dim, h_dim))
                    layers.append(nn.ELU())
                    in_dim = h_dim
                layers.append(nn.Linear(in_dim, action_dim))
                self.actor = nn.Sequential(*layers)

            def forward(self, obs):
                return self.actor(obs)

        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        state_dict = checkpoint['model_state_dict']

        policy = SimpleActor(obs_dim=27, action_dim=3)

        # Actor 가중치만 추출
        actor_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('actor.'):
                actor_state_dict[k] = v

        policy.load_state_dict(actor_state_dict)
        policy.eval()

        print(f"[Policy] 로드 완료 (obs=27, action=3)")
        return policy

    def build_observation(
        self,
        cap_pos: np.ndarray,
        pen_z: np.ndarray
    ) -> np.ndarray:
        """V7 형식 Observation 구성 (27차원)"""
        joint_pos = self.robot.get_joint_positions()
        joint_vel = self.robot.get_joint_velocities()
        gripper_pos = self.robot.get_tcp_position()

        # 상대 위치
        rel_pos = cap_pos - gripper_pos

        # 거리 계산
        distance_to_cap = np.linalg.norm(rel_pos)

        # Perpendicular distance 계산
        proj = np.dot(rel_pos, pen_z) * pen_z
        perp_vec = rel_pos - proj
        perp_dist = np.linalg.norm(perp_vec)

        # Phase (항상 0 = APPROACH)
        phase = 0.0

        obs = np.concatenate([
            joint_pos,          # 6
            joint_vel,          # 6
            gripper_pos,        # 3
            cap_pos,            # 3
            rel_pos,            # 3
            pen_z,              # 3
            [perp_dist],        # 1
            [distance_to_cap],  # 1
            [phase],            # 1
        ])  # 총 27

        return obs.astype(np.float32)

    def wait_for_bridge(self, timeout: float = 30.0) -> bool:
        """브릿지 연결 대기"""
        print("[Bridge] 연결 대기 중...")
        start = time.time()
        while time.time() - start < timeout:
            if self.robot.is_connected():
                print("[Bridge] 연결됨!")
                return True
            time.sleep(0.5)
            print(".", end="", flush=True)
        print("\n[Error] 브릿지 연결 타임아웃")
        return False

    def run(self) -> bool:
        """전체 펜 잡기 시퀀스 실행"""
        print("=" * 60)
        print("  V7 Pen Grasp Sim2Real Controller")
        print("=" * 60)

        # 브릿지 연결 확인
        if not self.wait_for_bridge():
            return False

        # 카메라 시작
        self.pen_tracker.start()

        # Home 이동 & 그리퍼 열기
        print("[Phase 0] 초기화...")
        self.robot.go_home()
        time.sleep(3.0)
        self.robot.gripper_open()
        time.sleep(1.0)

        # Phase 1: RL Policy로 접근
        print("[Phase 1] RL Policy로 펜 캡 접근...")
        success = self._phase_approach()

        if not success:
            print("[Failed] 접근 실패")
            self._cleanup()
            return False

        # Phase 2: 그리퍼 닫기
        print("[Phase 2] 그리퍼 닫기...")
        self.robot.gripper_close()
        time.sleep(1.0)

        # Phase 3: 들어올리기
        print("[Phase 3] 펜 들어올리기...")
        self._phase_lift()
        time.sleep(1.0)

        # 완료
        print("=" * 60)
        print("  펜 잡기 완료!")
        print("=" * 60)

        self._cleanup()
        return True

    def _check_safety(self, gripper_pos: np.ndarray, dist_to_cap: float) -> Tuple[bool, str]:
        """안전 조건 체크

        Args:
            gripper_pos: 현재 그리퍼 위치 [3] (미터)
            dist_to_cap: 타겟까지 거리 (미터)

        Returns:
            (is_safe, reason): 안전하면 (True, ""), 위험하면 (False, 이유)
        """
        # 1. Z 높이 제한 (테이블 충돌 방지)
        if gripper_pos[2] < self.config.min_z_height_m:
            return False, f"Z 높이 위험! ({gripper_pos[2]*100:.1f}cm < {self.config.min_z_height_m*100:.0f}cm)"

        # 2. 작업 공간 한계
        ws_min = self.config.workspace_min
        ws_max = self.config.workspace_max

        if gripper_pos[0] < ws_min[0] or gripper_pos[0] > ws_max[0]:
            return False, f"X 범위 초과! ({gripper_pos[0]*100:.1f}cm, 허용: {ws_min[0]*100:.0f}~{ws_max[0]*100:.0f}cm)"
        if gripper_pos[1] < ws_min[1] or gripper_pos[1] > ws_max[1]:
            return False, f"Y 범위 초과! ({gripper_pos[1]*100:.1f}cm, 허용: {ws_min[1]*100:.0f}~{ws_max[1]*100:.0f}cm)"
        if gripper_pos[2] > ws_max[2]:
            return False, f"Z 범위 초과! ({gripper_pos[2]*100:.1f}cm > {ws_max[2]*100:.0f}cm)"

        # 3. 타겟에서 너무 멀어짐
        if dist_to_cap > self.config.max_dist_from_target_m:
            return False, f"타겟에서 너무 멀어짐! ({dist_to_cap*100:.1f}cm > {self.config.max_dist_from_target_m*100:.0f}cm)"

        return True, ""

    def _phase_approach(self) -> bool:
        """Phase 1: RL Policy로 접근"""
        import torch

        dt = 1.0 / self.config.control_freq_hz
        hold_count = 0

        self.pen_tracker.reset()
        self.processor.reset()

        # 진행 상황 모니터링 변수
        best_dist = float('inf')
        no_progress_count = 0
        initial_dist = None

        for step in range(self.config.max_steps):
            loop_start = time.time()

            # 현재 관절 각도
            q = self.robot.get_joint_positions()

            # link_6 위치 및 방향 (FK)
            link6_pos, link6_rot = self.ik.forward_kinematics(q)

            # 그리퍼 Z축 방향 (link_6 Z축 = 그리퍼 방향)
            gripper_z = link6_rot[:, 2]

            # grasp_point = link_6 + gripper_offset * z_axis
            gripper_pos = link6_pos + gripper_z * self.config.gripper_offset_m

            # 펜 위치/방향 (Freeze mode 자동 적용)
            cap_pos, pen_z = self.pen_tracker.update(gripper_pos)

            # 거리 계산
            rel_pos = cap_pos - gripper_pos
            dist_to_cap = np.linalg.norm(rel_pos)

            # Perpendicular distance
            proj = np.dot(rel_pos, pen_z) * pen_z
            perp_dist = np.linalg.norm(rel_pos - proj)

            # ===== 안전 체크 =====
            # 초기 거리 기록
            if initial_dist is None:
                initial_dist = dist_to_cap
                best_dist = dist_to_cap
                print(f"  초기 거리: {initial_dist*100:.1f}cm")

            # 안전 조건 체크
            is_safe, safety_reason = self._check_safety(gripper_pos, dist_to_cap)
            if not is_safe:
                print(f"\n[SAFETY ABORT] {safety_reason}")
                return False

            # 진행 상황 모니터링
            if dist_to_cap < best_dist - self.config.no_progress_threshold_m:
                # 진전 있음
                best_dist = dist_to_cap
                no_progress_count = 0
            else:
                # 진전 없음
                no_progress_count += 1
                if no_progress_count >= self.config.no_progress_steps:
                    print(f"\n[NO PROGRESS] {self.config.no_progress_steps}스텝 동안 진전 없음 "
                          f"(best={best_dist*100:.1f}cm, current={dist_to_cap*100:.1f}cm)")
                    return False

            # 성공 조건 체크
            if dist_to_cap < self.config.success_dist_m and \
               perp_dist < self.config.success_perp_m:
                hold_count += 1
                if hold_count >= self.config.success_hold_steps:
                    print(f"\n[Success] 목표 도달! "
                          f"(dist={dist_to_cap*100:.1f}cm, perp={perp_dist*100:.1f}cm)")
                    return True
            else:
                hold_count = 0

            # Observation 구성
            obs = self.build_observation(cap_pos, pen_z)
            obs_tensor = torch.from_numpy(obs).unsqueeze(0)

            # Policy 추론
            with torch.no_grad():
                raw_action = self.policy(obs_tensor).numpy().squeeze()

            # ActionProcessor 적용 (scale-by-dist)
            action = self.processor.process(raw_action, dist_to_cap)

            # 로봇 명령 (V7: action은 위치 델타 [dx, dy, dz])
            # IK 또는 직접 TCP 이동 필요 - 여기서는 간단히 관절 델타로 변환
            # 실제 구현에서는 IK solver 사용 권장
            action_scaled = action * 0.03  # action_scale과 동일
            self._send_tcp_delta(action_scaled)

            # 로그
            if step % 30 == 0:
                freeze_str = "[FROZEN]" if self.pen_tracker.frozen else "[LIVE]"
                progress_pct = (1 - dist_to_cap / initial_dist) * 100 if initial_dist > 0 else 0
                print(f"  Step {step:3d}: dist={dist_to_cap*100:.1f}cm, "
                      f"perp={perp_dist*100:.1f}cm, hold={hold_count}, "
                      f"진행={progress_pct:.0f}% {freeze_str}")

            # 제어 주기 유지
            elapsed = time.time() - loop_start
            if elapsed < dt:
                time.sleep(dt - elapsed)

        print("\n[Timeout] 최대 스텝 도달")
        return False

    def _send_tcp_delta(self, delta_xyz: np.ndarray):
        """TCP 델타 이동 명령 (Jacobian IK 사용 - Isaac Lab V7과 동일)

        Args:
            delta_xyz: TCP 위치 델타 [3] (미터)
        """
        # 현재 관절 각도
        q = self.robot.get_joint_positions()

        # Jacobian IK로 관절 델타 계산 (DLS 방식, lambda=0.05)
        delta_q = self.ik.compute(q, delta_xyz)

        # 로봇 이동
        self.robot.move_joint_delta(delta_q)

    def _phase_lift(self):
        """Phase 3: 들어올리기"""
        # 현재 위치에서 5cm 위로
        for _ in range(10):
            self._send_tcp_delta(np.array([0.0, 0.0, 0.005]))  # 0.5cm씩 위로
            time.sleep(0.1)

    def _cleanup(self):
        """정리"""
        self.pen_tracker.stop()
        print("[Cleanup] 완료")


# =============================================================================
# Main
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description='V7 Pen Grasp Sim2Real Controller')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='V7 학습 모델 경로 (.pt)')
    parser.add_argument('--freeze_dist', type=float, default=10.0,
                        help='Freeze mode 전환 거리 (cm). Default: 10')
    parser.add_argument('--scale_min', type=float, default=0.1,
                        help='Scale-by-dist 최소값. Default: 0.1')
    parser.add_argument('--no_scale', action='store_true',
                        help='Scale-by-dist 비활성화')

    args = parser.parse_args()

    # 설정
    config = ControllerConfig(
        freeze_distance_m=args.freeze_dist / 100.0,
        scale_by_dist=not args.no_scale,
        scale_min=args.scale_min,
    )

    # 시그널 핸들러
    controller = None

    def signal_handler(sig, frame):
        print("\n[Signal] 종료 신호")
        if controller:
            controller.running = False

    signal.signal(signal.SIGINT, signal_handler)

    # 컨트롤러 실행
    controller = PenGraspController(args.checkpoint, config)
    success = controller.run()

    if success:
        print("\n펜 잡기 성공!")
    else:
        print("\n펜 잡기 실패")


if __name__ == '__main__':
    main()
