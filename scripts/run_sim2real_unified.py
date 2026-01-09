#!/usr/bin/env python3
"""
Sim2Real 통합 스크립트 - IK/OSC 모드 지원

강화학습으로 학습된 Policy를 실제 Doosan E0509 로봇에서 실행합니다.
IK(위치 제어)와 OSC(토크 제어) 두 가지 모드를 지원합니다.

=== 모드 비교 ===
- IK 모드: Jacobian IK → MoveJoint (위치 제어) - 안전, 느림
- OSC 모드: OSC 컨트롤러 → Torque (토크 제어) - 정확, 빠름

=== 사용법 ===
# IK 모드 (기존 방식, 안전하게 테스트)
python run_sim2real_unified.py --checkpoint model.pt --mode ik

# OSC 모드 (학습 환경과 동일)
python run_sim2real_unified.py --checkpoint model.pt --mode osc

=== 제어 흐름 ===
1. YOLO로 펜 캡 위치 + 방향 인식
2. 로봇 상태 + 펜 정보 → 27차원 Observation
3. Policy → 3차원 Action [Δx, Δy, Δz]
4. IK 모드: Jacobian IK → 관절 위치 명령
   OSC 모드: OSC 컨트롤러 → 토크 명령
"""

import argparse
import numpy as np
import torch
import torch.nn as nn
import time
import signal
import sys
import os
import cv2
from typing import Optional, Tuple
from dataclasses import dataclass
from enum import Enum

# 로컬 모듈
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from robot_interface import DoosanRobot, RobotStateRt
from osc_controller import OperationalSpaceController, euler_zyz_to_quat
from jacobian_ik import JacobianIK

# YOLO 펜 감지기
try:
    from pen_detector_yolo import YOLOPenDetector, DetectionState
    HAS_YOLO = True
except ImportError:
    HAS_YOLO = False
    print("[Warning] YOLO detector not available")


class ControlMode(Enum):
    """제어 모드"""
    IK = "ik"      # Jacobian IK + 위치 제어
    OSC = "osc"    # OSC + 토크 제어


# =============================================================================
# 설정
# =============================================================================
@dataclass
class Sim2RealConfig:
    """Sim2Real 설정"""
    # 모드
    control_mode: ControlMode = ControlMode.IK

    # Policy
    checkpoint_path: str = ""
    action_scale: float = 0.03  # 환경과 동일

    # 제어 주파수
    control_freq_ik: float = 10.0    # IK 모드 (Hz)
    control_freq_osc: float = 100.0  # OSC 모드 (Hz)

    # 최대 실행 시간
    max_duration: float = 60.0  # 초

    # 캘리브레이션
    calibration_path: str = ""

    # 성공 조건
    success_dist_to_cap: float = 0.03  # 3cm
    success_perp_dist: float = 0.01    # 1cm
    success_hold_steps: int = 30

    # 그리퍼 오프셋
    gripper_offset_z: float = 0.08  # TCP → Grasp point

    # 안전 설정
    disable_safety: bool = False
    max_tcp_delta: float = 0.05  # 5cm per step
    safety_min_z: float = 0.05   # 최소 Z 높이
    workspace_x: Tuple[float, float] = (-0.2, 0.8)
    workspace_y: Tuple[float, float] = (-0.5, 0.5)
    workspace_z: Tuple[float, float] = (0.05, 0.6)

    # Action 후처리
    smooth_alpha: float = 0.7
    dead_zone_cm: float = 2.0
    scale_by_dist: bool = True
    scale_min: float = 0.3
    scale_range_cm: float = 10.0

    # OSC 설정
    osc_stiffness: float = 150.0
    osc_damping_ratio: float = 1.0

    # YOLO
    yolo_model_path: str = "/home/fhekwn549/runs/segment/train/weights/best.pt"


# =============================================================================
# Action 후처리
# =============================================================================
class ActionProcessor:
    """Action 후처리 (Smoothing, Dead Zone, Scale by Distance)"""

    def __init__(self, config: Sim2RealConfig):
        self.smooth_alpha = config.smooth_alpha
        self.dead_zone = config.dead_zone_cm / 100.0
        self.scale_by_dist = config.scale_by_dist
        self.scale_min = config.scale_min
        self.scale_range = config.scale_range_cm / 100.0
        self.prev_action = None

    def process(self, action: np.ndarray, dist: float) -> np.ndarray:
        processed = action.copy()

        # Dead Zone
        if self.dead_zone > 0 and dist < self.dead_zone:
            return np.zeros_like(action)

        # Scale by Distance
        if self.scale_by_dist:
            scale = np.clip(dist / self.scale_range, self.scale_min, 1.0)
            processed = processed * scale

        # Smoothing
        if self.smooth_alpha < 1.0:
            if self.prev_action is None:
                self.prev_action = processed.copy()
            else:
                processed = self.smooth_alpha * processed + (1 - self.smooth_alpha) * self.prev_action
                self.prev_action = processed.copy()

        return processed

    def reset(self):
        self.prev_action = None


# =============================================================================
# Policy 네트워크
# =============================================================================
class PolicyNetwork(nn.Module):
    """Policy 네트워크 (27 → 3)"""

    def __init__(self):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(27, 256),
            nn.ELU(),
            nn.Linear(256, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, 3),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.actor(obs)


# =============================================================================
# 좌표 변환기
# =============================================================================
class CoordinateTransformer:
    """카메라 → 로봇 좌표 변환"""

    def __init__(self, calibration_path: str = None):
        self.R_axes = None
        self.t_offset = None

        if calibration_path:
            self.load(calibration_path)

    def load(self, path: str) -> bool:
        if not os.path.exists(path):
            print(f"[CoordTransform] 캘리브레이션 파일 없음: {path}")
            return False

        data = np.load(path)
        if 'R_axes' in data and 't_offset' in data:
            self.R_axes = data['R_axes']
            self.t_offset = data['t_offset']
            print(f"[CoordTransform] 로드 완료: {path}")
            return True
        return False

    def cam_to_robot(self, point_cam: np.ndarray) -> np.ndarray:
        if self.R_axes is None:
            return point_cam
        return self.R_axes @ point_cam + self.t_offset

    def direction_to_robot(self, dir_cam: np.ndarray) -> np.ndarray:
        if self.R_axes is None:
            return dir_cam
        dir_robot = self.R_axes @ dir_cam
        norm = np.linalg.norm(dir_robot)
        if norm > 0:
            dir_robot /= norm
        return dir_robot


# =============================================================================
# 통합 Sim2Real 컨트롤러
# =============================================================================
class Sim2RealUnified:
    """통합 Sim2Real 컨트롤러 (IK/OSC 모드 지원)"""

    def __init__(self, config: Sim2RealConfig):
        self.config = config
        self.running = False
        self.mode = config.control_mode

        print("=" * 60)
        print(f"Sim2Real 통합 컨트롤러 초기화")
        print(f"  모드: {self.mode.value.upper()}")
        print("=" * 60)

        # 1. Policy 로드
        print("\n[1/5] Policy 로드...")
        self.policy = self._load_policy(config.checkpoint_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy.to(self.device)
        self.policy.eval()
        print(f"  → Device: {self.device}")

        # 2. 로봇 연결
        print("\n[2/5] 로봇 연결...")
        self.robot = DoosanRobot(use_ros2=True)
        if not self.robot.connect():
            raise RuntimeError("로봇 연결 실패")

        # 3. 좌표 변환기
        print("\n[3/5] 좌표 변환기 로드...")
        calib_path = config.calibration_path or os.path.join(
            os.path.dirname(__file__), "config", "calibration_eye_to_hand.npz"
        )
        self.transformer = CoordinateTransformer(calib_path)

        # 4. 펜 감지기
        print("\n[4/5] YOLO 펜 감지기 시작...")
        if HAS_YOLO:
            self.detector = YOLOPenDetector(config.yolo_model_path)
            if not self.detector.start():
                raise RuntimeError("카메라 시작 실패")
        else:
            self.detector = None
            print("  [Warning] YOLO 없음 - 시뮬레이션 모드")

        # 5. 제어기 초기화 (모드별)
        print("\n[5/5] 제어기 초기화...")
        self._init_controller()

        # Action 후처리기
        self.action_processor = ActionProcessor(config)

        # 상태 변수
        self.success_hold_count = 0

        print("\n" + "=" * 60)
        print("초기화 완료!")
        print(f"  모드: {self.mode.value.upper()}")
        if self.mode == ControlMode.IK:
            print(f"  제어 주파수: {config.control_freq_ik} Hz")
            print(f"  제어 방식: Jacobian IK → MoveJoint")
        else:
            print(f"  제어 주파수: {config.control_freq_osc} Hz")
            print(f"  제어 방식: OSC → Torque")
            print(f"  OSC 강성: {config.osc_stiffness}")
        print("=" * 60)

    def _load_policy(self, checkpoint_path: str) -> PolicyNetwork:
        """Policy 로드"""
        policy = PolicyNetwork()
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        state_dict = checkpoint['model_state_dict']

        # actor 부분만 추출
        actor_dict = {k: v for k, v in state_dict.items()
                      if k.startswith('actor') or k == 'std'}
        policy.load_state_dict(actor_dict, strict=False)

        print(f"  → Checkpoint: {checkpoint_path}")
        print(f"  → Iteration: {checkpoint.get('iter', 'unknown')}")

        return policy

    def _init_controller(self):
        """모드별 제어기 초기화"""
        if self.mode == ControlMode.IK:
            # Jacobian IK
            self.ik = JacobianIK(lambda_val=0.05)
            self.osc = None
            print(f"  → Jacobian IK 초기화 완료")

        else:  # OSC 모드
            # OSC 컨트롤러
            stiffness = np.array([
                self.config.osc_stiffness,
                self.config.osc_stiffness,
                self.config.osc_stiffness,
                self.config.osc_stiffness / 3,
                self.config.osc_stiffness / 3,
                self.config.osc_stiffness / 3,
            ])
            self.osc = OperationalSpaceController(
                stiffness=stiffness,
                damping_ratio=self.config.osc_damping_ratio,
                inertial_dynamics_decoupling=True,
                gravity_compensation=True,
            )
            self.ik = JacobianIK(lambda_val=0.05)  # FK 계산용
            print(f"  → OSC 컨트롤러 초기화 완료")
            print(f"     강성: {stiffness[:3]}")

    def build_observation(
        self,
        joint_pos: np.ndarray,
        joint_vel: np.ndarray,
        grasp_pos: np.ndarray,
        cap_pos: np.ndarray,
        pen_z: np.ndarray,
    ) -> np.ndarray:
        """27차원 Observation 구성"""
        rel_pos = cap_pos - grasp_pos

        # 펜 축까지 거리
        grasp_to_cap = cap_pos - grasp_pos
        axis_distance = np.dot(grasp_to_cap, pen_z)
        projection = axis_distance * pen_z
        perp_vec = grasp_to_cap - projection
        perpendicular_dist = np.linalg.norm(perp_vec)
        distance_to_cap = np.linalg.norm(rel_pos)

        phase = 0.0  # APPROACH

        obs = np.concatenate([
            joint_pos,                    # 6
            joint_vel,                    # 6
            grasp_pos,                    # 3
            cap_pos,                      # 3
            rel_pos,                      # 3
            pen_z,                        # 3
            [perpendicular_dist],         # 1
            [distance_to_cap],            # 1
            [phase],                      # 1
        ]).astype(np.float32)             # 총 27

        return obs

    def _check_safety(self, tcp_pos: np.ndarray, cap_pos: np.ndarray) -> Tuple[bool, str]:
        """안전 체크"""
        if self.config.disable_safety:
            return True, "Safety disabled"

        # Z 높이
        if tcp_pos[2] < self.config.safety_min_z:
            return False, f"Z 높이 위험 ({tcp_pos[2]*100:.1f}cm)"

        # 작업 영역
        if not (self.config.workspace_x[0] <= tcp_pos[0] <= self.config.workspace_x[1]):
            return False, f"X 범위 초과"
        if not (self.config.workspace_y[0] <= tcp_pos[1] <= self.config.workspace_y[1]):
            return False, f"Y 범위 초과"
        if not (self.config.workspace_z[0] <= tcp_pos[2] <= self.config.workspace_z[1]):
            return False, f"Z 범위 초과"

        return True, "OK"

    def step_ik(self, pen_result: dict) -> dict:
        """IK 모드 스텝 (위치 제어)"""
        # 로봇 상태
        joint_pos = self.robot.get_joint_positions()
        joint_vel = self.robot.get_joint_velocities()
        tcp_pos, tcp_rot = self.robot.get_tcp_pose()

        if np.linalg.norm(tcp_pos) < 0.01:
            return {'skip': True}

        # Grasp point
        grasp_pos = tcp_pos.copy()
        grasp_pos[2] -= self.config.gripper_offset_z

        # 펜 정보
        cap_pos = pen_result['cap_robot']
        pen_z = pen_result.get('direction_robot', np.array([0, 0, 1]))

        # Observation
        obs = self.build_observation(joint_pos, joint_vel, grasp_pos, cap_pos, pen_z)

        # Policy 추론
        with torch.no_grad():
            obs_tensor = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)
            raw_action = self.policy(obs_tensor).cpu().numpy()[0]
            raw_action = np.clip(raw_action, -1.0, 1.0)

        # 거리 계산
        distance_to_cap = np.linalg.norm(cap_pos - grasp_pos)

        # Action 후처리
        action = self.action_processor.process(raw_action, distance_to_cap)

        # 자동 정지 (2cm 이내)
        if distance_to_cap < 0.02:
            action = np.zeros(3)

        # TCP 델타
        tcp_delta = action * self.config.action_scale
        tcp_delta = np.clip(tcp_delta, -self.config.max_tcp_delta, self.config.max_tcp_delta)

        # 안전 체크
        new_tcp_pos = tcp_pos + tcp_delta
        safe, msg = self._check_safety(new_tcp_pos, cap_pos)
        if not safe:
            return {'safety_stop': True, 'safety_msg': msg}

        # Jacobian IK
        delta_q = self.ik.compute(joint_pos, tcp_delta)
        delta_q = np.clip(delta_q, -np.deg2rad(1.5), np.deg2rad(1.5))

        new_joint_pos = joint_pos + delta_q
        new_joint_pos = self.robot.clamp_joint_positions(new_joint_pos)

        # 로봇 이동
        self.robot.move_joint(new_joint_pos, vel=20, acc=20, wait=False)

        # 결과
        return self._compute_result(action, cap_pos, pen_z, grasp_pos)

    def step_osc(self, pen_result: dict) -> dict:
        """OSC 모드 스텝 (토크 제어)"""
        # 실시간 상태 읽기
        state = self.robot.read_rt_state_fast()
        if state is None:
            state = self.robot.read_rt_state()
        if state is None:
            return {'skip': True}

        # 관절 상태
        joint_pos = state.joint_position
        joint_vel = state.joint_velocity

        # TCP 상태
        tcp_pos = state.tcp_position[:3]
        tcp_rot = state.tcp_position[3:]  # ZYZ Euler (rad)
        tcp_quat = euler_zyz_to_quat(tcp_rot[0], tcp_rot[1], tcp_rot[2])

        if np.linalg.norm(tcp_pos) < 0.01:
            return {'skip': True}

        # Grasp point
        grasp_pos = tcp_pos.copy()
        grasp_pos[2] -= self.config.gripper_offset_z

        # 펜 정보
        cap_pos = pen_result['cap_robot']
        pen_z = pen_result.get('direction_robot', np.array([0, 0, 1]))

        # Observation
        obs = self.build_observation(joint_pos, joint_vel, grasp_pos, cap_pos, pen_z)

        # Policy 추론
        with torch.no_grad():
            obs_tensor = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)
            raw_action = self.policy(obs_tensor).cpu().numpy()[0]
            raw_action = np.clip(raw_action, -1.0, 1.0)

        # 거리 계산
        distance_to_cap = np.linalg.norm(cap_pos - grasp_pos)

        # Action 후처리
        action = self.action_processor.process(raw_action, distance_to_cap)

        # 자동 정지 (2cm 이내)
        if distance_to_cap < 0.02:
            action = np.zeros(3)

        # 위치 델타
        pos_delta = action * self.config.action_scale
        pos_delta = np.clip(pos_delta, -self.config.max_tcp_delta, self.config.max_tcp_delta)
        rot_delta = np.zeros(3)  # 회전 유지

        # 안전 체크
        new_tcp_pos = tcp_pos + pos_delta
        safe, msg = self._check_safety(new_tcp_pos, cap_pos)
        if not safe:
            return {'safety_stop': True, 'safety_msg': msg}

        # OSC 목표 설정
        self.osc.set_target_delta(pos_delta, rot_delta, tcp_pos, tcp_quat)

        # EE 속도 (간단히 0으로)
        ee_vel = np.zeros(6)

        # OSC 토크 계산
        torque = self.osc.compute(
            jacobian=state.jacobian_matrix,
            current_ee_pos=tcp_pos,
            current_ee_quat=tcp_quat,
            current_ee_vel=ee_vel,
            mass_matrix=state.mass_matrix,
            gravity=state.gravity_torque,
        )

        # 토크 명령 전송
        self.robot.set_torque(torque)

        # 결과
        return self._compute_result(action, cap_pos, pen_z, grasp_pos)

    def _compute_result(self, action, cap_pos, pen_z, grasp_pos) -> dict:
        """결과 계산 (공통)"""
        rel_pos = cap_pos - grasp_pos
        distance_to_cap = np.linalg.norm(rel_pos)

        axis_dist = np.dot(rel_pos, pen_z)
        perp_vec = rel_pos - axis_dist * pen_z
        perpendicular_dist = np.linalg.norm(perp_vec)
        on_correct_side = axis_dist > 0

        # 성공 조건
        success_condition = (
            distance_to_cap < self.config.success_dist_to_cap and
            perpendicular_dist < self.config.success_perp_dist and
            on_correct_side
        )

        if success_condition:
            self.success_hold_count += 1
        else:
            self.success_hold_count = 0

        success = self.success_hold_count >= self.config.success_hold_steps

        return {
            'action': action,
            'distance_to_cap': distance_to_cap,
            'perpendicular_dist': perpendicular_dist,
            'on_correct_side': on_correct_side,
            'success_hold': self.success_hold_count,
            'success': success,
        }

    def step(self, pen_result: dict) -> dict:
        """모드에 따른 스텝 실행"""
        if self.mode == ControlMode.IK:
            return self.step_ik(pen_result)
        else:
            return self.step_osc(pen_result)

    def run(self):
        """메인 제어 루프"""
        print("\n" + "=" * 60)
        print(f"Sim2Real 대기 모드 [{self.mode.value.upper()}]")
        print("=" * 60)
        print("조작:")
        print("  g: Policy 실행 시작")
        print("  h: Home 위치로 이동")
        print("  q: 종료")
        print("=" * 60)

        # 제어 주파수
        if self.mode == ControlMode.IK:
            dt = 1.0 / self.config.control_freq_ik
        else:
            dt = 1.0 / self.config.control_freq_osc

        self.running = True
        policy_running = False
        start_time = None
        step_count = 0
        fixed_pen_result = None
        reached_target = False
        min_distance = float('inf')

        # OSC 모드: 실시간 제어 시작
        rt_started = False

        try:
            while self.running:
                loop_start = time.time()

                # 펜 감지
                pen_result = None
                cap_robot = None
                dir_robot = None

                if self.detector is not None:
                    result = self.detector.detect()
                    color_image, _ = self.detector.get_last_frames()

                    if result is not None and result.state == DetectionState.DETECTED:
                        cap_cam = result.grasp_point
                        if cap_cam is not None and np.linalg.norm(cap_cam) > 0.01:
                            cap_robot = self.transformer.cam_to_robot(cap_cam)
                            if result.direction_3d is not None:
                                dir_robot = -self.transformer.direction_to_robot(result.direction_3d)
                            else:
                                dir_robot = np.array([0, 0, 1])
                            pen_result = {'cap_robot': cap_robot, 'direction_robot': dir_robot}

                    # 시각화
                    if color_image is not None:
                        display = self.detector.visualize(color_image.copy(), result)

                        # 상태 표시
                        mode_str = f"[{self.mode.value.upper()}]"
                        if policy_running:
                            elapsed = time.time() - start_time
                            status = f"{mode_str} RUNNING Step:{step_count} [{elapsed:.1f}s]"
                            color = (0, 255, 0)
                        else:
                            status = f"{mode_str} READY - Press 'g'"
                            color = (0, 200, 200)

                        cv2.putText(display, status, (10, display.shape[0] - 10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                        cv2.imshow("Sim2Real", display)

                key = cv2.waitKey(1) & 0xFF

                # 키 입력
                if key == ord('q'):
                    break
                elif key == ord('h'):
                    print("\n[Home] Home 위치로 이동...")
                    policy_running = False
                    fixed_pen_result = None
                    if rt_started:
                        self.robot.stop_rt_control()
                        rt_started = False
                    self.robot.move_to_home()
                    print("[Home] 완료")
                elif key == ord('g'):
                    if not policy_running and pen_result is not None:
                        fixed_pen_result = {
                            'cap_robot': pen_result['cap_robot'].copy(),
                            'direction_robot': pen_result['direction_robot'].copy(),
                        }
                        print(f"\n[Start] Policy 실행 시작! [{self.mode.value.upper()}]")
                        print(f"  펜 위치: {fixed_pen_result['cap_robot'] * 1000} mm")

                        # OSC 모드: 실시간 제어 시작
                        if self.mode == ControlMode.OSC:
                            if self.robot.start_rt_control():
                                rt_started = True
                                print("  [RT] 실시간 제어 시작됨")
                            else:
                                print("  [RT] 실시간 제어 시작 실패!")
                                continue

                        policy_running = True
                        reached_target = False
                        min_distance = float('inf')
                        start_time = time.time()
                        step_count = 0
                        self.success_hold_count = 0
                        self.action_processor.reset()

                # Policy 실행
                if policy_running and fixed_pen_result is not None:
                    if reached_target:
                        if step_count % 30 == 0:
                            print(f"\r  [도달] 대기 중... (Step {step_count})", end="")
                        step_count += 1
                    else:
                        info = self.step(fixed_pen_result)
                        step_count += 1

                        if info.get('skip'):
                            continue

                        if info.get('safety_stop'):
                            print(f"\n[안전 정지] {info.get('safety_msg')}")
                            policy_running = False
                            fixed_pen_result = None
                            if rt_started:
                                self.robot.stop_rt_control()
                                rt_started = False
                            continue

                        current_dist = info.get('distance_to_cap', 1.0)
                        if current_dist < min_distance:
                            min_distance = current_dist

                        if current_dist < 0.02:
                            reached_target = True
                            print(f"\n  [목표 도달!] dist={current_dist*100:.1f}cm")

                        if info.get('success'):
                            print(f"\n\n[성공!] {self.config.success_hold_steps} 스텝 유지")
                            policy_running = False
                            fixed_pen_result = None

                        if step_count % 10 == 0:
                            print(f"\r  Step {step_count}: dist={current_dist*100:.1f}cm", end="")

                    # 시간 제한
                    elapsed = time.time() - start_time
                    if elapsed > self.config.max_duration:
                        print(f"\n[시간 초과] {self.config.max_duration}초")
                        policy_running = False
                        fixed_pen_result = None

                # 제어 주기
                loop_elapsed = time.time() - loop_start
                if loop_elapsed < dt:
                    time.sleep(dt - loop_elapsed)

        except KeyboardInterrupt:
            print("\n\n[중단됨]")

        finally:
            # OSC 모드: 실시간 제어 종료
            if rt_started:
                self.robot.stop_rt_control()

            self.running = False
            cv2.destroyAllWindows()

        print("\n" + "=" * 60)
        print("종료")
        print("=" * 60)

    def shutdown(self):
        """종료"""
        self.running = False
        if self.detector:
            self.detector.stop()
        self.robot.disconnect()
        print("Sim2Real 종료")


# =============================================================================
# 메인
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="Sim2Real 통합 (IK/OSC)")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="학습된 모델 경로 (.pt)")
    parser.add_argument("--mode", type=str, default="ik", choices=["ik", "osc"],
                        help="제어 모드: ik (위치 제어), osc (토크 제어)")
    parser.add_argument("--calibration", type=str, default=None,
                        help="캘리브레이션 파일 경로")
    parser.add_argument("--yolo_model", type=str,
                        default="/home/fhekwn549/runs/segment/train/weights/best.pt",
                        help="YOLO 모델 경로")
    parser.add_argument("--duration", type=float, default=60.0,
                        help="최대 실행 시간 (초)")
    parser.add_argument("--freq-ik", type=float, default=10.0,
                        help="IK 모드 제어 주파수 (Hz)")
    parser.add_argument("--freq-osc", type=float, default=100.0,
                        help="OSC 모드 제어 주파수 (Hz)")
    parser.add_argument("--osc-stiffness", type=float, default=150.0,
                        help="OSC 강성")
    parser.add_argument("--osc-damping", type=float, default=1.0,
                        help="OSC 댐핑 비율")
    parser.add_argument("--gripper-offset", type=float, default=0.08,
                        help="그리퍼 오프셋 (미터)")
    parser.add_argument("--no-safety", action="store_true",
                        help="안전 체크 비활성화")

    args = parser.parse_args()

    # 설정
    config = Sim2RealConfig(
        control_mode=ControlMode(args.mode),
        checkpoint_path=args.checkpoint,
        calibration_path=args.calibration,
        yolo_model_path=args.yolo_model,
        max_duration=args.duration,
        control_freq_ik=args.freq_ik,
        control_freq_osc=args.freq_osc,
        osc_stiffness=args.osc_stiffness,
        osc_damping_ratio=args.osc_damping,
        gripper_offset_z=args.gripper_offset,
        disable_safety=args.no_safety,
    )

    # 컨트롤러 생성
    controller = Sim2RealUnified(config)

    # 시그널 핸들러
    def signal_handler(sig, frame):
        print("\n\n[Signal] 종료 신호")
        controller.running = False

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # 실행
    try:
        controller.run()
    finally:
        controller.shutdown()


if __name__ == "__main__":
    main()
