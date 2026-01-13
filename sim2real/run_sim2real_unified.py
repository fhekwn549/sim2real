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
from scipy.spatial.transform import Rotation as R

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

# 펜 작업 공간 설정
try:
    from config.pen_workspace import DEFAULT_PEN_WORKSPACE, calculate_tilt_from_direction
    HAS_WORKSPACE = True
except ImportError:
    HAS_WORKSPACE = False
    print("[Warning] pen_workspace not available")


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
    action_scale: float = 0.01  # 1cm per action unit

    # 제어 주파수
    control_freq_ik: float = 30.0    # IK 모드 (Hz) - 웨이포인트 수집 주파수
    control_freq_osc: float = 30.0   # OSC 모드 (Hz) - 낮을수록 느린 움직임

    # 스플라인 설정 (IK 모드)
    spline_batch_size: int = 50      # 한 번에 전송할 웨이포인트 수 (최대 100)
    spline_vel: float = 60.0         # 스플라인 속도 (도/초)
    spline_acc: float = 60.0         # 스플라인 가속도 (도/초^2)
    spline_skip: int = 1             # N개 중 1개만 저장 (1=전부, 2=절반, 3=1/3)

    # 최대 실행 시간
    max_duration: float = 0.0  # 0=무제한

    # 캘리브레이션
    calibration_path: str = ""

    # 성공 조건
    success_dist_to_cap: float = 0.03  # 3cm
    success_perp_dist: float = 0.01    # 1cm
    success_hold_steps: int = 30

    # 그리퍼 오프셋 (TCP에서 실제 그립 포인트까지 거리)
    gripper_offset_z: float = 0.07  # TCP에서 그립 포인트까지 오프셋 (7cm)

    # 안전 설정
    disable_safety: bool = False
    max_tcp_delta: float = 0.010  # 10mm per step (OSC용)
    max_tcp_delta_ik: float = 0.01  # 10mm per step (IK용, 30Hz면 300mm/s)
    safety_min_z: float = 0.05   # 최소 Z 높이
    workspace_x: Tuple[float, float] = (-0.2, 0.8)
    workspace_y: Tuple[float, float] = (-0.5, 0.5)
    workspace_z: Tuple[float, float] = (0.05, 0.9)  # 상한 90cm로 확장

    # 관절 제한 (초기 자세 대비 최대 변화량, 도)
    joint1_max_delta_deg: float = 45.0  # joint 1 최대 ±45도
    joint_limits_from_init: bool = True  # 초기 자세 기준 제한 활성화

    # Action 후처리
    smooth_alpha: float = 0.3  # 더 부드러운 움직임 (vel=30 수준)
    dead_zone_cm: float = 2.0
    scale_by_dist: bool = True
    scale_min: float = 0.3
    scale_range_cm: float = 10.0

    # OSC 설정
    osc_stiffness_pos: float = 200.0  # 위치 강성 (너무 높으면 충돌 감지 트리거)
    osc_stiffness_rot: float = 100.0  # 회전 강성
    osc_damping_ratio: float = 1.0

    # YOLO
    yolo_model_path: str = "/home/fhekwn549/runs/segment/train/weights/best.pt"

    # 외부 트리거 (UI 연동)
    trigger_file: str = None  # 트리거 파일 경로 (있으면 파일 감지 시 시작)


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
            print(f"  OSC 위치강성: {config.osc_stiffness_pos}, 회전강성: {config.osc_stiffness_rot}")
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
            # 위치/회전 강성 분리 설정
            stiffness = np.array([
                self.config.osc_stiffness_pos,  # 위치 X
                self.config.osc_stiffness_pos,  # 위치 Y
                self.config.osc_stiffness_pos,  # 위치 Z
                self.config.osc_stiffness_rot,  # 회전 X (높은 강성)
                self.config.osc_stiffness_rot,  # 회전 Y
                self.config.osc_stiffness_rot,  # 회전 Z
            ])
            self.osc = OperationalSpaceController(
                stiffness=stiffness,
                damping_ratio=self.config.osc_damping_ratio,
                inertial_dynamics_decoupling=True,
                gravity_compensation=True,
            )
            self.ik = JacobianIK(lambda_val=0.05)  # FK 계산용
            print(f"  → OSC 컨트롤러 초기화 완료")
            print(f"     위치 강성: {stiffness[:3]}, 회전 강성: {stiffness[3:]}")

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

    def _compute_rotation_to_pen(self, current_quat: np.ndarray, pen_direction: np.ndarray) -> np.ndarray:
        """
        펜 방향에 맞게 그리퍼를 정렬하기 위한 회전 델타 계산

        Args:
            current_quat: 현재 그리퍼 quaternion [w, x, y, z]
            pen_direction: 펜 방향 벡터 (정규화됨)

        Returns:
            rot_delta: axis-angle 회전 [rx, ry, rz] (rad)
        """
        # 그리퍼가 펜을 잡을 때, 그리퍼의 Z축이 펜 축과 정렬되어야 함
        # 펜이 아래를 향하면 그리퍼도 아래를 향해야 함

        # 현재 그리퍼 Z축 (quaternion에서 추출)
        # quat = [w, x, y, z]
        w, x, y, z = current_quat
        # 회전 행렬의 Z축 (3번째 열)
        current_z = np.array([
            2 * (x*z + w*y),
            2 * (y*z - w*x),
            1 - 2*(x*x + y*y)
        ])

        # 목표: 그리퍼 Z축을 -pen_direction으로 정렬 (펜 캡을 향해 접근)
        # 펜이 위를 향하면 그리퍼는 아래를 향해야 함
        target_z = -pen_direction
        target_z = target_z / (np.linalg.norm(target_z) + 1e-8)

        # 두 벡터 사이의 회전 계산
        current_z = current_z / (np.linalg.norm(current_z) + 1e-8)

        # 회전축 (cross product)
        axis = np.cross(current_z, target_z)
        axis_norm = np.linalg.norm(axis)

        if axis_norm < 1e-6:
            # 이미 정렬됨 또는 반대 방향
            dot = np.dot(current_z, target_z)
            if dot < 0:
                # 180도 회전 필요 - 임의의 수직 축 사용
                axis = np.array([1, 0, 0]) if abs(current_z[0]) < 0.9 else np.array([0, 1, 0])
                axis = np.cross(current_z, axis)
                axis = axis / np.linalg.norm(axis)
                angle = np.pi * 0.1  # 천천히 회전
            else:
                return np.zeros(3)  # 이미 정렬됨
        else:
            axis = axis / axis_norm
            # 회전 각도 (dot product)
            dot = np.clip(np.dot(current_z, target_z), -1, 1)
            angle = np.arccos(dot)

        # 회전 속도 제한 (최대 0.05 rad/step = ~3도/step)
        max_rot_delta = 0.05
        angle = np.clip(angle, -max_rot_delta, max_rot_delta)

        # Axis-angle 형태로 반환
        rot_delta = axis * angle

        return rot_delta

    def step_ik(self, pen_result: dict) -> dict:
        """IK 모드 스텝 (스플라인 웨이포인트 수집)"""
        # 웨이포인트 버퍼 초기화
        if not hasattr(self, '_spline_waypoints'):
            self._spline_waypoints = []
            self._spline_joint_pos = None  # 시뮬레이션용 현재 위치
            self._spline_step_count = 0    # 스텝 카운터
            self._total_waypoints = 0      # 총 웨이포인트 수

        # 로봇 상태 (실제 또는 시뮬레이션)
        if self._spline_joint_pos is None:
            # 첫 스텝: 실제 로봇 상태 읽기
            joint_pos = self.robot.get_joint_positions()
            joint_vel = self.robot.get_joint_velocities()
            tcp_pos, tcp_rot = self.robot.get_tcp_pose()
        else:
            # 시뮬레이션: ROS2 호출 없이 계산
            joint_pos = self._spline_joint_pos
            joint_vel = np.zeros(6)  # 시뮬레이션에서는 속도 0
            tcp_pos, tcp_rot = self.ik.forward_kinematics(joint_pos)

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

        # 목표 도달 시 웨이포인트 즉시 실행하고 종료 (5cm 이내)
        if distance_to_cap < 0.05:
            print(f"\n  [목표 도달] dist={distance_to_cap*100:.1f}cm - 웨이포인트 실행")
            if len(self._spline_waypoints) > 0:
                self.robot.move_spline_joint(
                    self._spline_waypoints,
                    vel=self.config.spline_vel,
                    acc=self.config.spline_acc,
                    wait=True
                )
                print(f"  [완료] 총 {self._total_waypoints}개 웨이포인트")
                self._spline_waypoints = []
                self._spline_joint_pos = None

            # 그리퍼 닫기
            print("  [Gripper] 그리퍼 닫는 중...")
            self.robot.gripper_close()
            time.sleep(2.0)  # 그리퍼 닫힐 때까지 대기
            print("  [Gripper] 완료!")

            # 자동 종료
            print("\n" + "=" * 60)
            print("  작업 완료! 프로그램을 종료합니다.")
            print("=" * 60)
            self.running = False

            return {
                'distance_to_cap': distance_to_cap,
                'success': True,
                'target_reached': True
            }

        # Action 후처리
        action = self.action_processor.process(raw_action, distance_to_cap)

        # TCP 위치 델타 (IK용 더 큰 delta 사용)
        tcp_delta = action * self.config.action_scale
        tcp_delta = np.clip(tcp_delta, -self.config.max_tcp_delta_ik, self.config.max_tcp_delta_ik)

        # 움직임이 거의 없으면 수렴으로 판단하고 종료
        if np.linalg.norm(tcp_delta) < 0.0005:  # 0.5mm 미만
            if not hasattr(self, '_no_move_count'):
                self._no_move_count = 0
            self._no_move_count += 1

            if self._no_move_count >= 10:  # 10번 연속 움직임 없음
                print(f"\n  [수렴] 움직임 없음 (dist={distance_to_cap*100:.1f}cm) - 웨이포인트 실행")
                if len(self._spline_waypoints) > 0:
                    self.robot.move_spline_joint(
                        self._spline_waypoints,
                        vel=self.config.spline_vel,
                        acc=self.config.spline_acc,
                        wait=True
                    )
                    print(f"  [완료] 총 {self._total_waypoints}개 웨이포인트")
                    self._spline_waypoints = []
                    self._spline_joint_pos = None

                # 그리퍼 닫기
                print("  [Gripper] 그리퍼 닫는 중...")
                self.robot.gripper_close()
                time.sleep(2.0)  # 그리퍼 닫힐 때까지 대기
                print("  [Gripper] 완료!")

                # 자동 종료
                print("\n" + "=" * 60)
                print("  작업 완료! 프로그램을 종료합니다.")
                print("=" * 60)
                self.running = False
                self._no_move_count = 0

                return {
                    'distance_to_cap': distance_to_cap,
                    'success': True,
                    'converged': True
                }
        else:
            self._no_move_count = 0

        # TCP 회전 델타 (펜 방향으로 정렬)
        # tcp_rot은 이미 위에서 계산됨 (rotation matrix [3,3] 또는 euler)
        if tcp_rot.shape == (3, 3):
            tcp_rot_matrix = tcp_rot
        else:
            # Euler angles인 경우 변환
            tcp_rot_matrix = R.from_euler('ZYZ', tcp_rot).as_matrix()
        # Rotation matrix를 quaternion으로 변환 (scipy: [x,y,z,w] → [w,x,y,z])
        quat_xyzw = R.from_matrix(tcp_rot_matrix).as_quat()
        tcp_quat = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])
        rot_delta = self._compute_rotation_to_pen(tcp_quat, pen_z)

        # 안전 체크
        new_tcp_pos = tcp_pos + tcp_delta
        safe, msg = self._check_safety(new_tcp_pos, cap_pos)
        if not safe:
            return {'safety_stop': True, 'safety_msg': msg}

        # Jacobian IK (위치 + 회전)
        delta_q = self.ik.compute(joint_pos, tcp_delta, rot_delta)
        delta_q = np.clip(delta_q, -np.deg2rad(5.0), np.deg2rad(5.0))

        new_joint_pos = joint_pos + delta_q
        new_joint_pos = self.robot.clamp_joint_positions(new_joint_pos)

        # 스텝 카운터 증가
        self._spline_step_count += 1
        self._spline_joint_pos = new_joint_pos.copy()

        # 결과 먼저 계산
        result = self._compute_result(action, cap_pos, pen_z, grasp_pos)

        # 웨이포인트 로깅
        dist = result.get('distance_to_cap', 1.0)
        tcp_new, _ = self.ik.forward_kinematics(new_joint_pos)
        joint_deg = np.degrees(new_joint_pos)

        # N개 중 1개만 저장 (skip 설정)
        if self._spline_step_count % self.config.spline_skip == 0:
            self._spline_waypoints.append(new_joint_pos.copy())
            self._total_waypoints += 1

            # 웨이포인트 로그 출력
            rot_deg = np.degrees(rot_delta)
            print(f"\r  [WP {self._total_waypoints:3d}] "
                  f"dist={dist*100:5.1f}cm | "
                  f"pos_d={np.linalg.norm(tcp_delta)*1000:4.1f}mm | "
                  f"rot_d=[{rot_deg[0]:5.2f},{rot_deg[1]:5.2f},{rot_deg[2]:5.2f}]°", end="")

        # 배치가 차면 스플라인 실행
        if len(self._spline_waypoints) >= self.config.spline_batch_size:
            print(f"\n  [Spline] {len(self._spline_waypoints)}개 웨이포인트 실행 중...")
            success = self.robot.move_spline_joint(
                self._spline_waypoints,
                vel=self.config.spline_vel,
                acc=self.config.spline_acc,
                wait=True
            )
            if success:
                print(f"  [Spline] 완료! (총 {self._total_waypoints}개 계산됨)")
            else:
                print(f"  [Spline] 실패")
            self._spline_waypoints = []
            self._spline_joint_pos = None  # 실제 로봇 위치로 리셋
            result['spline_executed'] = True

        return result

    def flush_spline_waypoints(self):
        """남은 웨이포인트 실행"""
        total = getattr(self, '_total_waypoints', 0)
        if hasattr(self, '_spline_waypoints') and len(self._spline_waypoints) > 0:
            print(f"\n  [Spline] 남은 {len(self._spline_waypoints)}개 웨이포인트 실행...")
            self.robot.move_spline_joint(
                self._spline_waypoints,
                vel=self.config.spline_vel,
                acc=self.config.spline_acc,
                wait=True
            )
            print(f"  [완료] 총 {total}개 웨이포인트 사용")
            self._spline_waypoints = []
            self._spline_joint_pos = None
        elif total > 0:
            print(f"\n  [완료] 총 {total}개 웨이포인트 사용")

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

        # 회전 델타 - 펜 방향에 맞게 그리퍼 정렬
        rot_delta = self._compute_rotation_to_pen(tcp_quat, pen_z)

        # 디버그 출력 (50스텝마다)
        if not hasattr(self, '_debug_count'):
            self._debug_count = 0
        self._debug_count += 1
        if self._debug_count % 50 == 0:
            rot_deg = np.degrees(rot_delta)
            print(f"[DEBUG] TCP: {tcp_pos*1000}mm, Grasp: {grasp_pos*1000}mm")
            print(f"[DEBUG] PenCap: {cap_pos*1000}mm, dist: {distance_to_cap*100:.1f}cm")
            print(f"[DEBUG] pos_delta: {pos_delta*1000}mm, rot_delta: {rot_deg}deg")

        # 안전 체크
        new_tcp_pos = tcp_pos + pos_delta
        safe, msg = self._check_safety(new_tcp_pos, cap_pos)
        if not safe:
            return {'safety_stop': True, 'safety_msg': msg}

        # Joint 1 회전 제한 체크
        if self.config.joint_limits_from_init and hasattr(self, '_init_joint_pos'):
            joint1_delta = np.degrees(abs(joint_pos[0] - self._init_joint_pos[0]))
            if joint1_delta > self.config.joint1_max_delta_deg:
                return {'safety_stop': True, 'safety_msg': f'Joint1 회전 초과: {joint1_delta:.1f}도 (최대 {self.config.joint1_max_delta_deg}도)'}

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

        # 토크 클램핑 (충돌 감지 방지, 각 관절 최대 40Nm)
        max_torque = 40.0
        torque = np.clip(torque, -max_torque, max_torque)

        # 토크 명령 전송
        self.robot.set_torque(torque)

        # 토크 디버그 (50스텝마다)
        if self._debug_count % 50 == 0:
            print(f"[DEBUG] torque: {np.round(torque, 2)}")

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

        # 시작 시 초기 자세로 이동
        print("[초기화] 학습 초기 자세로 이동 중...")
        if self.robot.move_to_home(vel=30, acc=30):
            print("[초기화] 초기 자세 이동 완료!")
        else:
            print("[초기화] 초기 자세 이동 실패 - 수동으로 'h' 키를 눌러주세요")

        # 그리퍼 열기
        print("[초기화] 그리퍼 열기...")
        self.robot.gripper_open()
        time.sleep(1.0)

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

                        # 펜 위치/각도 유효성 표시 (로봇 좌표계 기준)
                        if HAS_WORKSPACE and cap_robot is not None:
                            y_offset = 150  # 시작 Y 위치

                            # 로봇 좌표계 펜 위치 표시
                            cv2.putText(display, f"Robot Coord: X={cap_robot[0]*100:.1f} Y={cap_robot[1]*100:.1f} Z={cap_robot[2]*100:.1f} cm",
                                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

                            # 위치 유효성 검사
                            pos_valid, pos_msg = DEFAULT_PEN_WORKSPACE.is_pen_position_valid(
                                cap_robot[0], cap_robot[1], cap_robot[2])

                            if pos_valid:
                                cv2.putText(display, "Position: OK (in training range)",
                                           (10, y_offset + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
                            else:
                                cv2.putText(display, f"Position: {pos_msg}",
                                           (10, y_offset + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)

                            # 기울기 유효성 검사
                            if dir_robot is not None:
                                tilt_rad = calculate_tilt_from_direction(dir_robot)
                                tilt_deg = np.degrees(tilt_rad)
                                tilt_valid, tilt_msg = DEFAULT_PEN_WORKSPACE.is_pen_tilt_valid(tilt_rad)

                                if tilt_valid:
                                    cv2.putText(display, f"Tilt: {tilt_deg:.1f} deg - OK (max 45 deg)",
                                               (10, y_offset + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
                                else:
                                    cv2.putText(display, f"Tilt: {tilt_deg:.1f} deg - {tilt_msg}",
                                               (10, y_offset + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)

                            # 전체 유효성 (g 키 누를 수 있는지)
                            all_valid = pos_valid and (dir_robot is None or tilt_valid)
                            if all_valid:
                                cv2.putText(display, ">>> READY TO GRASP (press 'g') <<<",
                                           (10, y_offset + 65), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                            else:
                                cv2.putText(display, ">>> MOVE PEN TO VALID RANGE <<<",
                                           (10, y_offset + 65), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

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

                        # 키 안내
                        key_help = "Keys: g=start h=home s=swap_cap_tip r=reset q=quit"
                        cv2.putText(display, key_help, (10, display.shape[0] - 35),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)

                        cv2.imshow("Sim2Real", display)

                key = cv2.waitKey(1) & 0xFF

                # 트리거 파일 확인 (UI 연동)
                trigger_start = False
                if self.config.trigger_file and os.path.exists(self.config.trigger_file):
                    trigger_start = True
                    try:
                        os.remove(self.config.trigger_file)
                    except:
                        pass

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
                elif key == ord('g') or trigger_start:
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

                        # 스플라인 카운터 초기화
                        self._spline_waypoints = []
                        self._spline_joint_pos = None
                        self._spline_step_count = 0
                        self._total_waypoints = 0

                        # 초기 관절 위치 저장 (관절 제한용)
                        self._init_joint_pos = self.robot.get_joint_positions().copy()
                        print(f"  초기 Joint1: {np.degrees(self._init_joint_pos[0]):.1f}도")
                        print(f"  스플라인: batch={self.config.spline_batch_size}, skip={self.config.spline_skip}")
                elif key == ord('s'):
                    # Cap/Tip swap
                    if self.detector is not None:
                        self.detector.swap_cap_tip()
                        print("[Swap] Cap/Tip 위치 교환됨")
                elif key == ord('r'):
                    # Tracking reset
                    if self.detector is not None:
                        self.detector.reset_tracking()
                        print("[Reset] 트래킹 리셋")

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
                            self.flush_spline_waypoints()  # 남은 웨이포인트 실행
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
                            self.flush_spline_waypoints()  # 남은 웨이포인트 실행
                            policy_running = False
                            fixed_pen_result = None

                        if step_count % 10 == 0:
                            print(f"\r  Step {step_count}: dist={current_dist*100:.1f}cm", end="")

                    # 시간 제한
                    elapsed = time.time() - start_time
                    if elapsed > self.config.max_duration:
                        print(f"\n[시간 초과] {self.config.max_duration}초")
                        self.flush_spline_waypoints()  # 남은 웨이포인트 실행
                        policy_running = False
                        fixed_pen_result = None

                # 제어 주기
                loop_elapsed = time.time() - loop_start
                if loop_elapsed < dt:
                    time.sleep(dt - loop_elapsed)

        except KeyboardInterrupt:
            print("\n\n[중단됨]")

        finally:
            # IK 모드: 남은 웨이포인트 실행
            self.flush_spline_waypoints()

            # OSC 모드: 실시간 제어 종료
            if rt_started:
                self.robot.stop_rt_control()

            self.running = False

        print("\n" + "=" * 60)
        print("종료")
        print("=" * 60)

    def shutdown(self):
        """종료"""
        self.running = False
        # 카메라 먼저 종료
        if self.detector:
            self.detector.stop()
        # OpenCV 창 종료 (카메라 종료 후)
        cv2.waitKey(1)
        cv2.destroyAllWindows()
        cv2.waitKey(1)
        # 로봇 연결 해제
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
    parser.add_argument("--duration", type=float, default=0.0,
                        help="최대 실행 시간 (초), 0=무제한")
    parser.add_argument("--spline-batch", type=int, default=50,
                        help="스플라인 웨이포인트 배치 크기 (최대 100)")
    parser.add_argument("--spline-skip", type=int, default=1,
                        help="웨이포인트 건너뛰기 (1=전부, 2=절반, 3=1/3만 저장)")
    parser.add_argument("--freq-ik", type=float, default=10.0,
                        help="IK 모드 제어 주파수 (Hz)")
    parser.add_argument("--freq-osc", type=float, default=30.0,
                        help="OSC 모드 제어 주파수 (Hz), 낮을수록 느림")
    parser.add_argument("--osc-stiffness-pos", type=float, default=200.0,
                        help="OSC 위치 강성 (200 권장, 높으면 충돌감지)")
    parser.add_argument("--osc-stiffness-rot", type=float, default=100.0,
                        help="OSC 회전 강성 (100 권장)")
    parser.add_argument("--osc-damping", type=float, default=1.0,
                        help="OSC 댐핑 비율")
    parser.add_argument("--gripper-offset", type=float, default=0.07,
                        help="그리퍼 오프셋 (미터), TCP에서 그립 포인트까지 거리")
    parser.add_argument("--no-safety", action="store_true",
                        help="안전 체크 비활성화")
    parser.add_argument("--trigger-file", type=str, default=None,
                        help="외부 트리거 파일 경로 (UI 연동용)")

    args = parser.parse_args()

    # 설정
    config = Sim2RealConfig(
        control_mode=ControlMode(args.mode),
        checkpoint_path=args.checkpoint,
        calibration_path=args.calibration,
        yolo_model_path=args.yolo_model,
        max_duration=args.duration if args.duration > 0 else float('inf'),
        spline_batch_size=min(args.spline_batch, 100),  # 최대 100
        spline_skip=max(args.spline_skip, 1),  # 최소 1
        control_freq_ik=args.freq_ik,
        control_freq_osc=args.freq_osc,
        osc_stiffness_pos=args.osc_stiffness_pos,
        osc_stiffness_rot=args.osc_stiffness_rot,
        osc_damping_ratio=args.osc_damping,
        gripper_offset_z=args.gripper_offset,
        disable_safety=args.no_safety,
        trigger_file=args.trigger_file,
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
        # 강제 종료 (OpenCV C++ 쓰레드 문제 우회)
        import os
        os._exit(0)


if __name__ == "__main__":
    main()
