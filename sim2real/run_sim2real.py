#!/usr/bin/env python3
"""
Sim2Real V7 - E0509 IK 환경 (APPROACH Only)

강화학습으로 학습된 Policy를 실제 Doosan E0509 로봇에서 실행합니다.

=== V7 특징 ===
- 입력: 27차원 (관절 + 펜 위치/방향)
- 출력: 3차원 [Δx, Δy, Δz] TCP 위치 변화
- 자세: 펜 축 기반 자동 계산
- IK: Differential IK로 관절 각도 변환

=== 사용법 ===
python run_sim2real.py --checkpoint /path/to/model.pt

=== 제어 루프 ===
1. YOLO로 펜 캡 위치 + 방향 인식
2. 로봇 상태 + 펜 정보 → 27차원 Observation
3. Policy → 3차원 Action (TCP 델타)
4. 펜 축 기반 자세 계산 + IK → 관절 명령
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
from typing import Optional, Tuple, List
from dataclasses import dataclass, field

# 로컬 모듈
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pen_detector_yolo import YOLOPenDetector, YOLODetectorConfig, DetectionState
from jacobian_ik import JacobianIK
from config.pen_workspace import DEFAULT_PEN_WORKSPACE, calculate_tilt_from_direction

# ROS2 (로봇 제어용)
try:
    import rclpy
    from rclpy.node import Node
    from dsr_msgs2.srv import MoveJoint, MoveLine, GetCurrentPosx, GetCurrentPosj
    HAS_ROS2 = True
except ImportError:
    HAS_ROS2 = False
    print("[Warning] ROS2 not available")


# =============================================================================
# 설정
# =============================================================================
@dataclass
class Sim2RealConfig:
    """Sim2Real 설정"""
    # Policy
    checkpoint_path: str = ""
    action_scale: float = 0.03  # V7 환경과 동일

    # 제어
    control_freq: float = 10.0  # Hz (IK 계산 고려해서 낮춤)
    max_duration: float = 60.0  # 초

    # 캘리브레이션
    calibration_path: str = ""

    # 성공 조건 (V7과 동일)
    success_dist_to_cap: float = 0.03  # 3cm
    success_perp_dist: float = 0.01    # 1cm
    success_hold_steps: int = 30

    # 자세 제어
    auto_orientation: bool = True  # True: 펜 축 기반 자동, False: 현재 자세 유지

    # 그리퍼 오프셋 (TCP → Grasp Point)
    # 주의: 이 값은 월드 Z축으로만 적용되므로, 로봇 자세에 따라 부정확할 수 있음
    # 0으로 설정하면 TCP 기준으로 거리 계산 (더 정확)
    gripper_offset_z: float = 0.08  # 8cm = 플랜지 → grasp point

    # 안전 (V7.2)
    disable_safety: bool = False  # True면 안전 체크 비활성화 (테스트용)
    max_tcp_delta: float = 0.05  # 5cm per step
    safety_min_z: float = 0.05   # TCP Z 최소 5cm (바닥 충돌 방지) - 실제 로봇은 더 높게
    safety_max_dist: float = 0.5  # 펜에서 최대 50cm
    workspace_x: Tuple[float, float] = (-0.2, 0.8)  # X 범위 (m)
    workspace_y: Tuple[float, float] = (-0.5, 0.5)  # Y 범위 (m)
    workspace_z: Tuple[float, float] = (0.05, 0.6)  # Z 범위 (m)

    # Action 후처리 (play_v7.py와 동일)
    smooth_alpha: float = 0.7    # Action smoothing (1.0=OFF, 0.5=50% 혼합)
    dead_zone_cm: float = 2.0    # 2cm 이내면 action=0
    scale_by_dist: bool = True   # 거리에 비례해서 action 축소
    scale_min: float = 0.3       # 최소 스케일
    scale_range_cm: float = 10.0 # 10cm에서 scale=1.0

    # YOLO
    yolo_model_path: str = os.path.join(os.path.expanduser("~"), "runs/segment/train/weights/best.pt")

    # 런처 모드
    auto_exit: bool = False      # True면 목표 도달 시 자동 종료
    auto_start: bool = False     # True면 펜 감지 시 자동 시작


# =============================================================================
# Action 후처리 (play_v7.py와 동일)
# =============================================================================
class ActionProcessor:
    """Action 후처리 클래스 (Sim2Real 안전장치)

    지원 기능:
    1. Smoothing: 이전 action과 현재 action을 혼합 (급격한 변화 방지)
    2. Dead Zone: 목표 근처에서 action을 0으로 (오버슈팅 방지)
    3. Scale by Distance: 거리에 비례해서 action 크기 조절
    """
    def __init__(self, config: Sim2RealConfig):
        self.smooth_alpha = config.smooth_alpha
        self.dead_zone = config.dead_zone_cm / 100.0  # cm → m
        self.scale_by_dist = config.scale_by_dist
        self.scale_min = config.scale_min
        self.scale_range = config.scale_range_cm / 100.0  # cm → m
        self.prev_action = None

    def process(self, action: np.ndarray, dist: float) -> np.ndarray:
        """Action 후처리

        Args:
            action: raw action [3]
            dist: 목표까지 거리 (m)

        Returns:
            processed action [3]
        """
        processed = action.copy()

        # 1. Dead Zone: 거리가 threshold 이내면 action=0
        if self.dead_zone > 0 and dist < self.dead_zone:
            processed = np.zeros_like(action)
            return processed

        # 2. Scale by Distance: 거리에 비례해서 action 축소
        if self.scale_by_dist:
            scale = np.clip(dist / self.scale_range, self.scale_min, 1.0)
            processed = processed * scale

        # 3. Smoothing
        if self.smooth_alpha < 1.0:
            if self.prev_action is None:
                self.prev_action = processed.copy()
            else:
                processed = self.smooth_alpha * processed + (1 - self.smooth_alpha) * self.prev_action
                self.prev_action = processed.copy()

        return processed

    def reset(self):
        """Reset processor state"""
        self.prev_action = None

    def get_status_str(self) -> str:
        """현재 설정 상태 문자열 반환"""
        status = []
        if self.smooth_alpha < 1.0:
            status.append(f"Smooth(α={self.smooth_alpha})")
        if self.dead_zone > 0:
            status.append(f"DeadZone({self.dead_zone*100:.1f}cm)")
        if self.scale_by_dist:
            status.append(f"ScaleDist(min={self.scale_min})")
        return ", ".join(status) if status else "None"


# =============================================================================
# Policy 네트워크
# =============================================================================
class PolicyNetwork(nn.Module):
    """V7 Policy 네트워크 (27 → 3)"""

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
# 로봇 인터페이스 (ROS2)
# =============================================================================
class RobotInterface:
    """Doosan E0509 로봇 인터페이스 (ROS2)"""

    def __init__(self, namespace: str = "dsr01"):
        self.namespace = namespace
        self.node = None
        self.initialized = False

        # 서비스 클라이언트
        self.cli_move_joint = None
        self.cli_move_line = None
        self.cli_get_posx = None
        self.cli_get_posj = None

        # 관절 한계 (라디안)
        self.joint_limits_lower = np.deg2rad([-360, -95, -135, -360, -135, -360])
        self.joint_limits_upper = np.deg2rad([360, 95, 135, 360, 135, 360])

        # Home 위치 (라디안) - V7 시뮬레이션 초기 자세와 동일하게!
        # 시뮬레이션: [0, -0.3, 0.8, 0, 1.57, 0] rad = [0, -17.2, 45.8, 0, 90, 0] deg
        self.home_joint = np.array([0.0, -0.3, 0.8, 0.0, 1.57, 0.0])  # 라디안

    def connect(self) -> bool:
        """ROS2 연결"""
        if not HAS_ROS2:
            print("[RobotInterface] ROS2 not available")
            return False

        try:
            if not rclpy.ok():
                rclpy.init()

            self.node = rclpy.create_node('sim2real_v7')

            # 서비스 클라이언트 생성
            self.cli_move_joint = self.node.create_client(
                MoveJoint, f'/{self.namespace}/motion/move_joint')
            self.cli_move_line = self.node.create_client(
                MoveLine, f'/{self.namespace}/motion/move_line')
            self.cli_get_posx = self.node.create_client(
                GetCurrentPosx, f'/{self.namespace}/aux_control/get_current_posx')
            self.cli_get_posj = self.node.create_client(
                GetCurrentPosj, f'/{self.namespace}/aux_control/get_current_posj')

            # 서비스 대기
            print("[RobotInterface] 서비스 연결 대기...")
            services = [
                (self.cli_move_joint, 'move_joint'),
                (self.cli_get_posx, 'get_current_posx'),
                (self.cli_get_posj, 'get_current_posj'),
            ]

            for cli, name in services:
                if not cli.wait_for_service(timeout_sec=5.0):
                    print(f"[Warning] {name} 서비스 연결 실패")
                    return False

            self.initialized = True
            print("[RobotInterface] 연결 완료")
            return True

        except Exception as e:
            print(f"[RobotInterface] 연결 실패: {e}")
            return False

    def get_joint_positions(self) -> np.ndarray:
        """현재 관절 위치 (라디안)"""
        if not self.initialized:
            return np.zeros(6)

        req = GetCurrentPosj.Request()
        future = self.cli_get_posj.call_async(req)
        rclpy.spin_until_future_complete(self.node, future, timeout_sec=1.0)

        if future.done() and future.result():
            # 도 → 라디안
            return np.deg2rad(np.array(future.result().pos[:6]))
        return np.zeros(6)

    def get_joint_velocities(self) -> np.ndarray:
        """현재 관절 속도 (라디안/초) - 추정"""
        # 실제로는 이전 값과 비교해서 계산해야 함
        return np.zeros(6)

    def get_tcp_pose(self) -> Tuple[np.ndarray, np.ndarray]:
        """현재 TCP 위치/자세 (미터, ZYZ 오일러)"""
        if not self.initialized:
            return np.zeros(3), np.zeros(3)

        req = GetCurrentPosx.Request()
        req.ref = 0  # DR_BASE

        future = self.cli_get_posx.call_async(req)
        rclpy.spin_until_future_complete(self.node, future, timeout_sec=1.0)

        if future.done() and future.result() and future.result().success:
            data = future.result().task_pos_info[0].data
            pos = np.array(data[:3]) / 1000.0  # mm → m
            rot = np.deg2rad(np.array(data[3:6]))  # deg → rad
            return pos, rot

        return np.zeros(3), np.zeros(3)

    def move_joint(self, joint_pos: np.ndarray, vel: float = 30.0, acc: float = 30.0, wait: bool = True):
        """관절 이동"""
        if not self.initialized:
            return False

        # 관절 한계 클램핑
        joint_pos = np.clip(joint_pos, self.joint_limits_lower, self.joint_limits_upper)

        req = MoveJoint.Request()
        req.pos = np.rad2deg(joint_pos).tolist()  # 라디안 → 도
        req.vel = float(vel)
        req.acc = float(acc)
        req.time = 0.0
        req.radius = 0.0
        req.mode = 0  # DR_MV_MOD_ABS
        req.blend_type = 0
        req.sync_type = 0 if wait else 1

        future = self.cli_move_joint.call_async(req)

        if wait:
            rclpy.spin_until_future_complete(self.node, future, timeout_sec=30.0)
            return future.done() and future.result() and future.result().success
        return True

    def move_line(self, pos: np.ndarray, rot: np.ndarray, vel: float = 50.0, acc: float = 50.0, wait: bool = True):
        """직선 이동 (TCP 좌표)"""
        if not self.initialized:
            return False

        req = MoveLine.Request()
        req.pos = [
            float(pos[0] * 1000),  # m → mm
            float(pos[1] * 1000),
            float(pos[2] * 1000),
            float(np.rad2deg(rot[0])),  # rad → deg
            float(np.rad2deg(rot[1])),
            float(np.rad2deg(rot[2])),
        ]
        req.vel = [float(vel), float(vel)]
        req.acc = [float(acc), float(acc)]
        req.time = 0.0
        req.radius = 0.0
        req.ref = 0  # DR_BASE
        req.mode = 0  # DR_MV_MOD_ABS
        req.blend_type = 0
        req.sync_type = 0 if wait else 1

        future = self.cli_move_line.call_async(req)

        if wait:
            rclpy.spin_until_future_complete(self.node, future, timeout_sec=30.0)
            return future.done() and future.result() and future.result().success
        return True

    def move_to_home(self):
        """Home 위치로 이동"""
        print("[Robot] Home 위치로 이동...")
        return self.move_joint(self.home_joint, vel=30, acc=30, wait=True)

    def shutdown(self):
        """종료"""
        if self.node:
            self.node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


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
        """캘리브레이션 로드"""
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
        """카메라 좌표 → 로봇 좌표"""
        if self.R_axes is None:
            return point_cam
        return self.R_axes @ point_cam + self.t_offset

    def direction_to_robot(self, dir_cam: np.ndarray) -> np.ndarray:
        """카메라 방향 → 로봇 방향"""
        if self.R_axes is None:
            return dir_cam
        dir_robot = self.R_axes @ dir_cam
        norm = np.linalg.norm(dir_robot)
        if norm > 0:
            dir_robot /= norm
        return dir_robot


# =============================================================================
# Sim2Real V7 컨트롤러
# =============================================================================
class Sim2RealV7:
    """V7 Sim2Real 컨트롤러"""

    def __init__(self, config: Sim2RealConfig):
        self.config = config
        self.running = False

        print("=" * 60)
        print("Sim2Real V7 초기화")
        print("=" * 60)

        # 1. Policy 로드
        print("\n[1/4] Policy 로드...")
        self.policy = self._load_policy(config.checkpoint_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy.to(self.device)
        self.policy.eval()
        print(f"  → Device: {self.device}")

        # 2. 로봇 연결
        print("\n[2/4] 로봇 연결...")
        self.robot = RobotInterface()
        if not self.robot.connect():
            raise RuntimeError("로봇 연결 실패")

        # 3. 좌표 변환기
        print("\n[3/4] 좌표 변환기 로드...")
        calib_path = config.calibration_path or os.path.join(
            os.path.dirname(__file__), "config", "calibration_eye_to_hand.npz"
        )
        self.transformer = CoordinateTransformer(calib_path)

        # 4. YOLO 펜 감지기
        print("\n[4/4] YOLO 펜 감지기 시작...")
        self.detector = YOLOPenDetector(config.yolo_model_path)
        if not self.detector.start():
            raise RuntimeError("카메라 시작 실패")

        # 5. Action 후처리기 (안전장치)
        self.action_processor = ActionProcessor(config)

        # 6. Jacobian IK (시뮬레이션과 동일한 방식)
        print("\n[5/5] Jacobian IK 초기화...")
        self.ik = JacobianIK(lambda_val=0.05)  # V7과 동일한 damping

        # 상태 변수
        self.success_hold_count = 0
        self.prev_joint_pos = None

        print("\n" + "=" * 60)
        print("초기화 완료!")
        print(f"  Action 후처리: {self.action_processor.get_status_str()}")
        print(f"  IK 방식: {self.ik.method} (Differential IK)")
        print(f"  자세 제어: {'자동 (펜 축 기반)' if config.auto_orientation else '현재 자세 유지'}")
        if config.disable_safety:
            print(f"  ⚠️  안전 체크: 비활성화 (테스트 모드)")
        else:
            print(f"  안전 TCP Z 최소: {config.safety_min_z*100:.0f}cm")
            print(f"  작업 영역: X={config.workspace_x}, Y={config.workspace_y}, Z={config.workspace_z}")
        print("=" * 60)

    def _load_policy(self, checkpoint_path: str) -> PolicyNetwork:
        """Policy 로드"""
        policy = PolicyNetwork()

        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        state_dict = checkpoint['model_state_dict']

        # actor 부분만 추출
        actor_dict = {k: v for k, v in state_dict.items() if k.startswith('actor') or k == 'std'}
        policy.load_state_dict(actor_dict, strict=False)

        print(f"  → Checkpoint: {checkpoint_path}")
        print(f"  → Iteration: {checkpoint.get('iter', 'unknown')}")

        return policy

    def build_observation(
        self,
        joint_pos: np.ndarray,
        joint_vel: np.ndarray,
        grasp_pos: np.ndarray,
        cap_pos: np.ndarray,
        pen_z: np.ndarray,
    ) -> np.ndarray:
        """
        27차원 Observation 구성

        Args:
            joint_pos: 관절 위치 (6)
            joint_vel: 관절 속도 (6)
            grasp_pos: 그리퍼 위치 (3) - 로봇 좌표계
            cap_pos: 펜 캡 위치 (3) - 로봇 좌표계
            pen_z: 펜 축 방향 (3) - 로봇 좌표계

        Returns:
            observation (27)
        """
        # 상대 위치
        rel_pos = cap_pos - grasp_pos

        # 펜 축까지 거리 계산
        grasp_to_cap = cap_pos - grasp_pos
        axis_distance = np.dot(grasp_to_cap, pen_z)
        projection = axis_distance * pen_z
        perp_vec = grasp_to_cap - projection
        perpendicular_dist = np.linalg.norm(perp_vec)

        # 캡까지 거리
        distance_to_cap = np.linalg.norm(rel_pos)

        # phase (항상 0 - APPROACH)
        phase = 0.0

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

    def compute_auto_orientation(self, pen_z: np.ndarray) -> np.ndarray:
        """
        펜 축 기반 자동 자세 계산 (V7 환경과 동일)

        그리퍼 Z축이 -pen_z 방향을 향하도록 계산

        Returns:
            rotation (3): ZYZ 오일러 각도 (라디안)
        """
        # 그리퍼 Z축 = -펜 Z축
        gripper_z = -pen_z

        # 그리퍼 X축: 펜 축과 월드 Z축의 외적
        world_z = np.array([0.0, 0.0, 1.0])
        gripper_x = np.cross(world_z, gripper_z)
        gripper_x_norm = np.linalg.norm(gripper_x)

        if gripper_x_norm < 0.1:
            # 펜이 거의 수직일 때
            gripper_x = np.array([1.0, 0.0, 0.0])
        else:
            gripper_x = gripper_x / gripper_x_norm

        # 그리퍼 Y축
        gripper_y = np.cross(gripper_z, gripper_x)
        gripper_y = gripper_y / (np.linalg.norm(gripper_y) + 1e-6)

        # 회전 행렬
        R = np.column_stack([gripper_x, gripper_y, gripper_z])

        # 회전 행렬 → ZYZ 오일러 (Doosan 로봇 규격)
        # 간단히 ZYX로 변환 후 사용
        sy = np.sqrt(R[0, 0]**2 + R[1, 0]**2)
        singular = sy < 1e-6

        if not singular:
            x = np.arctan2(R[2, 1], R[2, 2])
            y = np.arctan2(-R[2, 0], sy)
            z = np.arctan2(R[1, 0], R[0, 0])
        else:
            x = np.arctan2(-R[1, 2], R[1, 1])
            y = np.arctan2(-R[2, 0], sy)
            z = 0

        return np.array([z, y, x])  # ZYX → Doosan 순서

    def _check_safety(self, tcp_pos: np.ndarray, grasp_pos: np.ndarray, cap_pos: np.ndarray) -> Tuple[bool, str]:
        """
        안전 체크 (V7.2 안전장치)

        Args:
            tcp_pos: TCP 위치 (로봇 플랜지)
            grasp_pos: Grasp point 위치 (그리퍼 끝점)
            cap_pos: 펜 캡 위치

        Returns:
            (is_safe, message)
        """
        # 1. TCP Z 높이 체크 (로봇 충돌 방지)
        if tcp_pos[2] < self.config.safety_min_z:
            return False, f"TCP Z 높이 위험 ({tcp_pos[2]*100:.1f}cm < {self.config.safety_min_z*100:.0f}cm)"

        # 2. Grasp point에서 펜까지 거리 체크
        dist_to_cap = np.linalg.norm(grasp_pos - cap_pos)
        if dist_to_cap > self.config.safety_max_dist:
            return False, f"펜에서 너무 멀어짐 ({dist_to_cap*100:.0f}cm > {self.config.safety_max_dist*100:.0f}cm)"

        # 3. TCP 작업 영역 체크
        if not (self.config.workspace_x[0] <= tcp_pos[0] <= self.config.workspace_x[1]):
            return False, f"X 범위 초과 ({tcp_pos[0]*100:.0f}cm)"
        if not (self.config.workspace_y[0] <= tcp_pos[1] <= self.config.workspace_y[1]):
            return False, f"Y 범위 초과 ({tcp_pos[1]*100:.0f}cm)"
        if not (self.config.workspace_z[0] <= tcp_pos[2] <= self.config.workspace_z[1]):
            return False, f"Z 범위 초과 ({tcp_pos[2]*100:.0f}cm)"

        return True, "OK"

    def step(self, pen_result: dict) -> dict:
        """
        한 스텝 실행

        Args:
            pen_result: 펜 감지 결과 (cap_pos, pen_z 등)

        Returns:
            info: 상태 정보
        """
        # 1. 로봇 상태
        joint_pos = self.robot.get_joint_positions()
        joint_vel = self.robot.get_joint_velocities()
        tcp_pos, tcp_rot = self.robot.get_tcp_pose()

        # 1.5. 로봇 상태 조회 실패 체크 (통신 에러 시 [0,0,0] 반환됨)
        if np.linalg.norm(tcp_pos) < 0.01:
            print("  [Warning] 로봇 상태 조회 실패, 스텝 건너뜀")
            return {
                'action': np.zeros(3),
                'tcp_pos': tcp_pos,
                'tcp_rot': tcp_rot,
                'cap_pos': pen_result['cap_robot'],
                'pen_z': pen_result['direction_robot'],
                'distance_to_cap': 0,
                'perpendicular_dist': 0,
                'on_correct_side': False,
                'success_hold': 0,
                'success': False,
                'skip': True,
            }

        # 2. Grasp point = TCP + 그리퍼 오프셋 (Z축 방향으로만)
        # 그리퍼가 아래를 향한다고 가정, Z축으로 offset 적용
        # offset > 0: grasp가 TCP보다 아래 (그리퍼 끝 방향)
        # offset < 0: grasp가 TCP보다 위
        grasp_pos = tcp_pos.copy()
        grasp_pos[2] -= self.config.gripper_offset_z  # 양수 offset = 아래로

        # 3. 펜 정보 (로봇 좌표계)
        cap_pos = pen_result['cap_robot']
        pen_z = pen_result['direction_robot']

        if pen_z is None:
            # 방향 없으면 수직 가정
            pen_z = np.array([0.0, 0.0, 1.0])

        # 4. Observation 구성 (grasp_pos 사용)
        obs = self.build_observation(joint_pos, joint_vel, grasp_pos, cap_pos, pen_z)

        # 5. Policy 추론
        with torch.no_grad():
            obs_tensor = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)
            raw_action = self.policy(obs_tensor).cpu().numpy()[0]
            # Action 클램핑 (시뮬레이션에서 학습된 범위로 제한)
            raw_action = np.clip(raw_action, -1.0, 1.0)

        # 6. 현재 거리 계산 (Action 후처리용) - grasp_pos 기준
        distance_to_cap = np.linalg.norm(cap_pos - grasp_pos)

        # === DEBUG 출력 ===
        if hasattr(self, '_debug_count'):
            self._debug_count += 1
        else:
            self._debug_count = 0

        if self._debug_count % 5 == 0:  # 0.5초마다 출력 (10Hz 기준)
            print("\n" + "="*50)
            print("[DEBUG] Observation 구성:")
            print(f"  joint_pos (rad): [{', '.join([f'{x:.3f}' for x in joint_pos])}]")
            print(f"  joint_vel: [{', '.join([f'{x:.3f}' for x in joint_vel])}]")
            print(f"  grasp_pos (m): [{grasp_pos[0]:.3f}, {grasp_pos[1]:.3f}, {grasp_pos[2]:.3f}]")
            print(f"  cap_pos (m): [{cap_pos[0]:.3f}, {cap_pos[1]:.3f}, {cap_pos[2]:.3f}]")
            rel_pos = cap_pos - grasp_pos
            print(f"  rel_pos (m): [{rel_pos[0]:.3f}, {rel_pos[1]:.3f}, {rel_pos[2]:.3f}]")
            print(f"  pen_z: [{pen_z[0]:.3f}, {pen_z[1]:.3f}, {pen_z[2]:.3f}]")
            print(f"  distance_to_cap: {distance_to_cap:.3f}m ({distance_to_cap*100:.1f}cm)")
            print(f"[DEBUG] Policy 출력:")
            print(f"  raw_action (clamped): [{raw_action[0]:.4f}, {raw_action[1]:.4f}, {raw_action[2]:.4f}]")
            print(f"  방향 확인: rel_pos=[{'+' if (cap_pos-grasp_pos)[0]>0 else '-'}, {'+' if (cap_pos-grasp_pos)[1]>0 else '-'}, {'+' if (cap_pos-grasp_pos)[2]>0 else '-'}] → action=[{'+' if raw_action[0]>0 else '-'}, {'+' if raw_action[1]>0 else '-'}, {'+' if raw_action[2]>0 else '-'}]")
            print("="*50)

        # 6. Action 후처리 (Smoothing, Dead Zone, Scale by Distance)
        action = self.action_processor.process(raw_action, distance_to_cap)

        # 6.5. 성공 조건 근처에서 자동 정지
        # 거리가 5cm 이내면 정지 (진동 방지)
        approach_stop_dist = 0.02  # 2cm

        if distance_to_cap < approach_stop_dist:
            # 성공 조건 근처 - 이동 멈춤
            action = np.zeros(3)
            if self._debug_count % 5 == 0:
                print(f"  [자동 정지] 펜 캡 근처! dist={distance_to_cap*100:.1f}cm")

        # 7. Action 스케일링
        tcp_delta = action * self.config.action_scale

        # 8. 안전 클램핑
        tcp_delta = np.clip(tcp_delta, -self.config.max_tcp_delta, self.config.max_tcp_delta)

        # === DEBUG 출력 (계속) ===
        if self._debug_count % 5 == 0:
            print(f"  processed_action: [{action[0]:.4f}, {action[1]:.4f}, {action[2]:.4f}]")
            print(f"  tcp_delta (m): [{tcp_delta[0]:.4f}, {tcp_delta[1]:.4f}, {tcp_delta[2]:.4f}]")
            print(f"  tcp_delta (mm): [{tcp_delta[0]*1000:.1f}, {tcp_delta[1]*1000:.1f}, {tcp_delta[2]*1000:.1f}]")

        # 9. 새 TCP 위치 (예측값 - 안전 체크용)
        new_tcp_pos_est = tcp_pos + tcp_delta
        new_grasp_pos_est = new_tcp_pos_est.copy()
        new_grasp_pos_est[2] -= self.config.gripper_offset_z

        # 10. 작업 영역 안전 체크 (비활성화 가능)
        if self.config.disable_safety:
            safety_ok, safety_msg = True, "Safety disabled"
        else:
            safety_ok, safety_msg = self._check_safety(new_tcp_pos_est, new_grasp_pos_est, cap_pos)
        if not safety_ok:
            print(f"\n[안전] {safety_msg} - 이동 취소")
            return {
                'action': action,
                'tcp_pos': tcp_pos,  # 이동하지 않음
                'tcp_rot': tcp_rot,
                'cap_pos': cap_pos,
                'pen_z': pen_z,
                'distance_to_cap': distance_to_cap,
                'perpendicular_dist': 0,
                'on_correct_side': False,
                'success_hold': 0,
                'success': False,
                'safety_stop': True,
                'safety_msg': safety_msg,
            }

        # 11. Differential IK로 관절 변화량 계산 (시뮬레이션과 동일!)
        delta_q = self.ik.compute(joint_pos, tcp_delta)

        # 12. 안전 제한: delta_q 최대 1.5도 (부드러운 움직임)
        max_delta_rad = np.deg2rad(1.5)
        delta_q = np.clip(delta_q, -max_delta_rad, max_delta_rad)

        # 13. 새 관절 위치
        new_joint_pos = joint_pos + delta_q

        # 관절 한계 적용
        new_joint_pos = np.clip(
            new_joint_pos,
            self.robot.joint_limits_lower,
            self.robot.joint_limits_upper
        )

        # === DEBUG: IK 결과 출력 ===
        if self._debug_count % 5 == 0:
            print(f"  [IK] delta_q (deg): [{', '.join([f'{np.degrees(x):.2f}' for x in delta_q])}]")
            # FK 검증
            fk_pos, _ = self.ik.forward_kinematics(new_joint_pos)
            print(f"  [FK] 예상 TCP: [{fk_pos[0]*1000:.1f}, {fk_pos[1]*1000:.1f}, {fk_pos[2]*1000:.1f}] mm")

        # 15. 로봇 이동 (MoveJoint) - 낮은 속도로 안전하게
        self.robot.move_joint(new_joint_pos, vel=20, acc=20, wait=False)

        # 16. FK로 새 TCP 위치 계산 (실제 예상 위치)
        new_tcp_pos_fk, _ = self.ik.forward_kinematics(new_joint_pos)
        new_grasp_pos = new_tcp_pos_fk.copy()
        new_grasp_pos[2] -= self.config.gripper_offset_z

        # 17. 성공 조건 체크 (grasp_pos 기준)
        rel_pos = cap_pos - new_grasp_pos
        distance_to_cap = np.linalg.norm(rel_pos)

        axis_dist = np.dot(rel_pos, pen_z)
        perp_vec = rel_pos - axis_dist * pen_z
        perpendicular_dist = np.linalg.norm(perp_vec)
        on_correct_side = axis_dist > 0  # 캡 위에 있음

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
            'tcp_pos': new_tcp_pos_fk,  # FK로 계산한 예상 위치
            'tcp_rot': tcp_rot,  # 현재 자세 (IK는 위치만 제어)
            'cap_pos': cap_pos,
            'pen_z': pen_z,
            'distance_to_cap': distance_to_cap,
            'perpendicular_dist': perpendicular_dist,
            'on_correct_side': on_correct_side,
            'success_hold': self.success_hold_count,
            'success': success,
        }

    def run(self) -> bool:
        """메인 제어 루프 - 'g' 키로 시작

        Returns:
            bool: 목표 도달 성공 여부 (auto_exit 모드에서 사용)
        """
        print("\n" + "=" * 60)
        print("Sim2Real V7 대기 모드")
        print("=" * 60)
        if self.config.auto_start:
            print("  [AUTO MODE] 펜 감지 시 자동 시작")
        if self.config.auto_exit:
            print("  [AUTO MODE] 목표 도달 시 자동 종료")
        print("조작:")
        print("  g: Policy 실행 시작")
        print("  h: Home 위치로 이동")
        print("  q: 종료")
        print("=" * 60)

        success_result = False  # 최종 성공 여부

        dt = 1.0 / self.config.control_freq
        self.running = True
        policy_running = False
        start_time = None
        step_count = 0
        fixed_pen_result = None  # 'g' 누를 때 펜 위치/방향 고정
        reached_target = False   # 한 번 목표 근처 도달하면 True (진동 방지)
        min_distance = float('inf')  # 최소 거리 기록 (진동 감지용)

        try:
            while self.running:
                loop_start = time.time()

                # 1. 펜 감지
                result = self.detector.detect()

                # 2. 시각화용 프레임 가져오기
                color_image, depth_image = self.detector.get_last_frames()
                if color_image is None:
                    time.sleep(0.01)
                    continue

                # 3. 시각화
                display = self.detector.visualize(color_image.copy(), result)

                # 펜 정보 준비
                pen_result = None
                cap_robot = None
                dir_robot = None
                dir_cam = None

                if result is not None and result.state == DetectionState.DETECTED:
                    cap_cam = result.grasp_point
                    if cap_cam is not None and np.linalg.norm(cap_cam) > 0.01:
                        cap_robot = self.transformer.cam_to_robot(cap_cam)

                        # 펜 방향 (필터링된 direction_3d 사용)
                        # 주의: 실제 감지는 cap→tip 방향, 시뮬레이션은 tip→cap 방향
                        # 시뮬레이션과 맞추기 위해 방향 반전 (-1)
                        if result.direction_3d is not None:
                            dir_cam = result.direction_3d.copy()
                            dir_robot = -self.transformer.direction_to_robot(dir_cam)  # 방향 반전!
                        else:
                            dir_cam = np.array([0.0, 0.0, 1.0])
                            dir_robot = np.array([0.0, 0.0, 1.0])

                        pen_result = {
                            'cap_robot': cap_robot,
                            'direction_robot': dir_robot,
                        }

                # 로봇 좌표 표시 (고정된 값 또는 실시간 값)
                x_offset = display.shape[1] - 280

                # Policy 실행 중이면 고정된 값 표시, 아니면 실시간 값 표시
                if policy_running and fixed_pen_result is not None:
                    display_cap = fixed_pen_result['cap_robot']
                    display_dir = fixed_pen_result['direction_robot']
                    label_suffix = " (FIXED)"
                    label_color = (0, 255, 0)  # 녹색 = 고정
                else:
                    display_cap = cap_robot
                    display_dir = dir_robot
                    label_suffix = " (LIVE)"
                    label_color = (0, 255, 255)  # 노란색 = 실시간

                if display_cap is not None:
                    cap_mm = display_cap * 1000

                    cv2.putText(display, f"=== TARGET{label_suffix} ===",
                               (x_offset, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, label_color, 2)
                    cv2.putText(display, f"X: {cap_mm[0]:+7.1f} mm",
                               (x_offset, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, label_color, 1)
                    cv2.putText(display, f"Y: {cap_mm[1]:+7.1f} mm",
                               (x_offset, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, label_color, 1)
                    cv2.putText(display, f"Z: {cap_mm[2]:+7.1f} mm",
                               (x_offset, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.5, label_color, 1)

                    # 펜 방향 표시
                    if display_dir is not None:
                        cv2.putText(display, f"=== DIRECTION{label_suffix} ===",
                                   (x_offset, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.5, label_color, 2)
                        cv2.putText(display, f"dX: {display_dir[0]:+.3f}",
                                   (x_offset, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, label_color, 1)
                        cv2.putText(display, f"dY: {display_dir[1]:+.3f}",
                                   (x_offset, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.5, label_color, 1)
                        cv2.putText(display, f"dZ: {display_dir[2]:+.3f}",
                                   (x_offset, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.5, label_color, 1)

                        # 방향 해석
                        if abs(display_dir[2]) > 0.7:
                            orient = "VERTICAL" if display_dir[2] > 0 else "VERTICAL (down)"
                        elif abs(display_dir[0]) > abs(display_dir[1]):
                            orient = "HORIZONTAL (X)"
                        else:
                            orient = "HORIZONTAL (Y)"
                        cv2.putText(display, f"Orient: {orient}",
                                   (x_offset, 215), cv2.FONT_HERSHEY_SIMPLEX, 0.45, label_color, 1)

                # 상태 표시
                if policy_running:
                    elapsed = time.time() - start_time
                    status = f"RUNNING Step:{step_count} [{elapsed:.1f}s]"
                    status_color = (0, 255, 0)
                else:
                    status = "READY - Press 'g' to start"
                    status_color = (0, 200, 200)

                cv2.putText(display, status, (10, display.shape[0] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)

                # 화면 확대
                scale = 1.5
                new_w = int(display.shape[1] * scale)
                new_h = int(display.shape[0] * scale)
                display = cv2.resize(display, (new_w, new_h))

                cv2.imshow("Sim2Real V7", display)
                key = cv2.waitKey(30) & 0xFF

                # 키 입력 처리
                if key == ord('q'):
                    break
                elif key == ord('h'):
                    print("\n[Home] Home 위치로 이동...")
                    policy_running = False
                    fixed_pen_result = None
                    self.action_processor.reset()
                    self.robot.move_to_home()
                    print("[Home] 완료")
                elif key == ord('g') or (self.config.auto_start and not policy_running and pen_result is not None and fixed_pen_result is None):
                    if not policy_running and pen_result is not None:
                        # 펜 위치/각도 유효성 검사 (학습 범위와 동일)
                        cap_pos = pen_result['cap_robot']
                        pen_dir = pen_result['direction_robot'] if pen_result['direction_robot'] is not None else np.array([0.0, 0.0, 1.0])

                        # 위치 검사
                        pos_valid, pos_msg = DEFAULT_PEN_WORKSPACE.is_pen_position_valid(
                            cap_pos[0], cap_pos[1], cap_pos[2]
                        )

                        # 기울기 검사
                        tilt_rad = calculate_tilt_from_direction(pen_dir)
                        tilt_valid, tilt_msg = DEFAULT_PEN_WORKSPACE.is_pen_tilt_valid(tilt_rad)

                        if not pos_valid:
                            print(f"\n[Error] 펜 위치가 학습 범위를 벗어남!")
                            print(f"  {pos_msg}")
                            print(f"  현재: X={cap_pos[0]*100:.1f}cm, Y={cap_pos[1]*100:.1f}cm, Z={cap_pos[2]*100:.1f}cm")
                            DEFAULT_PEN_WORKSPACE.print_config()
                            continue

                        if not tilt_valid:
                            print(f"\n[Error] 펜 기울기가 학습 범위를 벗어남!")
                            print(f"  {tilt_msg}")
                            print(f"  현재 기울기: {np.degrees(tilt_rad):.1f}°")
                            continue

                        # 스크린샷 자동 저장
                        timestamp = time.strftime("%Y%m%d_%H%M%S")
                        screenshot_dir = os.path.join(os.path.expanduser("~"), "Pictures/스크린샷")
                        os.makedirs(screenshot_dir, exist_ok=True)
                        screenshot_path = os.path.join(screenshot_dir, f"sim2real_start_{timestamp}.png")
                        cv2.imwrite(screenshot_path, display)
                        print(f"\n[Screenshot] 저장: {screenshot_path}")

                        # 현재 펜 위치/방향 저장 (고정)
                        fixed_pen_result = {
                            'cap_robot': pen_result['cap_robot'].copy(),
                            'direction_robot': pen_result['direction_robot'].copy() if pen_result['direction_robot'] is not None else np.array([0.0, 0.0, 1.0]),
                        }
                        print("[Start] Policy 실행 시작!")
                        print(f"  펜 위치 고정: [{fixed_pen_result['cap_robot'][0]*1000:.1f}, {fixed_pen_result['cap_robot'][1]*1000:.1f}, {fixed_pen_result['cap_robot'][2]*1000:.1f}] mm")
                        print(f"  펜 방향 고정: [{fixed_pen_result['direction_robot'][0]:.3f}, {fixed_pen_result['direction_robot'][1]:.3f}, {fixed_pen_result['direction_robot'][2]:.3f}]")
                        print(f"  펜 기울기: {np.degrees(tilt_rad):.1f}° (유효 범위 내)")
                        policy_running = True
                        reached_target = False  # 리셋
                        min_distance = float('inf')  # 리셋
                        start_time = time.time()
                        step_count = 0
                        self.success_hold_count = 0
                        self.action_processor.reset()
                    elif not policy_running and pen_result is None:
                        print("\n[Error] 펜이 감지되지 않음! 펜을 카메라에 보이게 하세요.")

                # Policy 실행 중 (고정된 펜 위치 사용)
                if policy_running and fixed_pen_result is not None:
                    info = None  # 초기화

                    # 이미 목표 도달했으면 대기 (진동 방지)
                    if reached_target:
                        if step_count % 30 == 0:  # 3초마다 출력
                            print(f"\r  [도달 완료] 목표 근처에서 대기 중... (Step {step_count})", end="")
                        step_count += 1
                    else:
                        # 스텝 실행 (고정된 펜 위치 사용!)
                        info = self.step(fixed_pen_result)
                        step_count += 1

                        # 현재 거리
                        current_dist = info.get('distance_to_cap', 1.0)

                        # 최소 거리 업데이트
                        if current_dist < min_distance:
                            min_distance = current_dist

                        # 목표 근처 도달 체크 (2cm 이내 → 영구 정지)
                        if current_dist < 0.02:  # 2cm
                            reached_target = True
                            success_result = True
                            print(f"\n  [목표 도달!] dist={current_dist*100:.1f}cm - 정지")
                            if self.config.auto_exit:
                                print("  [AUTO EXIT] 목표 도달 - 자동 종료")
                                self.running = False

                        # 진동 감지: 최소 거리에서 3cm 이상 멀어지면 정지 (더 민감하게)
                        elif min_distance < 0.10 and current_dist > min_distance + 0.03:
                            reached_target = True
                            success_result = True  # 진동 감지도 목표 근처 도달로 간주
                            print(f"\n  [진동 감지!] min={min_distance*100:.1f}cm → now={current_dist*100:.1f}cm - 정지")
                            if self.config.auto_exit:
                                print("  [AUTO EXIT] 목표 근처 도달 - 자동 종료")
                                self.running = False

                        # 로그
                        if step_count % 10 == 0:
                            print(f"\r  Step {step_count}: "
                                  f"dist={info['distance_to_cap']*100:.1f}cm, "
                                  f"perp={info['perpendicular_dist']*100:.1f}cm, "
                                  f"hold={info['success_hold']}", end="")

                    # 성공 체크 (info가 있을 때만)
                    if info is not None and info.get('success', False):
                        print(f"\n\n[성공!] {self.config.success_hold_steps} 스텝 유지")
                        policy_running = False
                        fixed_pen_result = None

                    # 안전 정지 (info가 있을 때만)
                    if info is not None and info.get('safety_stop', False):
                        print(f"\n[안전 정지] {info.get('safety_msg', '')}")
                        policy_running = False
                        fixed_pen_result = None

                    # 시간 제한
                    elapsed = time.time() - start_time
                    if elapsed > self.config.max_duration:
                        print(f"\n[시간 초과] {self.config.max_duration}초")
                        policy_running = False
                        fixed_pen_result = None
                        reached_target = False

                # 제어 주기 유지
                loop_elapsed = time.time() - loop_start
                if loop_elapsed < dt:
                    time.sleep(dt - loop_elapsed)

        except KeyboardInterrupt:
            print("\n\n[중단됨]")

        self.running = False
        cv2.destroyAllWindows()

        print("\n" + "=" * 60)
        print(f"종료 (성공: {success_result})")
        print("=" * 60)

        return success_result

    def shutdown(self):
        """종료"""
        self.running = False
        self.detector.stop()
        self.robot.shutdown()
        print("Sim2Real V7 종료")


# =============================================================================
# 메인
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="Sim2Real V7")
    parser.add_argument("--checkpoint", type=str,
                       default=os.path.join(os.path.expanduser("~"), "ikv7/model_99999.pt"),
                       help="학습된 모델 경로 (.pt)")
    parser.add_argument("--calibration", type=str, default=None,
                       help="캘리브레이션 파일 경로")
    parser.add_argument("--yolo_model", type=str,
                       default=os.path.join(os.path.expanduser("~"), "runs/segment/train/weights/best.pt"),
                       help="YOLO 모델 경로")
    parser.add_argument("--duration", type=float, default=60.0,
                       help="최대 실행 시간 (초)")
    parser.add_argument("--freq", type=float, default=10.0,
                       help="제어 주파수 (Hz)")
    parser.add_argument("--auto-orient", action="store_true",
                       help="펜 축 기반 자동 자세 계산 (기본: 현재 자세 유지)")
    parser.add_argument("--gripper-offset", type=float, default=0.15,
                       help="그리퍼 오프셋 (TCP→Grasp, 미터, 기본: 0.15)")
    parser.add_argument("--no-safety", action="store_true",
                       help="안전 체크 비활성화 (테스트용, 주의!)")
    parser.add_argument("--safety-z", type=float, default=0.05,
                       help="안전 Z 최소 높이 (미터, 기본: 0.05)")
    parser.add_argument("--auto-exit", action="store_true",
                       help="목표 도달 시 자동 종료 (런처 모드)")
    parser.add_argument("--auto-start", action="store_true",
                       help="펜 감지 시 자동 시작 (런처 모드)")

    args = parser.parse_args()

    # 설정
    config = Sim2RealConfig(
        checkpoint_path=args.checkpoint,
        calibration_path=args.calibration,
        yolo_model_path=args.yolo_model,
        max_duration=args.duration,
        control_freq=args.freq,
        auto_orientation=args.auto_orient,
        gripper_offset_z=args.gripper_offset,
        disable_safety=args.no_safety,
        safety_min_z=args.safety_z,
        auto_exit=args.auto_exit,
        auto_start=args.auto_start,
    )

    # 컨트롤러 생성
    controller = Sim2RealV7(config)

    # 시그널 핸들러
    def signal_handler(sig, frame):
        print("\n\n[Signal] 종료 신호 수신")
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
