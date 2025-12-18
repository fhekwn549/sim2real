"""
펜 캡 접근 환경 v3 (Curriculum Learning)

=== 목표 ===
1. gripper_grasp_point → pen_cap_point 거리 최소화
2. gripper_z축 · pen_z축 → -1 (반대 방향 정렬)

=== Curriculum Learning ===
Stage 1: 거리 < 10cm, dot < -0.7  (85% 성공 시 전환)
Stage 2: 거리 < 5cm,  dot < -0.85 (90% 성공 시 전환)
Stage 3: 거리 < 2cm,  dot < -0.95 (최종 목표: 95%)

=== 설계 원칙 ===
- 엄격한 성공 조건
- Stage별 다른 보상 스케일
- 성공률 추적
"""
from __future__ import annotations

import os
import json
import torch
from typing import Dict, Tuple

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, AssetBaseCfg, RigidObject, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.actuators import ImplicitActuatorCfg

# =============================================================================
# 경로 및 상수
# =============================================================================
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROBOT_USD_PATH = os.path.join(_SCRIPT_DIR, "..", "models", "first_control.usd")
PEN_USD_PATH = os.path.join(_SCRIPT_DIR, "..", "models", "pen.usd")

PEN_LENGTH = 0.1207  # 120.7mm
PEN_MASS = 0.0163    # 16.3g

# =============================================================================
# Curriculum Learning 설정
# =============================================================================
CURRICULUM_STAGES = {
    1: {"distance_threshold": 0.10, "dot_threshold": -0.70, "success_rate_to_advance": 0.85},
    2: {"distance_threshold": 0.05, "dot_threshold": -0.85, "success_rate_to_advance": 0.90},
    3: {"distance_threshold": 0.02, "dot_threshold": -0.95, "success_rate_to_advance": 0.95},  # 최종
}

# 전역 변수로 현재 stage 관리 (환경 인스턴스 간 공유)
_current_stage = 1


def set_curriculum_stage(stage: int):
    """현재 curriculum stage 설정"""
    global _current_stage
    _current_stage = max(1, min(3, stage))
    print(f"[Curriculum] Stage set to {_current_stage}: "
          f"distance < {CURRICULUM_STAGES[_current_stage]['distance_threshold']*100:.0f}cm, "
          f"dot < {CURRICULUM_STAGES[_current_stage]['dot_threshold']:.2f}")


def get_curriculum_stage() -> int:
    """현재 curriculum stage 반환"""
    global _current_stage
    return _current_stage


def get_stage_thresholds() -> Tuple[float, float]:
    """현재 stage의 (distance_threshold, dot_threshold) 반환"""
    global _current_stage
    stage = CURRICULUM_STAGES[_current_stage]
    return stage["distance_threshold"], stage["dot_threshold"]


# =============================================================================
# Curriculum 상태 저장/로드
# =============================================================================
def save_curriculum_state(filepath: str, stage: int, success_rates: Dict[int, float], iteration: int):
    """Curriculum 상태를 JSON 파일로 저장"""
    state = {
        "current_stage": stage,
        "success_rates": success_rates,
        "last_iteration": iteration,
    }
    with open(filepath, 'w') as f:
        json.dump(state, f, indent=2)
    print(f"[Curriculum] State saved to {filepath}")


def load_curriculum_state(filepath: str) -> Dict:
    """Curriculum 상태를 JSON 파일에서 로드"""
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            state = json.load(f)
        print(f"[Curriculum] State loaded from {filepath}: Stage {state['current_stage']}")
        return state
    return {"current_stage": 1, "success_rates": {}, "last_iteration": 0}


# #############################################################################
#                              Scene Configuration
# #############################################################################
@configclass
class PenGraspSceneCfg(InteractiveSceneCfg):
    """장면 설정: 로봇 + 펜"""

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        debug_vis=False
    )

    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=500.0),
    )

    robot: ArticulationCfg = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=ROBOT_USD_PATH,
            activate_contact_sensors=False,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=5.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=0,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),
            joint_pos={
                "joint_1": 0.0,
                "joint_2": 0.3,
                "joint_3": 0.5,
                "joint_4": 0.0,
                "joint_5": 0.5,
                "joint_6": 0.0,
                "gripper_rh_r1": 0.0,
                "gripper_rh_r2": 0.0,
                "gripper_rh_l1": 0.0,
                "gripper_rh_l2": 0.0,
            },
        ),
        actuators={
            "arm": ImplicitActuatorCfg(
                joint_names_expr=["joint_[1-6]"],
                effort_limit=200.0,
                velocity_limit=3.14,
                stiffness=400.0,
                damping=40.0,
            ),
            "gripper": ImplicitActuatorCfg(
                joint_names_expr=["gripper_rh_.*"],
                effort_limit=50.0,
                velocity_limit=1.0,
                stiffness=2000.0,
                damping=100.0,
            ),
        },
    )

    pen: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Pen",
        spawn=sim_utils.UsdFileCfg(
            usd_path=PEN_USD_PATH,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                kinematic_enabled=True,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=PEN_MASS),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.5, 0.0, 0.3),
        ),
    )


# #############################################################################
#                              Action Term
# #############################################################################
class ArmActionTerm(ActionTerm):
    """팔 관절 제어 (그리퍼 열린 상태 고정)"""

    _asset: Articulation

    def __init__(self, cfg: ActionTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._raw_actions = torch.zeros(env.num_envs, 6, device=self.device)
        self._processed_actions = torch.zeros(env.num_envs, 6, device=self.device)
        self._joint_pos_target = torch.zeros(env.num_envs, 10, device=self.device)
        self.arm_scale = 0.1

    @property
    def action_dim(self) -> int:
        return 6

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions
        self._processed_actions[:] = actions

    def apply_actions(self):
        current_pos = self._asset.data.joint_pos
        arm_delta = self._processed_actions * self.arm_scale
        self._joint_pos_target[:, :6] = current_pos[:, :6] + arm_delta
        self._joint_pos_target[:, 6:10] = 0.0
        self._asset.set_joint_position_target(self._joint_pos_target)


@configclass
class ArmActionTermCfg(ActionTermCfg):
    class_type: type = ArmActionTerm


# #############################################################################
#                              Helper Functions
# #############################################################################
def get_grasp_point(robot: Articulation) -> torch.Tensor:
    """그리퍼 잡기 포인트 계산"""
    l1 = robot.data.body_pos_w[:, 7, :]
    r1 = robot.data.body_pos_w[:, 8, :]
    l2 = robot.data.body_pos_w[:, 9, :]
    r2 = robot.data.body_pos_w[:, 10, :]

    base_center = (l1 + r1) / 2.0
    tip_center = (l2 + r2) / 2.0
    finger_dir = tip_center - base_center
    finger_dir = finger_dir / (torch.norm(finger_dir, dim=-1, keepdim=True) + 1e-6)

    return base_center + finger_dir * 0.02


def get_pen_cap_pos(pen: RigidObject) -> torch.Tensor:
    """펜 캡 위치 계산 (펜 +Z 방향)"""
    pen_pos = pen.data.root_pos_w
    pen_quat = pen.data.root_quat_w

    qw, qx, qy, qz = pen_quat[:, 0], pen_quat[:, 1], pen_quat[:, 2], pen_quat[:, 3]
    cap_dir_x = 2.0 * (qx * qz + qw * qy)
    cap_dir_y = 2.0 * (qy * qz - qw * qx)
    cap_dir_z = 1.0 - 2.0 * (qx * qx + qy * qy)
    cap_dir = torch.stack([cap_dir_x, cap_dir_y, cap_dir_z], dim=-1)

    return pen_pos + (PEN_LENGTH / 2) * cap_dir


def get_pen_z_axis(pen: RigidObject) -> torch.Tensor:
    """펜 Z축 방향 벡터"""
    pen_quat = pen.data.root_quat_w
    qw, qx, qy, qz = pen_quat[:, 0], pen_quat[:, 1], pen_quat[:, 2], pen_quat[:, 3]

    z_x = 2.0 * (qx * qz + qw * qy)
    z_y = 2.0 * (qy * qz - qw * qx)
    z_z = 1.0 - 2.0 * (qx * qx + qy * qy)

    return torch.stack([z_x, z_y, z_z], dim=-1)


def get_gripper_z_axis(robot: Articulation) -> torch.Tensor:
    """그리퍼 Z축 방향 벡터 (link_6 기준)"""
    link6_quat = robot.data.body_quat_w[:, 6, :]
    qw, qx, qy, qz = link6_quat[:, 0], link6_quat[:, 1], link6_quat[:, 2], link6_quat[:, 3]

    z_x = 2.0 * (qx * qz + qw * qy)
    z_y = 2.0 * (qy * qz - qw * qx)
    z_z = 1.0 - 2.0 * (qx * qx + qy * qy)

    return torch.stack([z_x, z_y, z_z], dim=-1)


def compute_success(env: ManagerBasedRLEnv) -> torch.Tensor:
    """
    현재 stage 기준으로 성공 여부 계산

    Returns:
        (num_envs,) - True/False 성공 여부
    """
    robot: Articulation = env.scene["robot"]
    pen: RigidObject = env.scene["pen"]

    grasp_pos = get_grasp_point(robot)
    cap_pos = get_pen_cap_pos(pen)
    pen_z = get_pen_z_axis(pen)
    gripper_z = get_gripper_z_axis(robot)

    # 거리 및 정렬 계산
    distance = torch.norm(grasp_pos - cap_pos, dim=-1)
    dot_product = torch.sum(pen_z * gripper_z, dim=-1)

    # 현재 stage threshold 가져오기
    dist_thresh, dot_thresh = get_stage_thresholds()

    # 성공 조건: 거리 < threshold AND dot < threshold
    success = (distance < dist_thresh) & (dot_product < dot_thresh)

    return success


# #############################################################################
#                              Observations
# #############################################################################
def joint_pos_obs(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.joint_pos[:, :6]


def joint_vel_obs(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.joint_vel[:, :6]


def grasp_point_obs(env: ManagerBasedRLEnv) -> torch.Tensor:
    robot: Articulation = env.scene["robot"]
    return get_grasp_point(robot) - env.scene.env_origins


def pen_cap_obs(env: ManagerBasedRLEnv) -> torch.Tensor:
    pen: RigidObject = env.scene["pen"]
    return get_pen_cap_pos(pen) - env.scene.env_origins


def relative_pos_obs(env: ManagerBasedRLEnv) -> torch.Tensor:
    """그리퍼 → 펜 캡 상대 위치"""
    robot: Articulation = env.scene["robot"]
    pen: RigidObject = env.scene["pen"]

    grasp_pos = get_grasp_point(robot)
    cap_pos = get_pen_cap_pos(pen)

    return cap_pos - grasp_pos


def pen_z_axis_obs(env: ManagerBasedRLEnv) -> torch.Tensor:
    pen: RigidObject = env.scene["pen"]
    return get_pen_z_axis(pen)


def gripper_z_axis_obs(env: ManagerBasedRLEnv) -> torch.Tensor:
    robot: Articulation = env.scene["robot"]
    return get_gripper_z_axis(robot)


# #############################################################################
#                              Rewards
# #############################################################################
def position_error_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    """
    위치 오차 페널티 (L2 거리)

    더 엄격한 기준: std를 현재 stage에 맞게 조정
    """
    robot: Articulation = env.scene["robot"]
    pen: RigidObject = env.scene["pen"]

    grasp_pos = get_grasp_point(robot)
    cap_pos = get_pen_cap_pos(pen)

    distance = torch.norm(grasp_pos - cap_pos, dim=-1)
    return distance


def position_fine_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    """
    위치 정밀 보상 (tanh 커널)

    현재 stage의 threshold를 std로 사용하여 더 엄격한 보상
    """
    robot: Articulation = env.scene["robot"]
    pen: RigidObject = env.scene["pen"]

    grasp_pos = get_grasp_point(robot)
    cap_pos = get_pen_cap_pos(pen)

    distance = torch.norm(grasp_pos - cap_pos, dim=-1)

    # 현재 stage threshold를 std로 사용 (더 가까이 가야 높은 보상)
    dist_thresh, _ = get_stage_thresholds()
    std = dist_thresh  # Stage 1: 0.1, Stage 2: 0.05, Stage 3: 0.02

    return 1.0 - torch.tanh(distance / std)


def orientation_error_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    """
    방향 오차 페널티

    dot = -1 목표, 오차 = 1 + dot
    """
    robot: Articulation = env.scene["robot"]
    pen: RigidObject = env.scene["pen"]

    pen_z = get_pen_z_axis(pen)
    gripper_z = get_gripper_z_axis(robot)

    dot_product = torch.sum(pen_z * gripper_z, dim=-1)
    orientation_error = 1.0 + dot_product

    return orientation_error


def orientation_fine_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    """
    방향 정밀 보상

    dot이 threshold에 가까울수록 높은 보상
    """
    robot: Articulation = env.scene["robot"]
    pen: RigidObject = env.scene["pen"]

    pen_z = get_pen_z_axis(pen)
    gripper_z = get_gripper_z_axis(robot)

    dot_product = torch.sum(pen_z * gripper_z, dim=-1)

    # dot = -1 → 1.0, dot = 0 → 0.5, dot = +1 → 0.0
    return (-dot_product + 1.0) / 2.0


def success_bonus_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    """
    성공 시 큰 보상

    현재 stage 조건을 만족하면 큰 보상
    """
    success = compute_success(env)
    return success.float() * 10.0  # 성공 시 10.0


def action_rate_penalty(env: ManagerBasedRLEnv) -> torch.Tensor:
    """행동 크기 페널티"""
    return torch.sum(torch.square(env.action_manager.action), dim=-1)


def approach_from_above_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    """위에서 접근 보상"""
    robot: Articulation = env.scene["robot"]
    pen: RigidObject = env.scene["pen"]

    grasp_pos = get_grasp_point(robot)
    cap_pos = get_pen_cap_pos(pen)

    height_diff = grasp_pos[:, 2] - cap_pos[:, 2]
    return torch.clamp(height_diff, min=0.0, max=0.1) * 10.0


# #############################################################################
#                              Terminations
# #############################################################################
def success_termination(env: ManagerBasedRLEnv) -> torch.Tensor:
    """성공 시 에피소드 종료"""
    return compute_success(env)


# #############################################################################
#                              Configuration Classes
# #############################################################################
@configclass
class ActionsCfg:
    arm_action = ArmActionTermCfg(asset_name="robot")


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(func=joint_pos_obs, params={"asset_cfg": SceneEntityCfg("robot")})
        joint_vel = ObsTerm(func=joint_vel_obs, params={"asset_cfg": SceneEntityCfg("robot")})
        grasp_point = ObsTerm(func=grasp_point_obs)
        pen_cap = ObsTerm(func=pen_cap_obs)
        relative_pos = ObsTerm(func=relative_pos_obs)
        pen_z_axis = ObsTerm(func=pen_z_axis_obs)
        gripper_z_axis = ObsTerm(func=gripper_z_axis_obs)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class RewardsCfg:
    """
    보상 설정 (Curriculum Learning)

    === 위치 ===
    - position_error: L2 거리 페널티 (weight: -1.0)
    - position_fine: tanh 보상, stage별 std 조정 (weight: 2.0)

    === 방향 ===
    - orientation_error: dot 오차 페널티 (weight: -0.5)
    - orientation_fine: dot 보상 (weight: 1.0)

    === 성공 ===
    - success_bonus: stage 조건 달성 시 큰 보상 (weight: 5.0)

    === 정규화 ===
    - action_rate: 행동 크기 페널티 (weight: -0.001)
    - approach_bonus: 위에서 접근 (weight: 0.3)
    """
    # 위치
    position_error = RewTerm(func=position_error_reward, weight=-1.0)
    position_fine = RewTerm(func=position_fine_reward, weight=2.0)

    # 방향
    orientation_error = RewTerm(func=orientation_error_reward, weight=-0.5)
    orientation_fine = RewTerm(func=orientation_fine_reward, weight=1.0)

    # 성공 보상 (핵심!)
    success_bonus = RewTerm(func=success_bonus_reward, weight=5.0)

    # 정규화
    action_rate = RewTerm(func=action_rate_penalty, weight=-0.001)
    approach_bonus = RewTerm(func=approach_from_above_reward, weight=0.3)


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    # 성공 시 조기 종료 → 빨리 성공하고 새 위치 경험하는 게 이득
    success = DoneTerm(func=success_termination, time_out=False)


@configclass
class EventsCfg:
    reset_robot = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": (-0.1, 0.1),
            "velocity_range": (0.0, 0.0),
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    reset_pen = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (-0.2, 0.2),
                "y": (-0.2, 0.2),
                "z": (-0.1, 0.1),
            },
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("pen"),
        },
    )


# #############################################################################
#                              Environment
# #############################################################################
@configclass
class PenGraspEnvCfg(ManagerBasedRLEnvCfg):
    """펜 캡 접근 환경 설정 (Curriculum Learning)"""

    scene: PenGraspSceneCfg = PenGraspSceneCfg(num_envs=64, env_spacing=2.0)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventsCfg = EventsCfg()

    def __post_init__(self):
        self.decimation = 2
        self.sim.dt = 1.0 / 60.0
        self.episode_length_s = 10.0


class PenGraspEnv(ManagerBasedRLEnv):
    """펜 캡 접근 환경 (Curriculum Learning 지원)"""
    cfg: PenGraspEnvCfg

    def __init__(self, cfg: PenGraspEnvCfg, render_mode=None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # 성공 추적용 버퍼
        self._success_buffer = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)

    def _get_observations(self) -> dict:
        """관찰 반환 + 성공 여부 업데이트"""
        obs = super()._get_observations()

        # 현재 스텝의 성공 여부 업데이트
        self._success_buffer = compute_success(self)

        return obs

    def get_success_rate(self) -> float:
        """현재 성공률 반환 (외부에서 호출)"""
        return self._success_buffer.float().mean().item()

    def get_current_stage(self) -> int:
        """현재 curriculum stage 반환"""
        return get_curriculum_stage()

    def set_stage(self, stage: int):
        """Curriculum stage 설정"""
        set_curriculum_stage(stage)
