"""
펜 캡 접근 환경 v2 (Reach 예제 기반)

=== 목표 ===
1. gripper_grasp_point → pen_cap_point 거리 최소화
2. gripper_z축 · pen_z축 → -1 (반대 방향 정렬)

=== 설계 원칙 ===
- Isaac Lab reach 예제 구조 기반
- 보상 함수 최소화 (위치 + 방향만)
- 검증된 보상 형태 사용 (L2, tanh)
"""
from __future__ import annotations

import os
import torch

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


# #############################################################################
#                              Scene Configuration
# #############################################################################
@configclass
class PenGraspSceneCfg(InteractiveSceneCfg):
    """장면 설정: 로봇 + 펜"""

    # 바닥
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        debug_vis=False
    )

    # 조명
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=500.0),
    )

    # 로봇
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
                # 중간 자세: 펜에 가깝지만 측면 접근은 어렵게
                # 완전 직립보다 약간 구부린 상태
                "joint_1": 0.0,
                "joint_2": 0.3,    # 약간 앞으로
                "joint_3": 0.5,    # 약간 구부림
                "joint_4": 0.0,
                "joint_5": 0.5,    # 약간 아래로
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

    # 펜 (공중 고정)
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
        self._joint_pos_target[:, 6:10] = 0.0  # 그리퍼 열림
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


# #############################################################################
#                              Observations
# #############################################################################
def joint_pos_obs(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """관절 위치"""
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.joint_pos[:, :6]  # 팔 관절만


def joint_vel_obs(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """관절 속도"""
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.joint_vel[:, :6]


def grasp_point_obs(env: ManagerBasedRLEnv) -> torch.Tensor:
    """그리퍼 잡기 포인트 위치"""
    robot: Articulation = env.scene["robot"]
    return get_grasp_point(robot) - env.scene.env_origins


def pen_cap_obs(env: ManagerBasedRLEnv) -> torch.Tensor:
    """펜 캡 위치"""
    pen: RigidObject = env.scene["pen"]
    return get_pen_cap_pos(pen) - env.scene.env_origins


def relative_pos_obs(env: ManagerBasedRLEnv) -> torch.Tensor:
    """그리퍼 → 펜 캡 상대 위치 (핵심 관찰!)"""
    robot: Articulation = env.scene["robot"]
    pen: RigidObject = env.scene["pen"]

    grasp_pos = get_grasp_point(robot)
    cap_pos = get_pen_cap_pos(pen)

    return cap_pos - grasp_pos


def pen_z_axis_obs(env: ManagerBasedRLEnv) -> torch.Tensor:
    """펜 Z축 방향"""
    pen: RigidObject = env.scene["pen"]
    return get_pen_z_axis(pen)


def gripper_z_axis_obs(env: ManagerBasedRLEnv) -> torch.Tensor:
    """그리퍼 Z축 방향"""
    robot: Articulation = env.scene["robot"]
    return get_gripper_z_axis(robot)


# #############################################################################
#                              Rewards (핵심: 2개만!)
# #############################################################################
def position_error_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    """
    목표 1: 위치 오차 (L2 거리)

    reach 예제의 position_command_error와 동일한 형태
    """
    robot: Articulation = env.scene["robot"]
    pen: RigidObject = env.scene["pen"]

    grasp_pos = get_grasp_point(robot)
    cap_pos = get_pen_cap_pos(pen)

    distance = torch.norm(grasp_pos - cap_pos, dim=-1)
    return distance  # 음수 weight로 페널티


def position_error_tanh_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    """
    목표 1: 위치 오차 (tanh 커널, 정밀 추적)

    reach 예제의 position_command_error_tanh와 동일한 형태
    """
    robot: Articulation = env.scene["robot"]
    pen: RigidObject = env.scene["pen"]

    grasp_pos = get_grasp_point(robot)
    cap_pos = get_pen_cap_pos(pen)

    distance = torch.norm(grasp_pos - cap_pos, dim=-1)
    std = 0.1  # 10cm 기준
    return 1.0 - torch.tanh(distance / std)


def orientation_error_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    """
    목표 2: 방향 오차 (dot product → -1 목표)

    pen_z · gripper_z = -1 이면 완벽한 정렬
    오차 = 1 + dot (0이면 완벽, 2면 최악)
    """
    robot: Articulation = env.scene["robot"]
    pen: RigidObject = env.scene["pen"]

    pen_z = get_pen_z_axis(pen)
    gripper_z = get_gripper_z_axis(robot)

    dot_product = torch.sum(pen_z * gripper_z, dim=-1)

    # dot = -1 → error = 0, dot = +1 → error = 2
    orientation_error = 1.0 + dot_product

    return orientation_error  # 음수 weight로 페널티


def action_rate_penalty(env: ManagerBasedRLEnv) -> torch.Tensor:
    """행동 변화율 페널티"""
    return torch.sum(torch.square(env.action_manager.action), dim=-1)


def approach_from_above_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    """
    위에서 접근 보상

    그리퍼가 펜 캡보다 위에 있을 때 보상.
    측면 접근 대신 위에서 내려오는 동작을 유도.
    """
    robot: Articulation = env.scene["robot"]
    pen: RigidObject = env.scene["pen"]

    grasp_pos = get_grasp_point(robot)
    cap_pos = get_pen_cap_pos(pen)

    # 그리퍼 Z > 캡 Z 이면 보상
    height_diff = grasp_pos[:, 2] - cap_pos[:, 2]
    return torch.clamp(height_diff, min=0.0, max=0.1) * 10.0  # 최대 1.0


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
        # 로봇 상태 (12)
        joint_pos = ObsTerm(func=joint_pos_obs, params={"asset_cfg": SceneEntityCfg("robot")})
        joint_vel = ObsTerm(func=joint_vel_obs, params={"asset_cfg": SceneEntityCfg("robot")})

        # 위치 정보 (9)
        grasp_point = ObsTerm(func=grasp_point_obs)
        pen_cap = ObsTerm(func=pen_cap_obs)
        relative_pos = ObsTerm(func=relative_pos_obs)  # 가장 중요!

        # 방향 정보 (6)
        pen_z_axis = ObsTerm(func=pen_z_axis_obs)
        gripper_z_axis = ObsTerm(func=gripper_z_axis_obs)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class RewardsCfg:
    """
    보상 설정 (reach 예제 스타일)

    === 목표 1: 위치 ===
    - position_error: L2 거리 (페널티)
    - position_fine: tanh 커널 (보상)

    === 목표 2: 방향 ===
    - orientation_error: dot product 오차 (페널티)

    === 목표 3: 위에서 접근 ===
    - approach_bonus: 그리퍼가 캡보다 위에 있으면 보상

    === 정규화 ===
    - action_rate: 행동 크기 페널티
    """
    # 위치 추적
    position_error = RewTerm(func=position_error_reward, weight=-0.5)
    position_fine = RewTerm(func=position_error_tanh_reward, weight=1.0)

    # 방향 추적
    orientation_error = RewTerm(func=orientation_error_reward, weight=-0.3)

    # 위에서 접근 유도
    approach_bonus = RewTerm(func=approach_from_above_reward, weight=0.3)

    # 정규화
    action_rate = RewTerm(func=action_rate_penalty, weight=-0.001)


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)


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
    """펜 캡 접근 환경 설정"""

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
    cfg: PenGraspEnvCfg
