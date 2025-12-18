"""
E0509 Reach 환경 (Isaac Lab reach 예제 기반)

=== 목표 ===
엔드이펙터(link_6)를 목표 위치로 이동

=== 설계 원칙 ===
- Isaac Lab reach 예제 구조 그대로 사용
- 검증된 보상 함수 사용
- orientation 추적 제거 (위치만)
- 그리퍼는 열린 상태 고정

=== 사용법 ===
python train_v4.py --headless --num_envs 4096
"""
from __future__ import annotations

import math
import os
import torch
from typing import TYPE_CHECKING

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObject
from isaaclab.utils.math import combine_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from isaaclab.actuators import ImplicitActuatorCfg

# =============================================================================
# 경로 설정
# =============================================================================
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROBOT_USD_PATH = os.path.join(_SCRIPT_DIR, "..", "models", "first_control.usd")


# =============================================================================
# 보상 함수 (reach 예제에서 가져옴)
# =============================================================================
def position_command_error(env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """
    L2 거리 기반 위치 오차 페널티

    목표 위치와 현재 엔드이펙터 위치 사이의 L2 거리를 계산합니다.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    # 목표 위치 (body frame → world frame)
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(asset.data.root_pos_w, asset.data.root_quat_w, des_pos_b)
    # 현재 엔드이펙터 위치
    curr_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids[0]]
    return torch.norm(curr_pos_w - des_pos_w, dim=1)


def position_command_error_tanh(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """
    tanh 커널 기반 위치 보상

    거리가 가까울수록 1에 가까운 값을 반환합니다.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    # 목표 위치 (body frame → world frame)
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(asset.data.root_pos_w, asset.data.root_quat_w, des_pos_b)
    # 현재 엔드이펙터 위치
    curr_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids[0]]
    distance = torch.norm(curr_pos_w - des_pos_w, dim=1)
    return 1 - torch.tanh(distance / std)


# #############################################################################
#                              Scene Configuration
# #############################################################################
@configclass
class E0509ReachSceneCfg(InteractiveSceneCfg):
    """E0509 로봇 reach 장면 설정"""

    # 바닥
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        debug_vis=False
    )

    # 조명
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=2500.0),
    )

    # E0509 로봇 + RH-P12-RN-A 그리퍼
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
                "joint_2": 0.0,
                "joint_3": 0.0,
                "joint_4": 0.0,
                "joint_5": 0.0,
                "joint_6": 0.0,
                # 그리퍼는 열린 상태로 고정
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


# #############################################################################
#                              MDP Settings
# #############################################################################
@configclass
class CommandsCfg:
    """목표 위치 명령 설정"""

    ee_pose = mdp.UniformPoseCommandCfg(
        asset_name="robot",
        body_name="link_6",  # E0509 엔드이펙터
        resampling_time_range=(4.0, 4.0),  # 4초마다 새 목표
        debug_vis=True,  # 목표 위치 시각화
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            # E0509 도달 가능 범위 (로봇 앞쪽)
            pos_x=(0.3, 0.5),
            pos_y=(-0.2, 0.2),
            pos_z=(0.2, 0.4),
            # orientation은 사용하지 않음 (고정)
            roll=(0.0, 0.0),
            pitch=(math.pi / 2, math.pi / 2),  # 아래 방향
            yaw=(0.0, 0.0),
        ),
    )


@configclass
class ActionsCfg:
    """액션 설정: 팔 관절만 제어 (그리퍼 고정)"""

    arm_action = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["joint_[1-6]"],
        scale=0.5,
        use_default_offset=True,
    )


@configclass
class ObservationsCfg:
    """관찰 설정"""

    @configclass
    class PolicyCfg(ObsGroup):
        """정책 관찰"""

        # 관절 위치 (상대값)
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        # 관절 속도 (상대값)
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        # 목표 위치 명령
        pose_command = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "ee_pose"},
        )
        # 이전 액션
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class RewardsCfg:
    """
    보상 설정 (위치만, orientation 제거)

    === 위치 추적 ===
    - end_effector_position_tracking: L2 거리 페널티
    - end_effector_position_tracking_fine_grained: tanh 보상

    === 정규화 ===
    - action_rate: 액션 변화율 페널티
    - joint_vel: 관절 속도 페널티
    """

    # 위치 추적 (L2 거리)
    end_effector_position_tracking = RewTerm(
        func=position_command_error,
        weight=-0.2,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["link_6"]),
            "command_name": "ee_pose",
        },
    )

    # 위치 추적 (tanh - 가까울수록 높은 보상)
    end_effector_position_tracking_fine_grained = RewTerm(
        func=position_command_error_tanh,
        weight=0.1,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["link_6"]),
            "std": 0.1,
            "command_name": "ee_pose",
        },
    )

    # 액션 변화율 페널티
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.0001)

    # 관절 속도 페널티
    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-0.0001,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )


@configclass
class TerminationsCfg:
    """종료 조건"""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)


@configclass
class EventsCfg:
    """이벤트 설정 (리셋 등)"""

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (0.5, 1.5),
            "velocity_range": (0.0, 0.0),
        },
    )


@configclass
class CurriculumCfg:
    """커리큘럼 설정: 점진적으로 페널티 증가"""

    action_rate = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "action_rate", "weight": -0.005, "num_steps": 4500},
    )

    joint_vel = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "joint_vel", "weight": -0.001, "num_steps": 4500},
    )


# #############################################################################
#                              Environment Configuration
# #############################################################################
@configclass
class E0509ReachEnvCfg(ManagerBasedRLEnvCfg):
    """E0509 Reach 환경 설정"""

    # 장면
    scene: E0509ReachSceneCfg = E0509ReachSceneCfg(num_envs=4096, env_spacing=2.5)

    # MDP
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventsCfg = EventsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """기본 설정"""
        # 시뮬레이션
        self.decimation = 2
        self.sim.render_interval = self.decimation
        self.episode_length_s = 12.0
        self.sim.dt = 1.0 / 60.0

        # 뷰어 (GUI 모드)
        self.viewer.eye = (2.0, 2.0, 2.0)
        self.viewer.lookat = (0.5, 0.0, 0.3)


@configclass
class E0509ReachEnvCfg_PLAY(E0509ReachEnvCfg):
    """테스트/시연용 설정"""

    def __post_init__(self):
        super().__post_init__()
        # 작은 환경 수
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # 노이즈 비활성화
        self.observations.policy.enable_corruption = False
