"""
Target Tracking 환경 (Sim2Real Visual Servoing용)

=== 목표 ===
Gripper grasp point를 랜덤한 target 위치로 이동
RealSense로 펜 캡 인식 → target으로 설정하면 펜 추적 가능

=== 특징 ===
- z축 정렬 없음 (순수 위치 추적)
- 랜덤 target으로 일반화 학습
- Sim2Real에 최적화된 간단한 구조

=== 사용법 ===
python train_target_tracking.py --headless --num_envs 2048 --max_iterations 1000
"""
from __future__ import annotations

import os
import math
import torch
from collections.abc import Sequence

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils import configclass
from isaaclab.utils.math import sample_uniform
from isaaclab.actuators import ImplicitActuatorCfg


# =============================================================================
# 경로 및 상수
# =============================================================================
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROBOT_USD_PATH = os.path.join(_SCRIPT_DIR, "..", "models", "first_control.usd")

# Home 자세 (도 → 라디안)
HOME_JOINT_POS = [
    math.radians(0),    # joint_1
    math.radians(0),    # joint_2
    math.radians(90),   # joint_3
    math.radians(0),    # joint_4
    math.radians(90),   # joint_5
    math.radians(0),    # joint_6
]

# 성공 조건
SUCCESS_THRESHOLD = 0.02  # 2cm 이내면 성공


# =============================================================================
# 환경 설정
# =============================================================================
@configclass
class TargetTrackingEnvCfg(DirectRLEnvCfg):
    """Target Tracking 환경 설정"""

    # 환경 기본 설정
    decimation = 2
    episode_length_s = 10.0
    action_scale = 0.05
    action_space = 6
    observation_space = 18  # joint_pos(6) + joint_vel(6) + grasp_pos(3) + target_pos(3)
    state_space = 0

    # 시뮬레이션
    sim: SimulationCfg = SimulationCfg(
        dt=1.0 / 60.0,
        render_interval=2,
    )

    # 씬
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=2048,
        env_spacing=2.5,
        replicate_physics=True,
    )

    # 로봇 설정
    robot_cfg: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
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
                "joint_1": HOME_JOINT_POS[0],
                "joint_2": HOME_JOINT_POS[1],
                "joint_3": HOME_JOINT_POS[2],
                "joint_4": HOME_JOINT_POS[3],
                "joint_5": HOME_JOINT_POS[4],
                "joint_6": HOME_JOINT_POS[5],
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

    # Target 위치 범위 (로봇 base 기준, 미터)
    # 로봇 작업 영역 내에서 안전한 범위
    target_pos_range = {
        "x": (0.25, 0.45),   # 앞쪽
        "y": (-0.15, 0.15),  # 좌우
        "z": (0.25, 0.45),   # 높이
    }

    # 보상 스케일
    rew_scale_distance = -5.0       # 거리 페널티
    rew_scale_progress = 10.0       # 진행 보상
    rew_scale_success = 100.0       # 성공 보상
    rew_scale_action = -0.01        # 액션 페널티


class TargetTrackingEnv(DirectRLEnv):
    """Target Tracking 환경"""

    cfg: TargetTrackingEnvCfg

    def __init__(self, cfg: TargetTrackingEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # 관절 인덱스
        self._arm_joint_ids, _ = self.robot.find_joints(["joint_[1-6]"])

        # 액션 스케일
        self.action_scale = self.cfg.action_scale

        # 관절 한계
        self.robot_dof_lower_limits = self.robot.data.soft_joint_pos_limits[0, :6, 0]
        self.robot_dof_upper_limits = self.robot.data.soft_joint_pos_limits[0, :6, 1]

        # Home 관절 위치
        self.home_joint_pos = torch.tensor(HOME_JOINT_POS, device=self.device).unsqueeze(0)

        # 목표 위치
        self.target_pos = torch.zeros(self.num_envs, 3, device=self.device)

        # 이전 거리 (progress 보상용)
        self.prev_distance = torch.zeros(self.num_envs, device=self.device)

        # 성공 카운터
        self.success_count = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)

    def _setup_scene(self):
        """씬 구성"""
        self.robot = Articulation(self.cfg.robot_cfg)
        self.scene.articulations["robot"] = self.robot

        # 바닥
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())

        # 조명
        light_cfg = sim_utils.DomeLightCfg(intensity=2500.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

        # 환경 복제
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[])

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        """액션 전처리"""
        self.actions = actions.clone()

    def _apply_action(self) -> None:
        """액션 적용 (delta position control)"""
        current_pos = self.robot.data.joint_pos[:, :6]
        target_pos = current_pos + self.actions * self.action_scale

        # 관절 한계 클램핑
        target_pos = torch.clamp(
            target_pos,
            self.robot_dof_lower_limits,
            self.robot_dof_upper_limits,
        )

        # 전체 관절 타겟
        full_target = torch.zeros(self.num_envs, 10, device=self.device)
        full_target[:, :6] = target_pos
        full_target[:, 6:] = 0.0  # 그리퍼 열림

        self.robot.set_joint_position_target(full_target)

    def _get_observations(self) -> dict:
        """관찰값"""
        joint_pos = self.robot.data.joint_pos[:, :6]
        joint_vel = self.robot.data.joint_vel[:, :6]

        # Grasp point (그리퍼 손가락 중심)
        grasp_pos = self._get_grasp_point()
        grasp_pos_local = grasp_pos - self.scene.env_origins

        # Target 위치 (로컬 좌표)
        target_pos_local = self.target_pos - self.scene.env_origins

        obs = torch.cat([
            joint_pos,        # 6
            joint_vel,        # 6
            grasp_pos_local,  # 3
            target_pos_local, # 3
        ], dim=-1)            # 총 18

        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        """보상 계산"""
        grasp_pos = self._get_grasp_point()
        distance = torch.norm(grasp_pos - self.target_pos, dim=-1)

        rewards = torch.zeros(self.num_envs, device=self.device)

        # 거리 페널티
        rewards += self.cfg.rew_scale_distance * distance

        # 진행 보상
        progress = self.prev_distance - distance
        rewards += self.cfg.rew_scale_progress * torch.clamp(progress, min=0)

        # 액션 페널티
        rewards += self.cfg.rew_scale_action * torch.sum(torch.square(self.actions), dim=-1)

        # 성공 보상
        success = distance < SUCCESS_THRESHOLD
        rewards[success] += self.cfg.rew_scale_success
        self.success_count[success] += 1

        self.prev_distance = distance.clone()

        return rewards

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """종료 조건"""
        grasp_pos = self._get_grasp_point()
        distance = torch.norm(grasp_pos - self.target_pos, dim=-1)

        # 성공하면 종료
        success = distance < SUCCESS_THRESHOLD

        # 타임아웃
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        return success, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        """환경 리셋"""
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES

        super()._reset_idx(env_ids)

        env_ids_tensor = torch.tensor(env_ids, device=self.device) if not isinstance(env_ids, torch.Tensor) else env_ids
        num_resets = len(env_ids)

        # 로봇을 Home 자세로 리셋 (약간의 노이즈)
        joint_pos = self.home_joint_pos.expand(num_resets, -1).clone()
        joint_pos += torch.randn(num_resets, 6, device=self.device) * 0.05  # 작은 노이즈
        joint_vel = torch.zeros_like(joint_pos)

        default_root_state = self.robot.data.default_root_state[env_ids].clone()
        default_root_state[:, :3] += self.scene.env_origins[env_ids]

        self.robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)

        # 전체 관절
        full_joint_pos = torch.zeros(num_resets, 10, device=self.device)
        full_joint_pos[:, :6] = joint_pos
        full_joint_vel = torch.zeros_like(full_joint_pos)

        self.robot.write_joint_state_to_sim(full_joint_pos, full_joint_vel, None, env_ids)

        # 랜덤 target 위치 설정
        target_x = sample_uniform(
            self.cfg.target_pos_range["x"][0],
            self.cfg.target_pos_range["x"][1],
            (num_resets,),
            device=self.device,
        )
        target_y = sample_uniform(
            self.cfg.target_pos_range["y"][0],
            self.cfg.target_pos_range["y"][1],
            (num_resets,),
            device=self.device,
        )
        target_z = sample_uniform(
            self.cfg.target_pos_range["z"][0],
            self.cfg.target_pos_range["z"][1],
            (num_resets,),
            device=self.device,
        )

        self.target_pos[env_ids_tensor, 0] = self.scene.env_origins[env_ids, 0] + target_x
        self.target_pos[env_ids_tensor, 1] = self.scene.env_origins[env_ids, 1] + target_y
        self.target_pos[env_ids_tensor, 2] = self.scene.env_origins[env_ids, 2] + target_z

        # 초기 거리
        grasp_pos = self._get_grasp_point()
        self.prev_distance[env_ids_tensor] = torch.norm(
            grasp_pos[env_ids_tensor] - self.target_pos[env_ids_tensor], dim=-1
        )

    def _get_grasp_point(self) -> torch.Tensor:
        """그리퍼 잡기 포인트 (손가락 끝에서 앞으로 offset)"""
        # 그리퍼 손가락 위치
        l1 = self.robot.data.body_pos_w[:, 7, :]   # gripper_rh_l1
        r1 = self.robot.data.body_pos_w[:, 8, :]   # gripper_rh_r1
        l2 = self.robot.data.body_pos_w[:, 9, :]   # gripper_rh_l2
        r2 = self.robot.data.body_pos_w[:, 10, :]  # gripper_rh_r2

        # 중심점 계산
        base_center = (l1 + r1) / 2.0
        tip_center = (l2 + r2) / 2.0

        # 손가락 방향 (base → tip)
        finger_dir = tip_center - base_center
        finger_dir = finger_dir / (torch.norm(finger_dir, dim=-1, keepdim=True) + 1e-6)

        # 손가락 끝에서 10cm 앞으로 offset (RealSense 인식 가능 범위)
        grasp_point = tip_center + finger_dir * 0.10

        return grasp_point

    def get_stats(self) -> dict:
        """통계 반환"""
        return {
            "total_success": self.success_count.sum().item(),
        }


@configclass
class TargetTrackingEnvCfg_PLAY(TargetTrackingEnvCfg):
    """테스트용 설정"""
    def __post_init__(self):
        self.scene.num_envs = 50
