"""
Simple Move 환경 (Sim2Real 테스트용)

=== 목표 ===
TCP를 Z축 방향으로 5cm 위로 이동 후 Home 위치로 복귀
안전하고 간단한 동작으로 Sim2Real 테스트에 적합

=== 단계 ===
1. PHASE_UP: TCP를 5cm 위로 이동
2. PHASE_DOWN: Home 위치로 복귀

=== 사용법 ===
python train_simple_move.py --headless --num_envs 2048 --max_iterations 1000
"""
from __future__ import annotations

import os
import torch
from collections.abc import Sequence

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils import configclass
from isaaclab.actuators import ImplicitActuatorCfg


# =============================================================================
# 경로 및 상수
# =============================================================================
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROBOT_USD_PATH = os.path.join(_SCRIPT_DIR, "..", "models", "first_control.usd")

# 단계 정의
PHASE_UP = 0    # TCP 올리기
PHASE_DOWN = 1  # Home으로 복귀

# Home 자세 (도 → 라디안)
import math
HOME_JOINT_POS = [
    math.radians(0),    # joint_1
    math.radians(0),    # joint_2
    math.radians(90),   # joint_3
    math.radians(0),    # joint_4
    math.radians(90),   # joint_5
    math.radians(0),    # joint_6
]

# 목표 거리
TARGET_HEIGHT = 0.05  # 5cm
SUCCESS_THRESHOLD = 0.01  # 1cm 이내면 성공


# =============================================================================
# 환경 설정
# =============================================================================
@configclass
class SimpleMoveEnvCfg(DirectRLEnvCfg):
    """Simple Move 환경 설정"""

    # 환경 기본 설정
    decimation = 2
    episode_length_s = 10.0
    action_scale = 0.05  # 작은 움직임
    action_space = 6
    observation_space = 18  # joint_pos(6) + joint_vel(6) + tcp_pos(3) + target_pos(3)
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

    # 보상 스케일
    rew_scale_distance = -5.0       # 거리 페널티
    rew_scale_progress = 10.0       # 진행 보상
    rew_scale_success = 50.0        # 단계 성공 보상
    rew_scale_action = -0.01        # 액션 페널티


class SimpleMoveEnv(DirectRLEnv):
    """Simple Move 환경"""

    cfg: SimpleMoveEnvCfg

    def __init__(self, cfg: SimpleMoveEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # 관절 인덱스
        self._arm_joint_ids, _ = self.robot.find_joints(["joint_[1-6]"])

        # 액션 스케일
        self.action_scale = self.cfg.action_scale

        # 관절 한계
        self.robot_dof_lower_limits = self.robot.data.soft_joint_pos_limits[0, :6, 0]
        self.robot_dof_upper_limits = self.robot.data.soft_joint_pos_limits[0, :6, 1]

        # Home 관절 위치 (텐서)
        self.home_joint_pos = torch.tensor(HOME_JOINT_POS, device=self.device).unsqueeze(0)

        # 상태 머신
        self.phase = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)

        # 초기 TCP 위치 저장 (리셋 시 설정)
        self.initial_tcp_pos = torch.zeros(self.num_envs, 3, device=self.device)

        # 목표 위치
        self.target_pos = torch.zeros(self.num_envs, 3, device=self.device)

        # 이전 거리 (progress 보상용)
        self.prev_distance = torch.zeros(self.num_envs, device=self.device)

        # 성공 카운터
        self.phase_up_success = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.phase_down_success = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)

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
        """액션 적용"""
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
        full_target[:, 6:] = 0.0

        self.robot.set_joint_position_target(full_target)

    def _get_observations(self) -> dict:
        """관찰값"""
        joint_pos = self.robot.data.joint_pos[:, :6]
        joint_vel = self.robot.data.joint_vel[:, :6]
        tcp_pos = self._get_tcp_pos()
        tcp_pos_local = tcp_pos - self.scene.env_origins

        # 현재 목표 (로컬 좌표)
        target_pos_local = self.target_pos - self.scene.env_origins

        obs = torch.cat([
            joint_pos,        # 6
            joint_vel,        # 6
            tcp_pos_local,    # 3
            target_pos_local, # 3
        ], dim=-1)            # 총 18

        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        """보상 계산"""
        tcp_pos = self._get_tcp_pos()
        distance = torch.norm(tcp_pos - self.target_pos, dim=-1)

        rewards = torch.zeros(self.num_envs, device=self.device)

        # 거리 페널티
        rewards += self.cfg.rew_scale_distance * distance

        # 진행 보상
        progress = self.prev_distance - distance
        rewards += self.cfg.rew_scale_progress * torch.clamp(progress, min=0)

        # 액션 페널티
        rewards += self.cfg.rew_scale_action * torch.sum(torch.square(self.actions), dim=-1)

        # 성공 체크 및 단계 전환
        success = distance < SUCCESS_THRESHOLD

        # PHASE_UP 성공 → PHASE_DOWN으로 전환
        up_success = success & (self.phase == PHASE_UP)
        if up_success.any():
            rewards[up_success] += self.cfg.rew_scale_success
            self.phase[up_success] = PHASE_DOWN
            self.phase_up_success[up_success] += 1
            # 새 목표: Home 위치의 TCP
            self._update_target_to_home(up_success)

        # PHASE_DOWN 성공 → 에피소드 종료
        down_success = success & (self.phase == PHASE_DOWN)
        if down_success.any():
            rewards[down_success] += self.cfg.rew_scale_success
            self.phase_down_success[down_success] += 1

        self.prev_distance = distance.clone()

        return rewards

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """종료 조건"""
        tcp_pos = self._get_tcp_pos()
        distance = torch.norm(tcp_pos - self.target_pos, dim=-1)

        # PHASE_DOWN에서 성공하면 종료
        success = (distance < SUCCESS_THRESHOLD) & (self.phase == PHASE_DOWN)

        # 타임아웃
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        return success, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        """환경 리셋"""
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES

        super()._reset_idx(env_ids)

        # 로봇을 Home 자세로 리셋
        joint_pos = self.home_joint_pos.expand(len(env_ids), -1).clone()
        joint_vel = torch.zeros_like(joint_pos)

        # 약간의 노이즈 추가 (옵션)
        # joint_pos += torch.randn_like(joint_pos) * 0.01

        default_root_state = self.robot.data.default_root_state[env_ids].clone()
        default_root_state[:, :3] += self.scene.env_origins[env_ids]

        self.robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)

        # 그리퍼 포함 전체 관절
        full_joint_pos = torch.zeros(len(env_ids), 10, device=self.device)
        full_joint_pos[:, :6] = joint_pos
        full_joint_vel = torch.zeros_like(full_joint_pos)

        self.robot.write_joint_state_to_sim(full_joint_pos, full_joint_vel, None, env_ids)

        # 단계 리셋
        self.phase[env_ids] = PHASE_UP

        # 초기 TCP 위치 저장 및 목표 설정
        # 시뮬레이션 스텝 후 TCP 위치가 정확해지므로 일단 추정값 사용
        env_ids_tensor = torch.tensor(env_ids, device=self.device) if not isinstance(env_ids, torch.Tensor) else env_ids

        # 목표: 현재 TCP + Z축 5cm
        # 리셋 직후라 TCP 위치를 바로 알기 어려우므로 대략적인 값 사용
        # 실제로는 _post_reset에서 정확히 설정하는 게 좋음
        self.target_pos[env_ids_tensor] = self.scene.env_origins[env_ids] + torch.tensor([0.3, 0.0, 0.5 + TARGET_HEIGHT], device=self.device)

        self.prev_distance[env_ids_tensor] = TARGET_HEIGHT

    def _update_target_to_home(self, env_ids: torch.Tensor):
        """목표를 Home TCP 위치로 업데이트"""
        # Home 자세에서의 TCP 위치 (대략적인 값)
        # 실제로는 FK로 계산해야 함
        home_tcp_offset = torch.tensor([0.3, 0.0, 0.5], device=self.device)
        self.target_pos[env_ids] = self.scene.env_origins[env_ids] + home_tcp_offset
        self.prev_distance[env_ids] = torch.norm(
            self._get_tcp_pos()[env_ids] - self.target_pos[env_ids], dim=-1
        )

    def _get_tcp_pos(self) -> torch.Tensor:
        """TCP (Tool Center Point) 위치"""
        # link_6의 위치를 TCP로 사용
        return self.robot.data.body_pos_w[:, 6, :]

    def get_stats(self) -> dict:
        """통계 반환"""
        return {
            "phase_up_total": self.phase_up_success.sum().item(),
            "phase_down_total": self.phase_down_success.sum().item(),
            "current_phase_up": (self.phase == PHASE_UP).sum().item(),
            "current_phase_down": (self.phase == PHASE_DOWN).sum().item(),
        }


@configclass
class SimpleMoveEnvCfg_PLAY(SimpleMoveEnvCfg):
    """테스트용 설정"""
    def __post_init__(self):
        self.scene.num_envs = 50
