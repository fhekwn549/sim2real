"""
E0509 IK V7 환경 테스트 스크립트

V6 주요 변경사항:
- 3DoF 위치 제어 (자세는 자동 정렬)
- 2단계 구조: APPROACH → GRASP

사용법:
    python play_ik_v6.py --checkpoint /path/to/model.pt
    python play_ik_v6.py --checkpoint /path/to/model.pt --num_envs 32
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from isaaclab.app import AppLauncher

# =============================================================================
# 명령행 인자
# =============================================================================
parser = argparse.ArgumentParser(description="E0509 IK V7 환경 테스트")
parser.add_argument("--num_envs", type=int, default=16, help="환경 개수")
parser.add_argument("--checkpoint", type=str, required=True, help="체크포인트 경로")
parser.add_argument("--num_steps", type=int, default=2000, help="실행할 스텝 수")
parser.add_argument("--level", type=int, default=0, choices=[0, 1, 2, 3, 4, 5, 6],
                    help="Curriculum Level (0: 수직, 1: 10°, 2: 20°, 3: 30°, 4: 45°, 5: 60°, 6: 75°)")
parser.add_argument("--smooth-alpha", type=float, default=1.0,
                    help="Action smoothing factor (0=no smooth, 1=full smooth). Default: 1.0 (OFF)")
parser.add_argument("--dead-zone", type=float, default=0.0,
                    help="Dead zone distance in cm. If dist < dead_zone, action=0. Default: 0 (OFF)")
parser.add_argument("--scale-by-dist", action="store_true",
                    help="Scale action by distance (closer = smaller action)")
parser.add_argument("--scale-min", type=float, default=0.1,
                    help="Minimum scale factor when using --scale-by-dist. Default: 0.1")
parser.add_argument("--scale-range", type=float, default=10.0,
                    help="Distance in cm at which scale=1.0. Default: 10cm")

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

# =============================================================================
# Isaac Sim 실행
# =============================================================================
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch
import torch.nn as nn
from envs.e0509_ik_env_v7 import (
    E0509IKEnvV7,
    E0509IKEnvV7Cfg_PLAY,
    E0509IKEnvV7Cfg_L0,
    E0509IKEnvV7Cfg_L1,
    E0509IKEnvV7Cfg_L2,
    E0509IKEnvV7Cfg_L3,
    CURRICULUM_TILT_MAX,
    PHASE_APPROACH,
    SUCCESS_DIST_TO_CAP,
    SUCCESS_PERP_DIST,
    SUCCESS_HOLD_STEPS,
    # V7.2 안전 상수
    SAFETY_MIN_Z_HEIGHT,
    SAFETY_MAX_DIST_FROM_PEN,
)


class ActionProcessor:
    """Action 후처리 클래스 (Sim2Real 호환)

    지원 기능:
    1. Smoothing: 이전 action과 현재 action을 혼합
    2. Dead Zone: 목표 근처에서 action을 0으로
    3. Scale by Distance: 거리에 비례해서 action 크기 조절
    """
    def __init__(
        self,
        smooth_alpha: float = 1.0,
        dead_zone_cm: float = 0.0,
        scale_by_dist: bool = False,
        scale_min: float = 0.1,
        scale_range_cm: float = 10.0,
    ):
        """
        Args:
            smooth_alpha: smoothing factor (1.0=no smooth)
            dead_zone_cm: dead zone 거리 (cm). 이 거리 이내면 action=0
            scale_by_dist: True면 거리에 비례해서 action 축소
            scale_min: scale_by_dist 시 최소 scale (0~1)
            scale_range_cm: 이 거리(cm)에서 scale=1.0
        """
        self.smooth_alpha = smooth_alpha
        self.dead_zone = dead_zone_cm / 100.0  # cm → m
        self.scale_by_dist = scale_by_dist
        self.scale_min = scale_min
        self.scale_range = scale_range_cm / 100.0  # cm → m
        self.prev_action = None

    def process(self, action: torch.Tensor, dist: torch.Tensor) -> torch.Tensor:
        """Action 후처리

        Args:
            action: raw action [num_envs, action_dim]
            dist: 목표까지 거리 [num_envs]

        Returns:
            processed action [num_envs, action_dim]
        """
        processed = action.clone()

        # 1. Dead Zone: 거리가 threshold 이내면 action=0
        if self.dead_zone > 0:
            in_dead_zone = dist < self.dead_zone  # [num_envs]
            processed[in_dead_zone] = 0.0

        # 2. Scale by Distance: 거리에 비례해서 action 축소
        if self.scale_by_dist:
            # scale = clamp(dist / scale_range, min, 1.0)
            scale = torch.clamp(dist / self.scale_range, min=self.scale_min, max=1.0)
            processed = processed * scale.unsqueeze(-1)

        # 3. Smoothing
        if self.smooth_alpha < 1.0:
            if self.prev_action is None:
                self.prev_action = processed.clone()
            else:
                processed = self.smooth_alpha * processed + (1 - self.smooth_alpha) * self.prev_action
                self.prev_action = processed.clone()

        return processed

    def reset(self):
        """Reset processor state"""
        self.prev_action = None

    def get_status_str(self) -> str:
        """현재 설정 상태 문자열 반환"""
        status = []
        if self.smooth_alpha < 1.0:
            status.append(f"Smoothing(α={self.smooth_alpha})")
        if self.dead_zone > 0:
            status.append(f"DeadZone({self.dead_zone*100:.1f}cm)")
        if self.scale_by_dist:
            status.append(f"ScaleByDist(min={self.scale_min}, range={self.scale_range*100:.0f}cm)")
        return ", ".join(status) if status else "None"


class SimpleActor(nn.Module):
    """학습된 Actor 네트워크 (추론용)"""
    def __init__(self, obs_dim, action_dim, hidden_dims=[256, 256, 128]):
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


def get_env_cfg_for_level(level: int):
    """Curriculum Level에 맞는 환경 설정 반환"""
    cfg_map = {
        0: E0509IKEnvV7Cfg_L0,
        1: E0509IKEnvV7Cfg_L1,
        2: E0509IKEnvV7Cfg_L2,
        3: E0509IKEnvV7Cfg_L3,
    }
    if level in cfg_map:
        return cfg_map[level]()
    else:
        # Level 4, 5, 6 등 테스트용: 기본 config에 curriculum_level만 설정
        from envs.e0509_ik_env_v7 import E0509IKEnvV7Cfg
        cfg = E0509IKEnvV7Cfg()
        cfg.curriculum_level = level
        return cfg


def main():
    """테스트 실행"""

    # 환경 설정
    env_cfg = get_env_cfg_for_level(args.level)
    env_cfg.scene.num_envs = args.num_envs

    env = E0509IKEnvV7(cfg=env_cfg)

    # 모델 로드
    print(f"모델 로드: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location="cuda:0")
    state_dict = checkpoint["model_state_dict"]

    obs_dim = env_cfg.observation_space
    action_dim = env_cfg.action_space

    policy = SimpleActor(obs_dim, action_dim, hidden_dims=[256, 256, 128]).to("cuda:0")

    # Actor 가중치만 추출해서 로드
    actor_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("actor."):
            actor_state_dict[k] = v

    policy.load_state_dict(actor_state_dict)
    policy.eval()

    # Action Processor 초기화
    processor = ActionProcessor(
        smooth_alpha=args.smooth_alpha,
        dead_zone_cm=args.dead_zone,
        scale_by_dist=args.scale_by_dist,
        scale_min=args.scale_min,
        scale_range_cm=args.scale_range,
    )

    tilt_deg = CURRICULUM_TILT_MAX[args.level] * 180 / 3.14159
    print("=" * 70)
    print("E0509 IK V7 테스트 시작 (3DoF 위치 + 자동 자세 정렬)")
    print("=" * 70)
    print(f"  Curriculum Level: {args.level} (펜 최대 기울기: {tilt_deg:.0f}°)")
    print(f"  [DEBUG] env_cfg.curriculum_level = {env_cfg.curriculum_level}")
    print(f"  환경 수: {args.num_envs}")
    print(f"  실행 스텝: {args.num_steps}")
    print(f"  관찰 차원: {obs_dim}")
    print(f"  액션 차원: {action_dim} (Δx, Δy, Δz)")
    print("=" * 70)
    print("V7 핵심 특징:")
    print("  - 3DoF 위치 제어 (자세는 펜 축 기반 자동 정렬)")
    print("  - APPROACH만 (GRASP 제거)")
    print(f"  - 성공 조건: dist < {SUCCESS_DIST_TO_CAP*100:.0f}cm, perp < {SUCCESS_PERP_DIST*100:.0f}cm, 캡 위")
    print(f"  - Action 후처리: {processor.get_status_str()}")
    print("V7.2 안전장치:")
    print(f"  - 최소 Z 높이: {SAFETY_MIN_Z_HEIGHT*100:.0f}cm (이하면 에피소드 종료)")
    print(f"  - 최대 펜 거리: {SAFETY_MAX_DIST_FROM_PEN*100:.0f}cm (초과하면 에피소드 종료)")
    print("=" * 70)

    # 테스트 루프
    obs_dict, _ = env.reset()
    obs = obs_dict["policy"]

    for step in range(args.num_steps):
        # 현재 거리 계산 (action 처리 전에 필요)
        grasp_pos = env._get_grasp_point()
        cap_pos = env._get_pen_cap_pos()
        dist_to_cap = torch.norm(grasp_pos - cap_pos, dim=-1)

        with torch.no_grad():
            raw_actions = policy(obs)
            actions = processor.process(raw_actions, dist_to_cap)

        obs_dict, rewards, terminated, truncated, infos = env.step(actions)
        obs = obs_dict["policy"]

        # 디버깅 출력 (100 스텝마다)
        if step % 100 == 0:
            # 단계별 통계
            phase_stats = env.get_phase_stats()

            # 메트릭 계산
            grasp_pos = env._get_grasp_point()
            cap_pos = env._get_pen_cap_pos()
            dist_to_cap = torch.norm(grasp_pos - cap_pos, dim=-1)
            perp_dist, axis_dist, on_correct_side = env._compute_axis_metrics()

            # 그리퍼 Z축과 펜 Z축 정렬
            gripper_z = env._get_gripper_z_axis()
            pen_z = env._get_pen_z_axis()
            dot = torch.sum(gripper_z * pen_z, dim=-1)

            mean_reward = rewards.mean().item()

            # 캡 위/아래 판단
            correct_side_pct = on_correct_side.float().mean().item() * 100

            # 성공 조건 체크 (V7.1: dot 조건 제거)
            success_condition = (
                (dist_to_cap < SUCCESS_DIST_TO_CAP) &
                (perp_dist < SUCCESS_PERP_DIST) &
                on_correct_side
            )
            success_pct = success_condition.float().mean().item() * 100

            print(f"\nStep {step}: reward={mean_reward:.4f}")
            print(f"  success={phase_stats['total_success']}, near_success={phase_stats.get('near_success', 0)}")
            print(f"  dist_to_cap={dist_to_cap.mean().item()*100:.2f}cm (need <{SUCCESS_DIST_TO_CAP*100:.0f}cm)")
            print(f"  perp_dist={perp_dist.mean().item()*100:.2f}cm (need <{SUCCESS_PERP_DIST*100:.0f}cm)")
            print(f"  axis_dist={axis_dist.mean().item()*100:.2f}cm (음수=캡위)")
            print(f"  캡 위에 있는 비율: {correct_side_pct:.0f}%")
            print(f"  dot(정렬)={dot.mean().item():.4f} (참고용, 자동정렬)")
            print(f"  성공 조건 충족: {success_pct:.1f}%")

    # 최종 결과
    final_stats = env.get_phase_stats()
    print("\n" + "=" * 70)
    print("테스트 완료!")
    print("=" * 70)
    print(f"Curriculum Level: {args.level} (펜 최대 기울기: {tilt_deg:.0f}°)")
    print(f"총 성공 횟수: {final_stats['total_success']}")
    print(f"캡 위에 있는 환경: {final_stats.get('on_correct_side', 0)}/{args.num_envs}")
    print("=" * 70)

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
