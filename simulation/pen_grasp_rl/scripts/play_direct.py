"""
E0509 Direct 환경 테스트/시연 스크립트

학습된 정책을 테스트하거나 시연합니다.

사용법:
    # 학습된 모델 테스트
    python play_direct.py --checkpoint /path/to/model.pt

    # 환경 수 조절
    python play_direct.py --checkpoint /path/to/model.pt --num_envs 16
"""
import argparse
import os
import sys

# 프로젝트 경로 추가
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from isaaclab.app import AppLauncher

# =============================================================================
# 명령행 인자
# =============================================================================
parser = argparse.ArgumentParser(description="E0509 Direct 테스트")
parser.add_argument("--checkpoint", type=str, required=True, help="모델 체크포인트 경로")
parser.add_argument("--num_envs", type=int, default=50, help="환경 개수")
parser.add_argument("--num_steps", type=int, default=1000, help="실행할 스텝 수")

# AppLauncher 인자 (GUI 모드가 기본)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

# =============================================================================
# Isaac Sim 실행
# =============================================================================
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# Isaac Sim 실행 후 import
import torch
import torch.nn as nn
from envs.e0509_direct_env import E0509DirectEnv, E0509DirectEnvCfg_PLAY


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


def main():
    """메인 테스트 함수"""

    # =============================================================================
    # 환경 설정 (테스트용)
    # =============================================================================
    env_cfg = E0509DirectEnvCfg_PLAY()
    env_cfg.scene.num_envs = args.num_envs

    # 환경 생성
    env = E0509DirectEnv(cfg=env_cfg)

    # =============================================================================
    # 모델 로드 (직접 로드)
    # =============================================================================
    print(f"모델 로드: {args.checkpoint}")

    # 체크포인트 로드
    checkpoint = torch.load(args.checkpoint, map_location="cuda:0")
    state_dict = checkpoint["model_state_dict"]

    # Actor 네트워크 생성 (학습 시 사용한 구조와 동일)
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

    print("=" * 60)
    print("E0509 Direct 테스트 시작")
    print("=" * 60)
    print(f"  환경 수: {args.num_envs}")
    print(f"  실행 스텝: {args.num_steps}")
    print(f"  관찰 차원: {env_cfg.observation_space}")
    print(f"  액션 차원: {env_cfg.action_space}")
    print("=" * 60)

    # =============================================================================
    # 테스트 실행
    # =============================================================================
    obs_dict, _ = env.reset()
    obs = obs_dict["policy"]

    total_success = 0

    for step in range(args.num_steps):
        with torch.no_grad():
            # policy는 함수로 반환됨
            actions = policy(obs)

        obs_dict, rewards, terminated, truncated, infos = env.step(actions)
        obs = obs_dict["policy"]

        # 성공 카운트
        total_success += terminated.sum().item()

        if step % 100 == 0:
            stats = env.get_phase_stats()
            mean_reward = rewards.mean().item()
            print(f"Step {step}: reward={mean_reward:.4f}, "
                  f"phases=[PRE:{stats['pre_grasp']}, DESC:{stats['descend']}], "
                  f"success={stats['total_success']}")

    # =============================================================================
    # 최종 통계
    # =============================================================================
    final_stats = env.get_phase_stats()
    print("\n" + "=" * 60)
    print("테스트 완료!")
    print("=" * 60)
    print(f"총 성공 횟수: {final_stats['total_success']}")
    print(f"최종 단계 분포: PRE_GRASP={final_stats['pre_grasp']}, "
          f"DESCEND={final_stats['descend']}")
    print("=" * 60)

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
