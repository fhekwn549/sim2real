"""
E0509 Reach 학습 스크립트

Isaac Lab reach 예제 기반의 단순화된 학습 스크립트입니다.

사용법:
    # 기본 실행 (headless 모드)
    python train_reach.py --headless --num_envs 4096

    # GUI 모드로 실행
    python train_reach.py --num_envs 64

    # 체크포인트에서 이어서 학습
    python train_reach.py --headless --num_envs 4096 --checkpoint /path/to/model.pt

주의:
    - 학습은 별도 터미널에서 실행해야 합니다 (Claude 터미널은 타임아웃 있음)
    - GPU 메모리에 따라 num_envs 조절 필요
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
parser = argparse.ArgumentParser(description="E0509 Reach 학습")
parser.add_argument("--num_envs", type=int, default=4096, help="병렬 환경 개수")
parser.add_argument("--max_iterations", type=int, default=5000, help="최대 학습 반복 횟수")
parser.add_argument("--checkpoint", type=str, default=None, help="이어서 학습할 체크포인트")

# AppLauncher 인자
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

# =============================================================================
# Isaac Sim 실행
# =============================================================================
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# Isaac Sim 실행 후 import
import torch
from isaaclab.envs import ManagerBasedRLEnv

from envs.e0509_reach_env import E0509ReachEnvCfg

# RSL-RL
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
    RslRlVecEnvWrapper,
)
from rsl_rl.runners import OnPolicyRunner


def main():
    """메인 학습 함수"""

    # =============================================================================
    # 환경 설정
    # =============================================================================
    env_cfg = E0509ReachEnvCfg()
    env_cfg.scene.num_envs = args.num_envs

    # 환경 생성
    env = ManagerBasedRLEnv(cfg=env_cfg)

    # =============================================================================
    # PPO 설정
    # =============================================================================
    agent_cfg = RslRlOnPolicyRunnerCfg(
        seed=42,
        device="cuda:0",
        num_steps_per_env=24,
        max_iterations=args.max_iterations,
        save_interval=100,
        experiment_name="e0509_reach",
        run_name="reach",
        logger="tensorboard",
        obs_groups={"policy": ["policy"]},

        policy=RslRlPpoActorCriticCfg(
            init_noise_std=0.3,
            actor_hidden_dims=[256, 256, 128],
            critic_hidden_dims=[256, 256, 128],
            activation="elu",
            actor_obs_normalization=False,
            critic_obs_normalization=False,
        ),

        algorithm=RslRlPpoAlgorithmCfg(
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.2,
            entropy_coef=0.01,
            num_learning_epochs=5,
            num_mini_batches=4,
            learning_rate=3e-4,
            schedule="adaptive",
            gamma=0.99,
            lam=0.95,
            desired_kl=0.01,
            max_grad_norm=1.0,
        ),
    )

    # =============================================================================
    # 학습 실행
    # =============================================================================
    # RSL-RL 래퍼
    wrapped_env = RslRlVecEnvWrapper(env)

    # 로그 디렉토리
    log_dir = "./logs/e0509_reach"

    # Runner 생성
    runner = OnPolicyRunner(
        wrapped_env,
        agent_cfg.to_dict(),
        log_dir=log_dir,
        device=agent_cfg.device,
    )

    # 체크포인트 로드
    if args.checkpoint and os.path.exists(args.checkpoint):
        print(f"체크포인트 로드: {args.checkpoint}")
        runner.load(args.checkpoint)

    # 학습 시작
    print("=" * 60)
    print("E0509 Reach 강화학습 시작")
    print("=" * 60)
    print(f"  병렬 환경 수: {args.num_envs}")
    print(f"  최대 반복 횟수: {args.max_iterations}")
    print(f"  로그 디렉토리: {log_dir}")
    if args.checkpoint:
        print(f"  체크포인트: {args.checkpoint}")
    print("=" * 60)

    runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=True)

    # =============================================================================
    # 학습 완료
    # =============================================================================
    runner.save("final_model")
    print("\n학습 완료! 모델 저장됨.")

    wrapped_env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
