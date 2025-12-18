"""
Simple Move 환경 학습 스크립트 (Sim2Real 테스트용)

TCP를 5cm 위로 이동 후 Home 복귀하는 간단한 동작 학습

사용법:
    python train_simple_move.py --headless --num_envs 2048 --max_iterations 1000
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Simple Move 학습")
parser.add_argument("--num_envs", type=int, default=2048, help="병렬 환경 개수")
parser.add_argument("--max_iterations", type=int, default=1000, help="최대 학습 반복")
parser.add_argument("--checkpoint", type=str, default=None, help="체크포인트")

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch
from envs.simple_move_env import SimpleMoveEnv, SimpleMoveEnvCfg

from rsl_rl.runners import OnPolicyRunner
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg
from isaaclab.utils import configclass


@configclass
class SimpleMoveRunnerCfg(RslRlOnPolicyRunnerCfg):
    """Simple Move PPO 설정"""
    seed = 42
    device = "cuda:0"
    num_steps_per_env = 24
    max_iterations = 1000
    save_interval = 100
    experiment_name = "simple_move"
    run_name = "sim2real_test"

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.5,
        actor_hidden_dims=[128, 128],  # 간단한 네트워크
        critic_hidden_dims=[128, 128],
        activation="elu",
    )

    algorithm = RslRlPpoAlgorithmCfg(
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
    )


def main():
    env_cfg = SimpleMoveEnvCfg()
    env_cfg.scene.num_envs = args.num_envs

    env = SimpleMoveEnv(cfg=env_cfg)
    env = RslRlVecEnvWrapper(env)

    agent_cfg = SimpleMoveRunnerCfg()
    agent_cfg.max_iterations = args.max_iterations

    log_dir = "./pen_grasp_rl/logs/simple_move"
    os.makedirs(log_dir, exist_ok=True)

    runner = OnPolicyRunner(
        env,
        agent_cfg.to_dict(),
        log_dir=log_dir,
        device=agent_cfg.device,
    )

    if args.checkpoint and os.path.exists(args.checkpoint):
        print(f"체크포인트 로드: {args.checkpoint}")
        runner.load(args.checkpoint)

    print("=" * 60)
    print("Simple Move 학습 (Sim2Real 테스트용)")
    print("=" * 60)
    print(f"  환경 수: {args.num_envs}")
    print(f"  최대 반복: {args.max_iterations}")
    print(f"  관찰 차원: {env_cfg.observation_space}")
    print(f"  액션 차원: {env_cfg.action_space}")
    print("=" * 60)
    print("목표:")
    print("  Phase 1: TCP를 Z축 5cm 위로 이동")
    print("  Phase 2: Home 위치로 복귀")
    print("  성공 조건: 거리 < 1cm")
    print("=" * 60)

    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    print("\n학습 완료!")
    print(f"모델 저장 위치: {log_dir}")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
