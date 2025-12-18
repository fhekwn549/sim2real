"""
Target Tracking 환경 학습 스크립트

Gripper grasp point를 랜덤 target으로 이동하는 학습
학습 후 RealSense로 펜 추적 가능

사용법:
    python train_target_tracking.py --headless --num_envs 2048 --max_iterations 1000
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Target Tracking 학습")
parser.add_argument("--num_envs", type=int, default=2048, help="병렬 환경 개수")
parser.add_argument("--max_iterations", type=int, default=1000, help="최대 학습 반복")
parser.add_argument("--checkpoint", type=str, default=None, help="체크포인트")

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch
from envs.target_tracking_env import TargetTrackingEnv, TargetTrackingEnvCfg

from rsl_rl.runners import OnPolicyRunner
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg
from isaaclab.utils import configclass


@configclass
class TargetTrackingRunnerCfg(RslRlOnPolicyRunnerCfg):
    """Target Tracking PPO 설정"""
    seed = 42
    device = "cuda:0"
    num_steps_per_env = 24
    max_iterations = 1000
    save_interval = 100
    experiment_name = "target_tracking"
    run_name = "sim2real_v1"

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
    env_cfg = TargetTrackingEnvCfg()
    env_cfg.scene.num_envs = args.num_envs

    env = TargetTrackingEnv(cfg=env_cfg)
    env = RslRlVecEnvWrapper(env)

    agent_cfg = TargetTrackingRunnerCfg()
    agent_cfg.max_iterations = args.max_iterations

    log_dir = "./pen_grasp_rl/logs/target_tracking"
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
    print("Target Tracking 학습 (Sim2Real Visual Servoing)")
    print("=" * 60)
    print(f"  환경 수: {args.num_envs}")
    print(f"  최대 반복: {args.max_iterations}")
    print(f"  관찰 차원: {env_cfg.observation_space}")
    print(f"  액션 차원: {env_cfg.action_space}")
    print(f"  네트워크: 128 x 128")
    print("=" * 60)
    print("목표:")
    print("  - Grasp point를 랜덤 target으로 이동")
    print("  - 성공 조건: 거리 < 2cm")
    print("=" * 60)
    print("Target 범위:")
    print(f"  X: {env_cfg.target_pos_range['x']}")
    print(f"  Y: {env_cfg.target_pos_range['y']}")
    print(f"  Z: {env_cfg.target_pos_range['z']}")
    print("=" * 60)

    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    print("\n학습 완료!")
    print(f"모델 저장 위치: {log_dir}")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
