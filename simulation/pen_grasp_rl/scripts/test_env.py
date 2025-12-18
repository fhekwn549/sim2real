"""
Pen Grasp 환경 테스트 스크립트

이 스크립트는 펜 잡기 환경이 올바르게 동작하는지 테스트합니다.
학습 전에 환경 설정이 제대로 되었는지 확인하는 용도로 사용합니다.

주요 확인 사항:
    - 로봇과 펜이 올바른 위치에 스폰되는지
    - 관측값과 액션 차원이 맞는지
    - 리워드가 정상적으로 계산되는지

사용법:
    # GUI 모드로 실행 (시각적 확인)
    python test_env.py --num_envs 16

    # Headless 모드로 실행 (빠른 테스트)
    python test_env.py --headless --num_envs 16
"""
import argparse
import os
import sys

# 프로젝트 경로 추가 (envs 모듈 import를 위해)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from isaaclab.app import AppLauncher

# ============================================================
# 명령행 인자 파싱
# ============================================================
parser = argparse.ArgumentParser(description="펜 잡기 환경 테스트")
parser.add_argument("--num_envs", type=int, default=16,
                    help="테스트할 환경 개수 (기본: 16)")

# AppLauncher 인자 추가 (--headless 등)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

# ============================================================
# Isaac Sim 실행
# ============================================================
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# Isaac Sim 실행 후 import (순서 중요!)
import torch
from envs.pen_grasp_env import PenGraspEnv, PenGraspEnvCfg


def main():
    """메인 테스트 함수"""

    # ============================================================
    # 환경 생성
    # ============================================================
    env_cfg = PenGraspEnvCfg()
    env_cfg.scene.num_envs = args.num_envs

    print("=" * 60)
    print("펜 잡기 환경 생성 중...")
    print("=" * 60)

    env = PenGraspEnv(cfg=env_cfg)

    # 환경 정보 출력
    print(f"\n환경 생성 완료!")
    print(f"  병렬 환경 수: {env.num_envs}")
    print(f"  관측 차원: {env.observation_manager.group_obs_dim}")
    print(f"  액션 차원: {env.action_manager.total_action_dim}")

    # ============================================================
    # 테스트 루프
    # ============================================================
    print("\n" + "=" * 60)
    print("테스트 루프 시작 (500 스텝)")
    print("=" * 60)

    # 환경 초기화
    obs, _ = env.reset()
    step_count = 0

    while simulation_app.is_running() and step_count < 500:
        with torch.inference_mode():
            # 랜덤 액션 생성 (작은 크기로 제한)
            # 액션: [joint1, joint2, joint3, joint4, joint5, joint6, gripper]
            actions = torch.randn(env.num_envs, 7, device=env.device) * 0.1

            # 환경 스텝 실행
            # 반환값: obs, reward, terminated, truncated, info
            obs, reward, terminated, truncated, info = env.step(actions)

            # 주기적으로 상태 출력
            if step_count % 50 == 0:
                print(f"스텝 {step_count:4d} | "
                      f"평균 리워드: {reward.mean().item():+.4f} | "
                      f"종료된 환경: {terminated.sum().item()}")

            step_count += 1

    # ============================================================
    # 테스트 완료
    # ============================================================
    print("\n" + "=" * 60)
    print("테스트 완료!")
    print("=" * 60)

    # 환경 정리
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
