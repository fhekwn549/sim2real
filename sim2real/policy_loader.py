#!/usr/bin/env python3
"""
강화학습 Policy 로더

Isaac Lab에서 학습된 .pt 파일을 로드하여 추론합니다.
다양한 환경(target_tracking, e0509_reach 등)을 지원합니다.

Usage:
    from policy_loader import PolicyLoader

    policy = PolicyLoader(
        checkpoint_path="/path/to/model.pt",
        env_type="target_tracking"  # 또는 "e0509_reach"
    )

    action = policy.get_action(observation)  # numpy array
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Dict, Any
from dataclasses import dataclass


# =============================================================================
# 환경별 설정
# =============================================================================
@dataclass
class EnvConfig:
    """환경별 observation/action 설정"""
    obs_dim: int
    action_dim: int
    hidden_dims: list
    action_scale: float
    description: str


ENV_CONFIGS = {
    "target_tracking": EnvConfig(
        obs_dim=18,         # joint_pos(6) + joint_vel(6) + grasp_pos(3) + target_pos(3)
        action_dim=6,
        hidden_dims=[128, 128],
        action_scale=0.05,
        description="Target position tracking (simple)"
    ),
    "e0509_reach": EnvConfig(
        obs_dim=33,         # joint_pos(6) + joint_vel(6) + ee_pos(3) + ee_quat(4) + target(3) + ...
        action_dim=6,
        hidden_dims=[256, 256, 128],
        action_scale=0.1,
        description="End-effector reaching task"
    ),
    "simple_move": EnvConfig(
        obs_dim=18,
        action_dim=6,
        hidden_dims=[128, 128],
        action_scale=0.05,
        description="Simple movement task"
    ),
    "pen_grasp": EnvConfig(
        obs_dim=36,         # joint_pos(10) + joint_vel(10) + ee_pos(3) + pen_pos(3) + pen_quat(4) + ...
        action_dim=10,      # arm(6) + gripper(4)
        hidden_dims=[256, 256, 128],
        action_scale=0.1,
        description="Pen grasping task"
    ),
}


# =============================================================================
# Actor 네트워크
# =============================================================================
class ActorNetwork(nn.Module):
    """
    Actor 네트워크 (MLP)

    구조: obs -> hidden layers (ELU) -> action
    """

    def __init__(self, obs_dim: int, action_dim: int, hidden_dims: list = [256, 256, 128]):
        super().__init__()

        layers = []
        prev_dim = obs_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ELU())
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, action_dim))

        self.actor = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.actor(obs)


# =============================================================================
# Policy 로더
# =============================================================================
class PolicyLoader:
    """강화학습 Policy 로더 및 추론기"""

    def __init__(
        self,
        checkpoint_path: str,
        env_type: str = "target_tracking",
        device: str = "cpu",
        custom_config: Optional[EnvConfig] = None
    ):
        """
        Args:
            checkpoint_path: .pt 파일 경로
            env_type: 환경 타입 ("target_tracking", "e0509_reach", ...)
            device: 실행 디바이스 ("cpu" 또는 "cuda")
            custom_config: 커스텀 설정 (ENV_CONFIGS 대신 사용)
        """
        self.device = device
        self.checkpoint_path = checkpoint_path

        # 환경 설정 로드
        if custom_config:
            self.config = custom_config
        elif env_type in ENV_CONFIGS:
            self.config = ENV_CONFIGS[env_type]
        else:
            raise ValueError(f"Unknown env_type: {env_type}. Available: {list(ENV_CONFIGS.keys())}")

        self.env_type = env_type
        self.obs_dim = self.config.obs_dim
        self.action_dim = self.config.action_dim
        self.action_scale = self.config.action_scale

        # Actor 네트워크 생성 및 로드
        self.actor = ActorNetwork(
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            hidden_dims=self.config.hidden_dims
        ).to(device)

        self._load_checkpoint()

        print(f"[Policy] Loaded: {checkpoint_path}")
        print(f"[Policy] Env: {env_type} ({self.config.description})")
        print(f"[Policy] Obs dim: {self.obs_dim}, Action dim: {self.action_dim}")
        print(f"[Policy] Action scale: {self.action_scale}")

    def _load_checkpoint(self):
        """체크포인트에서 Actor 가중치 로드"""
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)

        # RSL-RL 형식
        if "model_state_dict" in checkpoint:
            model_state = checkpoint["model_state_dict"]
        # 다른 형식
        elif "actor" in checkpoint:
            model_state = checkpoint["actor"]
        else:
            model_state = checkpoint

        # Actor 가중치만 추출
        actor_state = {}
        for key, value in model_state.items():
            if key.startswith("actor."):
                actor_state[key] = value

        if not actor_state:
            # actor. prefix 없이 저장된 경우
            actor_state = model_state

        self.actor.load_state_dict(actor_state)
        self.actor.eval()

    @torch.no_grad()
    def get_action(self, observation: np.ndarray, apply_scale: bool = True) -> np.ndarray:
        """
        관찰값으로부터 액션 계산

        Args:
            observation: (obs_dim,) numpy 배열
            apply_scale: action_scale 적용 여부

        Returns:
            action: (action_dim,) numpy 배열
        """
        # 입력 검증
        if observation.shape[0] != self.obs_dim:
            raise ValueError(f"Expected obs_dim={self.obs_dim}, got {observation.shape[0]}")

        # numpy -> tensor
        obs_tensor = torch.FloatTensor(observation).unsqueeze(0).to(self.device)

        # 추론
        action_tensor = self.actor(obs_tensor)

        # tensor -> numpy
        action = action_tensor.squeeze(0).cpu().numpy()

        if apply_scale:
            action = action * self.action_scale

        return action

    @torch.no_grad()
    def get_action_batch(self, observations: np.ndarray, apply_scale: bool = True) -> np.ndarray:
        """
        배치 추론

        Args:
            observations: (batch, obs_dim) numpy 배열

        Returns:
            actions: (batch, action_dim) numpy 배열
        """
        obs_tensor = torch.FloatTensor(observations).to(self.device)
        action_tensor = self.actor(obs_tensor)
        actions = action_tensor.cpu().numpy()

        if apply_scale:
            actions = actions * self.action_scale

        return actions

    def get_config(self) -> EnvConfig:
        """환경 설정 반환"""
        return self.config


# =============================================================================
# 유틸리티 함수
# =============================================================================
def list_available_envs():
    """사용 가능한 환경 목록 출력"""
    print("Available environments:")
    print("-" * 60)
    for name, config in ENV_CONFIGS.items():
        print(f"  {name}:")
        print(f"    obs_dim: {config.obs_dim}, action_dim: {config.action_dim}")
        print(f"    hidden_dims: {config.hidden_dims}")
        print(f"    action_scale: {config.action_scale}")
        print(f"    description: {config.description}")
        print()


def test_inference(checkpoint_path: str, env_type: str = "target_tracking"):
    """추론 테스트 (랜덤 입력)"""
    print("=" * 60)
    print("Policy Inference Test")
    print("=" * 60)

    policy = PolicyLoader(checkpoint_path, env_type)

    # 랜덤 관찰값으로 테스트
    print("\nRandom observation inference:")
    for i in range(5):
        obs = np.random.randn(policy.obs_dim).astype(np.float32)
        action = policy.get_action(obs)
        print(f"  Test {i+1}: action = {action.round(4)}")

    print("\nInference test passed!")


# =============================================================================
# 테스트
# =============================================================================
if __name__ == "__main__":
    list_available_envs()

    # 체크포인트가 있으면 테스트
    import os
    test_paths = [
        os.path.join(os.path.expanduser("~"), "IsaacLab/final_model"),
        os.path.join(os.path.expanduser("~"), "IsaacLab/logs/e0509_reach"),
    ]

    for path in test_paths:
        if os.path.exists(path):
            print(f"\nTesting with: {path}")
            try:
                test_inference(path, "target_tracking")
            except Exception as e:
                print(f"Test failed: {e}")
            break
