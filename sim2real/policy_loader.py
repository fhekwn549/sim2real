#!/usr/bin/env python3
"""
강화학습 Policy 로더

Isaac Lab에서 학습된 .pt 파일을 로드하여 추론합니다.

Usage:
    from policy_loader import PolicyLoader

    policy = PolicyLoader(
        checkpoint_path="/path/to/model.pt",
        env_type="target_tracking"
    )

    action = policy.get_action(observation)
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional
from dataclasses import dataclass


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
        obs_dim=18,
        action_dim=6,
        hidden_dims=[128, 128],
        action_scale=0.05,
        description="Target position tracking"
    ),
    "e0509_reach": EnvConfig(
        obs_dim=33,
        action_dim=6,
        hidden_dims=[256, 256, 128],
        action_scale=0.1,
        description="End-effector reaching"
    ),
    "pen_grasp": EnvConfig(
        obs_dim=36,
        action_dim=10,
        hidden_dims=[256, 256, 128],
        action_scale=0.1,
        description="Pen grasping (arm + gripper)"
    ),
}


class ActorNetwork(nn.Module):
    """Actor 네트워크 (MLP)"""

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


class PolicyLoader:
    """강화학습 Policy 로더"""

    def __init__(
        self,
        checkpoint_path: str,
        env_type: str = "target_tracking",
        device: str = "cpu",
        custom_config: Optional[EnvConfig] = None
    ):
        self.device = device
        self.checkpoint_path = checkpoint_path

        if custom_config:
            self.config = custom_config
        elif env_type in ENV_CONFIGS:
            self.config = ENV_CONFIGS[env_type]
        else:
            raise ValueError(f"Unknown env_type: {env_type}")

        self.env_type = env_type
        self.obs_dim = self.config.obs_dim
        self.action_dim = self.config.action_dim
        self.action_scale = self.config.action_scale

        self.actor = ActorNetwork(
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            hidden_dims=self.config.hidden_dims
        ).to(device)

        self._load_checkpoint()

        print(f"[Policy] Loaded: {checkpoint_path}")
        print(f"[Policy] Env: {env_type}, obs_dim: {self.obs_dim}, action_dim: {self.action_dim}")

    def _load_checkpoint(self):
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)

        if "model_state_dict" in checkpoint:
            model_state = checkpoint["model_state_dict"]
        elif "actor" in checkpoint:
            model_state = checkpoint["actor"]
        else:
            model_state = checkpoint

        actor_state = {}
        for key, value in model_state.items():
            if key.startswith("actor."):
                actor_state[key] = value

        if not actor_state:
            actor_state = model_state

        self.actor.load_state_dict(actor_state)
        self.actor.eval()

    @torch.no_grad()
    def get_action(self, observation: np.ndarray, apply_scale: bool = True) -> np.ndarray:
        """관찰값으로부터 액션 계산"""
        if observation.shape[0] != self.obs_dim:
            raise ValueError(f"Expected obs_dim={self.obs_dim}, got {observation.shape[0]}")

        obs_tensor = torch.FloatTensor(observation).unsqueeze(0).to(self.device)
        action_tensor = self.actor(obs_tensor)
        action = action_tensor.squeeze(0).cpu().numpy()

        if apply_scale:
            action = action * self.action_scale

        return action


def list_available_envs():
    """사용 가능한 환경 목록"""
    print("Available environments:")
    for name, config in ENV_CONFIGS.items():
        print(f"  {name}: obs={config.obs_dim}, action={config.action_dim}")


if __name__ == "__main__":
    list_available_envs()
