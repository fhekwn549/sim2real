"""
Test trained policy for Pen Grasp RL
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Test trained pen grasping policy")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
parser.add_argument("--num_envs", type=int, default=16, help="Number of environments")
parser.add_argument("--env_version", type=str, default="v2", choices=["v1", "v2"],
                    help="환경 버전 (v1: 기존, v2: reach 기반)")

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

# Launch with GUI (no --headless)
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch
import torch.nn as nn

# 환경 버전에 따라 import
if args.env_version == "v2":
    from envs.pen_grasp_env_v2 import PenGraspEnv, PenGraspEnvCfg, PEN_LENGTH
    print("환경: v2 (reach 기반)")
else:
    from envs.pen_grasp_env import PenGraspEnv, PenGraspEnvCfg, PEN_LENGTH
    print("환경: v1 (기존)")

# Visual markers for tip and cap
import isaaclab.sim as sim_utils
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg


class SimplePolicy(nn.Module):
    """Simple MLP policy for inference."""
    def __init__(self, obs_dim, act_dim, hidden_dims=[256, 256, 128]):
        super().__init__()

        layers = []
        in_dim = obs_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ELU())
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, act_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, obs):
        return self.network(obs)


def quat_rotate_vector(quat, vec):
    """Rotate vector by quaternion (w, x, y, z format)."""
    # quat: (N, 4), vec: (N, 3)
    q_w, q_x, q_y, q_z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]

    # Cross product: q_xyz x vec
    t = 2.0 * torch.stack([
        q_y * vec[:, 2] - q_z * vec[:, 1],
        q_z * vec[:, 0] - q_x * vec[:, 2],
        q_x * vec[:, 1] - q_y * vec[:, 0]
    ], dim=-1)

    # Result: vec + q_w * t + q_xyz x t
    result = vec + q_w.unsqueeze(-1) * t + torch.stack([
        q_y * t[:, 2] - q_z * t[:, 1],
        q_z * t[:, 0] - q_x * t[:, 2],
        q_x * t[:, 1] - q_y * t[:, 0]
    ], dim=-1)

    return result


def main():
    # Create environment
    env_cfg = PenGraspEnvCfg()
    env_cfg.scene.num_envs = args.num_envs
    env = PenGraspEnv(cfg=env_cfg)

    # Create visual markers: cap (red), grasp point (green), and z-axis markers
    marker_cfg = VisualizationMarkersCfg(
        prim_path="/World/Visuals/PenMarkers",
        markers={
            "cap": sim_utils.SphereCfg(
                radius=0.015,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),  # Red - pen cap (target)
            ),
            "grasp_point": sim_utils.SphereCfg(
                radius=0.015,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),  # Green - gripper grasp point
            ),
            "pen_axis": sim_utils.SphereCfg(
                radius=0.008,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.5, 1.0)),  # Blue - pen z-axis
            ),
            "gripper_axis": sim_utils.SphereCfg(
                radius=0.008,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 1.0, 0.0)),  # Yellow - gripper z-axis
            ),
        }
    )
    pen_markers = VisualizationMarkers(marker_cfg)

    # Number of points to draw along each axis
    AXIS_POINTS = 5
    AXIS_LENGTH = 0.15  # 15cm line

    # Get observation and action dimensions
    obs_dim = env.observation_manager.group_obs_dim["policy"][0]
    act_dim = env.action_manager.total_action_dim

    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location="cuda:0", weights_only=False)

    # Create simple policy network
    policy = SimplePolicy(obs_dim, act_dim, hidden_dims=[256, 256, 128]).to("cuda:0")

    # Extract actor weights from checkpoint
    model_state = checkpoint["model_state_dict"]

    # Map weights to our simple policy
    # Checkpoint structure: actor.0, actor.2, actor.4, actor.6 (with ELU activations in between)
    # SimplePolicy structure: network.0 (Linear), network.1 (ELU), network.2 (Linear), network.3 (ELU), ...
    policy_state = {}
    actor_layer_map = {
        "actor.0.weight": "network.0.weight",
        "actor.0.bias": "network.0.bias",
        "actor.2.weight": "network.2.weight",
        "actor.2.bias": "network.2.bias",
        "actor.4.weight": "network.4.weight",
        "actor.4.bias": "network.4.bias",
        "actor.6.weight": "network.6.weight",
        "actor.6.bias": "network.6.bias",
    }

    for ckpt_key, policy_key in actor_layer_map.items():
        if ckpt_key in model_state:
            policy_state[policy_key] = model_state[ckpt_key]

    # Try to load weights
    try:
        policy.load_state_dict(policy_state, strict=False)
        print("Loaded policy weights successfully")
        print(f"  Loaded {len(policy_state)} actor layers")
    except Exception as e:
        print(f"Could not load exact weights: {e}")
        print("Using random policy for visualization")

    policy.eval()

    print("=" * 50)
    print("Testing trained policy...")
    print(f"  Obs dim: {obs_dim}")
    print(f"  Act dim: {act_dim}")
    print(f"  Num envs: {args.num_envs}")
    print("=" * 50)

    # Debug: Print robot body names
    print("\n[DEBUG] Robot body names:")
    body_names = env.scene["robot"].data.body_names
    for i, name in enumerate(body_names):
        print(f"  [{i}] {name}")
    print("=" * 50)
    print("Press Ctrl+C to exit")

    # Run test loop
    obs, _ = env.reset()
    policy_obs = obs["policy"]

    step_count = 0

    try:
        while simulation_app.is_running():
            with torch.no_grad():
                actions = policy(policy_obs)
                # Clamp actions to reasonable range
                actions = torch.clamp(actions, -1.0, 1.0)

            obs, rewards, dones, truncated, info = env.step(actions)
            policy_obs = obs["policy"]
            step_count += 1

            # Update pen markers (tip=blue, cap=red, gripper=green)
            pen_pos = env.scene["pen"].data.root_pos_w  # (num_envs, 3)
            pen_quat = env.scene["pen"].data.root_quat_w  # (num_envs, 4) - (w,x,y,z)

            # Get gripper joint positions
            robot = env.scene["robot"]
            l1_pos = robot.data.body_pos_w[:, 7, :]        # [7] gripper_rh_p12_rn_l1
            r1_pos = robot.data.body_pos_w[:, 8, :]        # [8] gripper_rh_p12_rn_r1
            l2_pos = robot.data.body_pos_w[:, 9, :]        # [9] gripper_rh_p12_rn_l2
            r2_pos = robot.data.body_pos_w[:, 10, :]       # [10] gripper_rh_p12_rn_r2

            # Grasp point: (l1+r1)/2 중심에서 손가락 방향으로 2cm
            base_center = (l1_pos + r1_pos) / 2.0
            tip_center = (l2_pos + r2_pos) / 2.0
            finger_dir = tip_center - base_center
            finger_dir = finger_dir / (torch.norm(finger_dir, dim=-1, keepdim=True) + 1e-6)  # normalize
            grasp_point = base_center + finger_dir * 0.02  # 2cm along finger direction

            # Calculate cap position
            half_len = PEN_LENGTH / 2.0
            pen_axis = torch.tensor([[0.0, 0.0, 1.0]], device=pen_pos.device).expand(pen_pos.shape[0], -1)
            pen_axis_world = quat_rotate_vector(pen_quat, pen_axis)
            cap_pos = pen_pos + pen_axis_world * half_len   # Cap (red) - pen +Z 방향이 캡

            # Get gripper z-axis from link_6 orientation
            link6_quat = robot.data.body_quat_w[:, 6, :]  # [6] link_6 orientation
            gripper_z_axis = torch.tensor([[0.0, 0.0, 1.0]], device=pen_pos.device).expand(pen_pos.shape[0], -1)
            gripper_z_world = quat_rotate_vector(link6_quat, gripper_z_axis)

            # Calculate distance
            dist_grasp_to_cap = torch.norm(grasp_point - cap_pos, dim=-1)

            # Calculate axis alignment (dot product of z-axes)
            axis_dot = torch.sum(pen_axis_world * gripper_z_world, dim=-1)  # -1 to 1

            # Combine marker positions: 2 base + AXIS_POINTS*2 per env
            num_envs = pen_pos.shape[0]
            markers_per_env = 2 + AXIS_POINTS * 2  # cap, grasp_point, pen_axis*5, gripper_axis*5
            all_positions = torch.zeros((num_envs * markers_per_env, 3), device=pen_pos.device)
            marker_indices = []

            for i in range(num_envs):
                base_idx = i * markers_per_env
                # 0: cap (red)
                all_positions[base_idx] = cap_pos[i]
                marker_indices.append(0)
                # 1: grasp_point (green)
                all_positions[base_idx + 1] = grasp_point[i]
                marker_indices.append(1)

                # Pen z-axis markers (blue) - from pen center along z-axis
                for j in range(AXIS_POINTS):
                    t = (j + 1) / AXIS_POINTS * AXIS_LENGTH
                    all_positions[base_idx + 2 + j] = pen_pos[i] + pen_axis_world[i] * t
                    marker_indices.append(2)  # pen_axis

                # Gripper z-axis markers (yellow) - from grasp point along z-axis
                for j in range(AXIS_POINTS):
                    t = (j + 1) / AXIS_POINTS * AXIS_LENGTH
                    all_positions[base_idx + 2 + AXIS_POINTS + j] = grasp_point[i] + gripper_z_world[i] * t
                    marker_indices.append(3)  # gripper_axis

            pen_markers.visualize(translations=all_positions, marker_indices=marker_indices)

            # Print detailed info every 50 steps
            if step_count % 50 == 0:
                print(f"\n{'='*60}")
                print(f"Step {step_count}")
                print(f"  Mean Reward:      {rewards.mean().item():>8.4f}")
                print(f"  GraspPoint→Cap:   {dist_grasp_to_cap.mean().item()*100:>6.2f} cm  (min: {dist_grasp_to_cap.min().item()*100:.2f})")
                print(f"  Z-axis alignment: {axis_dot.mean().item():>6.3f}  (1.0=parallel, -1.0=opposite, 0=perpendicular)")
                print(f"{'='*60}")

    except KeyboardInterrupt:
        print("\nTest stopped by user")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
