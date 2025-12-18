#!/bin/bash
# Setup script for pen_grasp_rl dependencies in Docker container
# Run this after entering the Docker container

set -e

echo "=========================================="
echo "Installing pen_grasp_rl dependencies..."
echo "=========================================="

# Install rsl_rl from GitHub (use python -m pip for Docker)
python -m pip install git+https://github.com/leggedrobotics/rsl_rl.git

# Install tensordict (required by rsl_rl)
python -m pip install tensordict

echo "=========================================="
echo "Dependencies installed successfully!"
echo "=========================================="
echo ""
echo "To run training:"
echo "  cd /workspace/isaaclab"
echo "  python pen_grasp_rl/scripts/train.py --headless --num_envs 4096"
echo ""
echo "To test policy:"
echo "  python pen_grasp_rl/scripts/play.py --checkpoint <path_to_checkpoint>"
