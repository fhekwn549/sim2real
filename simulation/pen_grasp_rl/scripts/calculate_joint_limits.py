#!/usr/bin/env python3
"""
작업 공간 기반 관절 범위 계산

펜이 위치할 수 있는 범위에서 IK를 계산하여
실제 필요한 관절 범위를 찾습니다.
"""
import numpy as np
from scipy.spatial.transform import Rotation as R

# Doosan E0509 DH 파라미터 (근사값)
# 실제 값은 로봇 매뉴얼 참조
DH_PARAMS = {
    'd1': 0.1555,   # base to joint 2
    'a2': 0.409,    # link 2 length
    'a3': 0.367,    # link 3 length
    'd4': 0.125,    # joint 4 offset
    'd5': 0.100,    # joint 5 offset
    'd6': 0.094,    # flange
}

# 작업 공간 정의 (펜 위치 범위)
WORKSPACE = {
    "x": (0.30, 0.50),
    "y": (-0.20, 0.20),
    "z": (0.20, 0.57),  # 펜 z (0.20~0.50) + pre-grasp 7cm
}

# 그리퍼가 아래를 향할 때의 방향 (Z축이 -Z를 향함)
TARGET_ORIENTATION = R.from_euler('xyz', [180, 0, 0], degrees=True).as_matrix()


def inverse_kinematics_analytic(target_pos, target_rot=None):
    """
    Doosan E0509 해석적 역기구학 (근사)

    실제로는 로봇 SDK의 IK를 사용하는 것이 정확하지만,
    관절 범위 추정용으로 간단한 기하학적 IK 사용
    """
    x, y, z = target_pos

    # Joint 1: base rotation (XY 평면에서의 방향)
    j1 = np.arctan2(y, x)

    # Wrist center 계산 (그리퍼 오프셋 고려)
    wrist_offset = 0.15  # 대략적인 그리퍼 길이

    # XY 평면에서의 거리
    r_xy = np.sqrt(x**2 + y**2)

    # Joint 2, 3 계산 (2-link IK)
    L1 = DH_PARAMS['a2']  # upper arm
    L2 = DH_PARAMS['a3']  # forearm

    # wrist center까지의 거리
    wc_x = r_xy - 0.05  # 약간의 오프셋
    wc_z = z - DH_PARAMS['d1']

    d = np.sqrt(wc_x**2 + wc_z**2)

    # 도달 가능성 체크
    if d > L1 + L2 or d < abs(L1 - L2):
        return None

    # Elbow angle (joint 3)
    cos_j3 = (d**2 - L1**2 - L2**2) / (2 * L1 * L2)
    cos_j3 = np.clip(cos_j3, -1, 1)
    j3 = np.arccos(cos_j3)  # elbow down configuration

    # Shoulder angle (joint 2)
    alpha = np.arctan2(wc_z, wc_x)
    beta = np.arctan2(L2 * np.sin(j3), L1 + L2 * np.cos(j3))
    j2 = alpha - beta

    # Joint 4, 5, 6 (wrist - 간단히 0으로 시작)
    # 실제로는 target_rot에 따라 계산해야 함
    j4 = 0.0
    j5 = np.pi / 2  # 그리퍼가 아래를 향하도록
    j6 = 0.0

    return np.array([j1, j2, j3, j4, j5, j6])


def sample_workspace_and_compute_ik(num_samples=1000):
    """
    작업 공간을 샘플링하고 IK 계산
    """
    valid_configs = []

    for _ in range(num_samples):
        # 랜덤 타겟 위치
        x = np.random.uniform(*WORKSPACE["x"])
        y = np.random.uniform(*WORKSPACE["y"])
        z = np.random.uniform(*WORKSPACE["z"])

        target_pos = np.array([x, y, z])

        # IK 계산
        joint_angles = inverse_kinematics_analytic(target_pos)

        if joint_angles is not None:
            valid_configs.append(joint_angles)

    return np.array(valid_configs)


def compute_joint_ranges(configs, margin_deg=10):
    """
    유효한 관절 설정에서 범위 계산
    """
    if len(configs) == 0:
        print("유효한 IK 솔루션 없음!")
        return None

    configs_deg = np.degrees(configs)

    ranges = []
    for i in range(6):
        min_val = configs_deg[:, i].min()
        max_val = configs_deg[:, i].max()

        # 마진 추가
        min_val -= margin_deg
        max_val += margin_deg

        ranges.append((min_val, max_val))

    return ranges


def main():
    print("=" * 60)
    print("  작업 공간 기반 관절 범위 계산")
    print("=" * 60)
    print(f"\n작업 공간:")
    print(f"  X: {WORKSPACE['x'][0]:.2f} ~ {WORKSPACE['x'][1]:.2f} m")
    print(f"  Y: {WORKSPACE['y'][0]:.2f} ~ {WORKSPACE['y'][1]:.2f} m")
    print(f"  Z: {WORKSPACE['z'][0]:.2f} ~ {WORKSPACE['z'][1]:.2f} m")

    print("\n샘플링 중...")
    configs = sample_workspace_and_compute_ik(num_samples=5000)
    print(f"유효한 IK 솔루션: {len(configs)}/{5000}")

    if len(configs) > 0:
        ranges = compute_joint_ranges(configs, margin_deg=15)

        print("\n" + "=" * 60)
        print("  추천 관절 범위 (도)")
        print("=" * 60)

        joint_names = ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"]

        for i, (name, (min_v, max_v)) in enumerate(zip(joint_names, ranges)):
            print(f"  {name}: ({min_v:7.1f}, {max_v:7.1f})")

        print("\n" + "=" * 60)
        print("  Python 코드 형식")
        print("=" * 60)
        print("joint_limits_deg = [")
        for i, (min_v, max_v) in enumerate(ranges):
            print(f"    ({min_v:.1f}, {max_v:.1f}),  # joint_{i+1}")
        print("]")

        print("\n" + "=" * 60)
        print("  라디안 형식")
        print("=" * 60)
        print("joint_limits_rad = [")
        for i, (min_v, max_v) in enumerate(ranges):
            print(f"    ({np.radians(min_v):.3f}, {np.radians(max_v):.3f}),  # joint_{i+1}")
        print("]")

        # 통계
        print("\n" + "=" * 60)
        print("  관절 사용 통계")
        print("=" * 60)
        configs_deg = np.degrees(configs)
        for i, name in enumerate(joint_names):
            mean = configs_deg[:, i].mean()
            std = configs_deg[:, i].std()
            print(f"  {name}: mean={mean:7.1f}°, std={std:5.1f}°")


if __name__ == "__main__":
    main()
