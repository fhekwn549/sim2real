"""
펜 잡기 작업 공간 설정 (Sim + Real 공유)

이 파일은 시뮬레이션과 실제 로봇 모두에서 사용되는
펜 위치/각도 유효 범위를 정의합니다.

사용처:
- pen_grasp_rl/envs/e0509_osc_env.py (시뮬레이션)
- sim2real/sim2real/run_sim2real.py (실제 로봇)

변경 시 양쪽 모두에 적용됩니다!
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple


@dataclass
class PenWorkspaceConfig:
    """펜 작업 공간 설정 (Sim2Real 공유)"""

    # =========================================================================
    # 펜 캡 위치 유효 범위 (로봇 베이스 기준, 미터)
    # =========================================================================
    # e0509_osc_env.py의 실제 학습 설정과 동일하게 맞춤
    pen_x_range: Tuple[float, float] = (0.25, 0.55)   # X: 로봇 앞쪽 방향 (25~55cm)
    pen_y_range: Tuple[float, float] = (-0.12, 0.12)  # Y: 좌우 (-12~12cm)
    pen_z_range: Tuple[float, float] = (0.22, 0.35)   # Z: 높이 (22~35cm)

    # =========================================================================
    # 펜 기울기 유효 범위 (라디안)
    # =========================================================================
    pen_tilt_max: float = 0.79  # 최대 45도
    pen_tilt_min: float = 0.0   # 최소 0도 (수직)

    # =========================================================================
    # 로봇 작업 공간 (안전 범위, 미터)
    # =========================================================================
    # TCP가 이 범위를 벗어나면 동작 중지
    workspace_x: Tuple[float, float] = (-0.1, 0.6)
    workspace_y: Tuple[float, float] = (-0.4, 0.4)
    workspace_z: Tuple[float, float] = (0.10, 0.5)

    # =========================================================================
    # 관절 제한 (DART platform 기준, 도)
    # =========================================================================
    # USD 파일에도 적용됨
    joint_limits_deg = {
        "joint_1": (-360, 360),
        "joint_2": (-95, 95),     # 자가 충돌 방지
        "joint_3": (-135, 135),
        "joint_4": (-360, 360),
        "joint_5": (-135, 135),
        "joint_6": (-360, 360),
    }

    def is_pen_position_valid(self, x: float, y: float, z: float) -> Tuple[bool, str]:
        """
        펜 캡 위치가 유효한지 검사

        Args:
            x, y, z: 펜 캡 위치 (로봇 좌표계, 미터)

        Returns:
            (is_valid, message)
        """
        if not (self.pen_x_range[0] <= x <= self.pen_x_range[1]):
            return False, f"X 범위 초과: {x:.3f}m (유효: {self.pen_x_range[0]:.2f}~{self.pen_x_range[1]:.2f}m)"

        if not (self.pen_y_range[0] <= y <= self.pen_y_range[1]):
            return False, f"Y 범위 초과: {y:.3f}m (유효: {self.pen_y_range[0]:.2f}~{self.pen_y_range[1]:.2f}m)"

        if not (self.pen_z_range[0] <= z <= self.pen_z_range[1]):
            return False, f"Z 범위 초과: {z:.3f}m (유효: {self.pen_z_range[0]:.2f}~{self.pen_z_range[1]:.2f}m)"

        return True, "OK"

    def is_pen_tilt_valid(self, tilt_rad: float) -> Tuple[bool, str]:
        """
        펜 기울기가 유효한지 검사

        Args:
            tilt_rad: 펜 기울기 (라디안, 수직에서 벗어난 각도)

        Returns:
            (is_valid, message)
        """
        tilt_deg = np.degrees(tilt_rad)

        if tilt_rad > self.pen_tilt_max:
            max_deg = np.degrees(self.pen_tilt_max)
            return False, f"기울기 초과: {tilt_deg:.1f}° (최대: {max_deg:.0f}°)"

        return True, "OK"

    def is_workspace_valid(self, x: float, y: float, z: float) -> Tuple[bool, str]:
        """
        TCP 위치가 작업 공간 내에 있는지 검사

        Args:
            x, y, z: TCP 위치 (로봇 좌표계, 미터)

        Returns:
            (is_valid, message)
        """
        if not (self.workspace_x[0] <= x <= self.workspace_x[1]):
            return False, f"X 작업영역 초과: {x:.3f}m"

        if not (self.workspace_y[0] <= y <= self.workspace_y[1]):
            return False, f"Y 작업영역 초과: {y:.3f}m"

        if not (self.workspace_z[0] <= z <= self.workspace_z[1]):
            return False, f"Z 작업영역 초과: {z:.3f}m"

        return True, "OK"

    def get_pen_spawn_range_dict(self) -> dict:
        """OSC 환경에서 사용하는 형식으로 반환"""
        return {
            "x": self.pen_x_range,
            "y": self.pen_y_range,
            "z": self.pen_z_range,
        }

    def print_config(self):
        """설정 출력"""
        print("=" * 50)
        print("펜 작업 공간 설정")
        print("=" * 50)
        print(f"펜 캡 X 범위: {self.pen_x_range[0]*100:.0f} ~ {self.pen_x_range[1]*100:.0f} cm")
        print(f"펜 캡 Y 범위: {self.pen_y_range[0]*100:.0f} ~ {self.pen_y_range[1]*100:.0f} cm")
        print(f"펜 캡 Z 범위: {self.pen_z_range[0]*100:.0f} ~ {self.pen_z_range[1]*100:.0f} cm")
        print(f"펜 최대 기울기: {np.degrees(self.pen_tilt_max):.0f}°")
        print("-" * 50)
        print(f"작업공간 X: {self.workspace_x[0]*100:.0f} ~ {self.workspace_x[1]*100:.0f} cm")
        print(f"작업공간 Y: {self.workspace_y[0]*100:.0f} ~ {self.workspace_y[1]*100:.0f} cm")
        print(f"작업공간 Z: {self.workspace_z[0]*100:.0f} ~ {self.workspace_z[1]*100:.0f} cm")
        print("=" * 50)


# 기본 설정 인스턴스 (import해서 사용)
DEFAULT_PEN_WORKSPACE = PenWorkspaceConfig()


def calculate_tilt_from_direction(direction: np.ndarray) -> float:
    """
    펜 방향 벡터에서 기울기 계산

    Args:
        direction: 펜 방향 벡터 (정규화됨, 3D)

    Returns:
        tilt_rad: 수직에서 벗어난 각도 (라디안)
    """
    # 펜이 수직일 때 direction = [0, 0, 1] 또는 [0, 0, -1]
    # 기울기 = arccos(|z|)
    z_component = abs(direction[2])
    z_component = np.clip(z_component, 0, 1)  # 수치 오류 방지
    tilt_rad = np.arccos(z_component)
    return tilt_rad
