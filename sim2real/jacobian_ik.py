#!/usr/bin/env python3
"""
Jacobian 기반 Differential IK

Isaac Lab V7 환경과 동일한 방식의 IK를 구현합니다.
- DLS (Damped Least Squares) 방식
- URDF 기반 Jacobian 계산

사용법:
    from jacobian_ik import JacobianIK

    ik = JacobianIK()  # URDF 자동 로드

    delta_q = ik.compute(
        q=current_joint_rad,
        delta_pos=np.array([0.01, 0.0, 0.0])  # 1cm 이동
    )

    target_q = current_joint_rad + delta_q
"""

import numpy as np
from typing import Optional, Tuple

# roboticstoolbox 사용 시도
try:
    import roboticstoolbox as rtb
    from spatialmath import SE3
    RTB_AVAILABLE = True
except ImportError:
    RTB_AVAILABLE = False
    print("[JacobianIK] roboticstoolbox 미설치, 수치적 Jacobian 사용")


# =============================================================================
# URDF 경로
# =============================================================================
import os
DEFAULT_URDF_PATH = os.path.join(os.path.expanduser("~"), "doosan_ws/src/doosan-robot2/dsr_description2/urdf/e0509.urdf")


# =============================================================================
# Jacobian IK (roboticstoolbox 사용)
# =============================================================================
class JacobianIK_RTB:
    """roboticstoolbox 기반 Jacobian IK"""

    def __init__(self, urdf_path: str = DEFAULT_URDF_PATH, lambda_val: float = 0.05):
        """
        Args:
            urdf_path: URDF 파일 경로
            lambda_val: DLS damping factor (V7과 동일하게 0.05)
        """
        self.lambda_val = lambda_val

        print(f"[JacobianIK] URDF 로드: {urdf_path}")
        self.robot = rtb.Robot.URDF(urdf_path)

        # End-effector를 link_6로 명시적 설정
        ee_link = None
        for link in self.robot.links:
            if link.name == "link_6":
                ee_link = link
                break

        if ee_link is not None:
            self.robot._ee_links = [ee_link]
            print(f"[JacobianIK] End-effector 설정: link_6")
        else:
            print(f"[JacobianIK] Warning: link_6를 찾을 수 없음, 기본 EE 사용")

        # 관절 정보 출력
        print(f"[JacobianIK] 로봇: {self.robot.name}")
        print(f"[JacobianIK] 관절 수: {self.robot.n}")
        print(f"[JacobianIK] End-effector: {self.robot.ee_links[0].name}")

    def forward_kinematics(self, q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """순방향 기구학

        Args:
            q: 관절 각도 [6] (라디안)

        Returns:
            pos: TCP 위치 [3]
            rot: 회전 행렬 [3, 3]
        """
        T = self.robot.fkine(q)
        pos = T.t
        rot = T.R
        return pos, rot

    def compute_jacobian(self, q: np.ndarray) -> np.ndarray:
        """Jacobian 계산

        Args:
            q: 관절 각도 [6] (라디안)

        Returns:
            J: Jacobian [6, 6] (위치 3 + 자세 3)
        """
        return self.robot.jacob0(q)

    def compute(
        self,
        q: np.ndarray,
        delta_pos: np.ndarray,
        delta_rot: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Differential IK 계산 (DLS 방식)

        Args:
            q: 현재 관절 각도 [6] (라디안)
            delta_pos: TCP 위치 델타 [3] (미터)
            delta_rot: TCP 회전 델타 [3] (라디안, 선택적)

        Returns:
            delta_q: 관절 각도 델타 [6] (라디안)
        """
        # Jacobian 계산
        J = self.compute_jacobian(q)  # [6, 6]

        if delta_rot is None:
            # 위치만 사용 (V7과 동일)
            J_pos = J[:3, :]  # [3, 6]
            delta_x = delta_pos  # [3]

            # DLS: Δq = Jᵀ(JJᵀ + λ²I)⁻¹ × Δx
            JJT = J_pos @ J_pos.T  # [3, 3]
            damped = JJT + (self.lambda_val ** 2) * np.eye(3)
            delta_q = J_pos.T @ np.linalg.solve(damped, delta_x)
        else:
            # 위치 + 자세
            delta_x = np.concatenate([delta_pos, delta_rot])  # [6]

            # DLS: Δq = Jᵀ(JJᵀ + λ²I)⁻¹ × Δx
            JJT = J @ J.T  # [6, 6]
            damped = JJT + (self.lambda_val ** 2) * np.eye(6)
            delta_q = J.T @ np.linalg.solve(damped, delta_x)

        return delta_q


# =============================================================================
# Jacobian IK (URDF 변환 직접 구현 - roboticstoolbox 없이)
# =============================================================================
class JacobianIK_Numerical:
    """URDF 변환 기반 Jacobian IK (라이브러리 없이)

    E0509 URDF의 조인트 변환을 직접 구현합니다.
    """

    # URDF 조인트 정의 (xyz, rpy, axis=z)
    # joint_1: xyz="0 0 0.2045" rpy="0 0 0"
    # joint_2: xyz="0 0 0" rpy="0 -1.5708 -1.5708"
    # joint_3: xyz="0.373 0 0" rpy="0 0 1.5708"
    # joint_4: xyz="0 -0.373 0" rpy="1.5708 0 0"
    # joint_5: xyz="0 0 0" rpy="-1.5708 0 0"
    # joint_6: xyz="0 -0.1725 0" rpy="1.5708 0 0"

    JOINTS = [
        # (xyz, rpy)
        ([0.0, 0.0, 0.2045], [0.0, 0.0, 0.0]),               # joint_1
        ([0.0, 0.0, 0.0], [0.0, -np.pi/2, -np.pi/2]),        # joint_2
        ([0.373, 0.0, 0.0], [0.0, 0.0, np.pi/2]),            # joint_3
        ([0.0, -0.373, 0.0], [np.pi/2, 0.0, 0.0]),           # joint_4
        ([0.0, 0.0, 0.0], [-np.pi/2, 0.0, 0.0]),             # joint_5
        ([0.0, -0.1725, 0.0], [np.pi/2, 0.0, 0.0]),          # joint_6
    ]

    def __init__(self, lambda_val: float = 0.05):
        self.lambda_val = lambda_val
        print("[JacobianIK] URDF 변환 기반 수치적 Jacobian 모드")

    def _rot_x(self, angle: float) -> np.ndarray:
        """X축 회전 행렬"""
        c, s = np.cos(angle), np.sin(angle)
        return np.array([
            [1, 0, 0],
            [0, c, -s],
            [0, s, c]
        ])

    def _rot_y(self, angle: float) -> np.ndarray:
        """Y축 회전 행렬"""
        c, s = np.cos(angle), np.sin(angle)
        return np.array([
            [c, 0, s],
            [0, 1, 0],
            [-s, 0, c]
        ])

    def _rot_z(self, angle: float) -> np.ndarray:
        """Z축 회전 행렬"""
        c, s = np.cos(angle), np.sin(angle)
        return np.array([
            [c, -s, 0],
            [s, c, 0],
            [0, 0, 1]
        ])

    def _rpy_to_rot(self, rpy: list) -> np.ndarray:
        """RPY (Roll-Pitch-Yaw) → 회전 행렬 (XYZ 순서)"""
        roll, pitch, yaw = rpy
        return self._rot_z(yaw) @ self._rot_y(pitch) @ self._rot_x(roll)

    def _make_transform(self, xyz: list, rpy: list, joint_angle: float = 0.0) -> np.ndarray:
        """URDF 조인트 변환 행렬 생성

        1. 먼저 xyz 이동
        2. rpy 회전 적용
        3. 조인트 회전 (z축)
        """
        T = np.eye(4)

        # 이동
        T[:3, 3] = xyz

        # 고정 회전 (rpy)
        T[:3, :3] = self._rpy_to_rot(rpy)

        # 조인트 회전 (z축)
        T_joint = np.eye(4)
        T_joint[:3, :3] = self._rot_z(joint_angle)

        return T @ T_joint

    def forward_kinematics(self, q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """순방향 기구학 (URDF 변환 기반)"""
        T = np.eye(4)

        for i, (xyz, rpy) in enumerate(self.JOINTS):
            T_joint = self._make_transform(xyz, rpy, q[i])
            T = T @ T_joint

        return T[:3, 3], T[:3, :3]

    def compute_jacobian(self, q: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        """수치적 Jacobian 계산 (finite difference)"""
        J = np.zeros((6, 6))

        pos0, rot0 = self.forward_kinematics(q)

        for i in range(6):
            q_plus = q.copy()
            q_plus[i] += eps

            pos_plus, rot_plus = self.forward_kinematics(q_plus)

            # 위치 미분
            J[:3, i] = (pos_plus - pos0) / eps

            # 자세 미분 (회전 행렬 차이에서 각속도 추출)
            dR = rot_plus @ rot0.T
            # 로그 맵 근사
            J[3, i] = (dR[2, 1] - dR[1, 2]) / (2 * eps)
            J[4, i] = (dR[0, 2] - dR[2, 0]) / (2 * eps)
            J[5, i] = (dR[1, 0] - dR[0, 1]) / (2 * eps)

        return J

    def compute(
        self,
        q: np.ndarray,
        delta_pos: np.ndarray,
        delta_rot: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Differential IK 계산"""
        J = self.compute_jacobian(q)

        if delta_rot is None:
            J_pos = J[:3, :]
            delta_x = delta_pos

            JJT = J_pos @ J_pos.T
            damped = JJT + (self.lambda_val ** 2) * np.eye(3)
            delta_q = J_pos.T @ np.linalg.solve(damped, delta_x)
        else:
            delta_x = np.concatenate([delta_pos, delta_rot])

            JJT = J @ J.T
            damped = JJT + (self.lambda_val ** 2) * np.eye(6)
            delta_q = J.T @ np.linalg.solve(damped, delta_x)

        return delta_q


# =============================================================================
# 통합 인터페이스
# =============================================================================
class JacobianIK:
    """Jacobian IK 통합 인터페이스

    roboticstoolbox 있으면 사용, 없으면 수치적 방식 사용
    """

    def __init__(
        self,
        urdf_path: str = DEFAULT_URDF_PATH,
        lambda_val: float = 0.05,
        force_numerical: bool = False
    ):
        """
        Args:
            urdf_path: URDF 파일 경로
            lambda_val: DLS damping factor (V7: 0.05)
            force_numerical: True면 강제로 수치적 방식 사용
        """
        if RTB_AVAILABLE and not force_numerical:
            try:
                self._ik = JacobianIK_RTB(urdf_path, lambda_val)
                self.method = "roboticstoolbox"
            except Exception as e:
                print(f"[JacobianIK] RTB 로드 실패: {e}")
                self._ik = JacobianIK_Numerical(lambda_val)
                self.method = "numerical"
        else:
            self._ik = JacobianIK_Numerical(lambda_val)
            self.method = "numerical"

        print(f"[JacobianIK] 방식: {self.method}")

    def forward_kinematics(self, q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """순방향 기구학"""
        return self._ik.forward_kinematics(q)

    def compute_jacobian(self, q: np.ndarray) -> np.ndarray:
        """Jacobian 계산"""
        return self._ik.compute_jacobian(q)

    def compute(
        self,
        q: np.ndarray,
        delta_pos: np.ndarray,
        delta_rot: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Differential IK 계산

        Args:
            q: 현재 관절 각도 [6] (라디안)
            delta_pos: TCP 위치 델타 [3] (미터)
            delta_rot: TCP 회전 델타 [3] (라디안, 선택적)

        Returns:
            delta_q: 관절 각도 델타 [6] (라디안)
        """
        return self._ik.compute(q, delta_pos, delta_rot)


# =============================================================================
# 테스트
# =============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("  Jacobian IK 테스트")
    print("=" * 60)

    # IK 초기화
    ik = JacobianIK()

    # 홈 포지션 (라디안)
    q_home = np.array([0, 0, np.pi/2, 0, np.pi/2, 0])

    # 순방향 기구학
    pos, rot = ik.forward_kinematics(q_home)
    print(f"\n홈 포지션 관절: {np.degrees(q_home).round(1)} deg")
    print(f"TCP 위치: {pos.round(4)} m")

    # Jacobian 계산
    J = ik.compute_jacobian(q_home)
    print(f"\nJacobian shape: {J.shape}")
    print(f"Jacobian (위치 부분):\n{J[:3, :].round(4)}")

    # IK 테스트: X 방향으로 1cm 이동
    delta_pos = np.array([0.01, 0.0, 0.0])
    delta_q = ik.compute(q_home, delta_pos)

    print(f"\n델타 위치: {delta_pos * 100} cm")
    print(f"델타 관절: {np.degrees(delta_q).round(4)} deg")

    # 검증: 새 위치 계산
    q_new = q_home + delta_q
    pos_new, _ = ik.forward_kinematics(q_new)
    actual_delta = pos_new - pos

    print(f"\n실제 이동량: {actual_delta * 100} cm")
    print(f"오차: {(actual_delta - delta_pos) * 1000} mm")

    print("\n" + "=" * 60)
    print("  테스트 완료!")
    print("=" * 60)
