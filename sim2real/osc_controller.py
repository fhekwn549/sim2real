#!/usr/bin/env python3
"""
Operational Space Controller (OSC) for Doosan E0509

Isaac Lab의 OSC 구현을 참고하여 실제 로봇용으로 구현.
numpy 기반으로 단일 로봇 제어에 최적화.

Reference:
1. Khatib, O. (1987). "A unified approach for motion and force control of robot manipulators:
   The operational space formulation." IEEE Journal on Robotics and Automation.
2. ETH Zurich Robot Dynamics Lecture Notes by Marco Hutter.

Usage:
    from osc_controller import OperationalSpaceController

    # 컨트롤러 생성
    osc = OperationalSpaceController(
        stiffness=[150, 150, 150, 50, 50, 50],
        damping_ratio=1.0,
    )

    # 제어 루프
    while running:
        state = robot.read_rt_state()

        # 목표 설정 (위치 델타)
        osc.set_target_delta(pos_delta, rot_delta, current_pose)

        # 토크 계산
        torque = osc.compute(
            jacobian=state.jacobian_matrix,
            mass_matrix=state.mass_matrix,
            gravity=state.gravity_torque,
            current_ee_pos=current_pos,
            current_ee_quat=current_quat,
            current_ee_vel=current_vel,
        )

        robot.set_torque(torque)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Tuple
from scipy.spatial.transform import Rotation as R


@dataclass
class OSCConfig:
    """OSC 컨트롤러 설정"""

    # 모션 제어 강성 (위치 x,y,z + 회전 rx,ry,rz)
    stiffness: np.ndarray = field(default_factory=lambda: np.array([150.0, 150.0, 150.0, 50.0, 50.0, 50.0]))

    # 댐핑 비율 (1.0 = 임계 감쇠)
    damping_ratio: float = 1.0

    # 관성 역학 디커플링 사용 여부
    inertial_dynamics_decoupling: bool = True

    # 부분 관성 디커플링 (위치/회전 분리)
    partial_inertial_dynamics_decoupling: bool = False

    # 중력 보상 사용 여부
    gravity_compensation: bool = True

    # 제어 축 (1=활성, 0=비활성)
    motion_control_axes: np.ndarray = field(default_factory=lambda: np.array([1, 1, 1, 1, 1, 1]))


class OperationalSpaceController:
    """
    Operational Space Controller (OSC)

    작업 공간에서의 위치/자세 제어를 관절 토크로 변환합니다.
    """

    def __init__(
        self,
        stiffness: np.ndarray = None,
        damping_ratio: float = 1.0,
        inertial_dynamics_decoupling: bool = True,
        gravity_compensation: bool = True,
    ):
        """
        Args:
            stiffness: (6,) 모션 강성 [pos_x, pos_y, pos_z, rot_x, rot_y, rot_z]
            damping_ratio: 댐핑 비율 (1.0 = 임계 감쇠)
            inertial_dynamics_decoupling: 관성 역학 디커플링 사용 여부
            gravity_compensation: 중력 보상 사용 여부
        """
        # 기본 강성
        if stiffness is None:
            stiffness = np.array([150.0, 150.0, 150.0, 50.0, 50.0, 50.0])
        self.stiffness = np.array(stiffness)

        self.damping_ratio = damping_ratio
        self.inertial_dynamics_decoupling = inertial_dynamics_decoupling
        self.gravity_compensation = gravity_compensation

        # 댐핑 계수 계산: D = 2 * sqrt(K) * damping_ratio
        self.damping = 2.0 * np.sqrt(self.stiffness) * self.damping_ratio

        # P, D 게인 행렬 (대각 행렬)
        self.Kp = np.diag(self.stiffness)
        self.Kd = np.diag(self.damping)

        # 목표 자세
        self.target_pos = None  # (3,) 목표 위치 [m]
        self.target_quat = None  # (4,) 목표 쿼터니언 [w, x, y, z]

        # 작업 공간 질량 행렬 캐시
        self._os_mass_matrix = np.zeros((6, 6))

    def set_stiffness(self, stiffness: np.ndarray):
        """강성 설정"""
        self.stiffness = np.array(stiffness)
        self.damping = 2.0 * np.sqrt(self.stiffness) * self.damping_ratio
        self.Kp = np.diag(self.stiffness)
        self.Kd = np.diag(self.damping)

    def set_damping_ratio(self, damping_ratio: float):
        """댐핑 비율 설정"""
        self.damping_ratio = damping_ratio
        self.damping = 2.0 * np.sqrt(self.stiffness) * self.damping_ratio
        self.Kd = np.diag(self.damping)

    def set_target_pose(self, position: np.ndarray, quaternion: np.ndarray):
        """
        절대 목표 자세 설정

        Args:
            position: (3,) 목표 위치 [x, y, z] (m)
            quaternion: (4,) 목표 쿼터니언 [w, x, y, z]
        """
        self.target_pos = np.array(position)
        self.target_quat = np.array(quaternion)

    def set_target_delta(
        self,
        pos_delta: np.ndarray,
        rot_delta: np.ndarray,
        current_pos: np.ndarray,
        current_quat: np.ndarray,
    ):
        """
        상대 목표 자세 설정 (RL 정책 출력용)

        Args:
            pos_delta: (3,) 위치 변화량 [dx, dy, dz] (m)
            rot_delta: (3,) 회전 변화량 (axis-angle) [rx, ry, rz] (rad)
            current_pos: (3,) 현재 위치 [x, y, z] (m)
            current_quat: (4,) 현재 쿼터니언 [w, x, y, z]
        """
        # 위치 목표 = 현재 + 델타
        self.target_pos = current_pos + pos_delta

        # 회전 목표 = 현재 * 델타 회전
        if np.linalg.norm(rot_delta) > 1e-6:
            # axis-angle -> 쿼터니언
            delta_quat = self._axis_angle_to_quat(rot_delta)
            # 쿼터니언 곱
            self.target_quat = self._quat_multiply(current_quat, delta_quat)
        else:
            self.target_quat = current_quat.copy()

    def compute(
        self,
        jacobian: np.ndarray,
        current_ee_pos: np.ndarray,
        current_ee_quat: np.ndarray,
        current_ee_vel: np.ndarray = None,
        mass_matrix: np.ndarray = None,
        gravity: np.ndarray = None,
        current_joint_pos: np.ndarray = None,
        current_joint_vel: np.ndarray = None,
    ) -> np.ndarray:
        """
        OSC 토크 계산

        Args:
            jacobian: (6, num_dof) 자코비안 행렬
            current_ee_pos: (3,) 현재 EE 위치 [m]
            current_ee_quat: (4,) 현재 EE 쿼터니언 [w, x, y, z]
            current_ee_vel: (6,) 현재 EE 속도 [vx, vy, vz, wx, wy, wz] (m/s, rad/s)
            mass_matrix: (num_dof, num_dof) 관절 공간 질량 행렬
            gravity: (num_dof,) 중력 보상 토크
            current_joint_pos: (num_dof,) 현재 관절 위치 (null-space 제어용)
            current_joint_vel: (num_dof,) 현재 관절 속도 (null-space 제어용)

        Returns:
            (num_dof,) 관절 토크 [Nm]
        """
        if self.target_pos is None or self.target_quat is None:
            raise ValueError("목표 자세가 설정되지 않았습니다. set_target_pose() 또는 set_target_delta()를 호출하세요.")

        num_dof = jacobian.shape[1]

        # 속도 기본값
        if current_ee_vel is None:
            current_ee_vel = np.zeros(6)

        # 1. 자세 에러 계산
        pose_error = self._compute_pose_error(
            current_ee_pos, current_ee_quat,
            self.target_pos, self.target_quat
        )

        # 2. 속도 에러 (목표 속도 = 0)
        velocity_error = -current_ee_vel

        # 3. 원하는 EE 가속도 (스프링-댐퍼 시스템)
        # a_des = Kp * e_pos + Kd * e_vel
        des_ee_acc = self.Kp @ pose_error + self.Kd @ velocity_error

        # 4. 관성 역학 디커플링
        if self.inertial_dynamics_decoupling and mass_matrix is not None:
            # 작업 공간 질량 행렬: Λ = (J * M^-1 * J^T)^-1
            M_inv = np.linalg.inv(mass_matrix)
            try:
                self._os_mass_matrix = np.linalg.inv(jacobian @ M_inv @ jacobian.T)
            except np.linalg.LinAlgError:
                # 특이점 근처에서는 pseudo-inverse 사용
                JMJt = jacobian @ M_inv @ jacobian.T
                self._os_mass_matrix = np.linalg.pinv(JMJt)

            # 작업 공간 힘: F = Λ * a_des
            os_force = self._os_mass_matrix @ des_ee_acc
        else:
            # 관성 디커플링 없이
            os_force = des_ee_acc

        # 5. 관절 토크: τ = J^T * F
        joint_torque = jacobian.T @ os_force

        # 6. 중력 보상
        if self.gravity_compensation and gravity is not None:
            joint_torque = joint_torque + gravity

        return joint_torque

    def _compute_pose_error(
        self,
        current_pos: np.ndarray,
        current_quat: np.ndarray,
        target_pos: np.ndarray,
        target_quat: np.ndarray,
    ) -> np.ndarray:
        """
        자세 에러 계산

        Args:
            current_pos: (3,) 현재 위치
            current_quat: (4,) 현재 쿼터니언 [w, x, y, z]
            target_pos: (3,) 목표 위치
            target_quat: (4,) 목표 쿼터니언 [w, x, y, z]

        Returns:
            (6,) 자세 에러 [pos_error(3), rot_error(3)]
        """
        # 위치 에러
        pos_error = target_pos - current_pos

        # 회전 에러 (axis-angle)
        rot_error = self._compute_rotation_error(current_quat, target_quat)

        return np.concatenate([pos_error, rot_error])

    def _compute_rotation_error(
        self,
        current_quat: np.ndarray,
        target_quat: np.ndarray,
    ) -> np.ndarray:
        """
        회전 에러 계산 (axis-angle)

        Args:
            current_quat: (4,) 현재 쿼터니언 [w, x, y, z]
            target_quat: (4,) 목표 쿼터니언 [w, x, y, z]

        Returns:
            (3,) 회전 에러 (axis-angle)
        """
        # q_error = q_target * q_current^-1
        current_quat_inv = self._quat_inverse(current_quat)
        q_error = self._quat_multiply(target_quat, current_quat_inv)

        # 쿼터니언 -> axis-angle
        # 최단 경로 보장 (w < 0이면 뒤집기)
        if q_error[0] < 0:
            q_error = -q_error

        # axis-angle 변환
        w, x, y, z = q_error
        sin_half_angle = np.sqrt(x*x + y*y + z*z)

        if sin_half_angle < 1e-6:
            # 거의 회전 없음
            return np.zeros(3)

        half_angle = np.arctan2(sin_half_angle, w)
        angle = 2.0 * half_angle

        axis = np.array([x, y, z]) / sin_half_angle

        return axis * angle

    def _quat_multiply(self, q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
        """
        쿼터니언 곱: q1 * q2

        Args:
            q1: (4,) [w, x, y, z]
            q2: (4,) [w, x, y, z]

        Returns:
            (4,) [w, x, y, z]
        """
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2

        w = w1*w2 - x1*x2 - y1*y2 - z1*z2
        x = w1*x2 + x1*w2 + y1*z2 - z1*y2
        y = w1*y2 - x1*z2 + y1*w2 + z1*x2
        z = w1*z2 + x1*y2 - y1*x2 + z1*w2

        return np.array([w, x, y, z])

    def _quat_inverse(self, q: np.ndarray) -> np.ndarray:
        """
        쿼터니언 역원 (정규화된 쿼터니언의 경우 켤레)

        Args:
            q: (4,) [w, x, y, z]

        Returns:
            (4,) [w, -x, -y, -z]
        """
        return np.array([q[0], -q[1], -q[2], -q[3]])

    def _axis_angle_to_quat(self, axis_angle: np.ndarray) -> np.ndarray:
        """
        Axis-angle -> 쿼터니언

        Args:
            axis_angle: (3,) axis * angle

        Returns:
            (4,) [w, x, y, z]
        """
        angle = np.linalg.norm(axis_angle)
        if angle < 1e-6:
            return np.array([1.0, 0.0, 0.0, 0.0])

        axis = axis_angle / angle
        half_angle = angle / 2.0

        w = np.cos(half_angle)
        xyz = axis * np.sin(half_angle)

        return np.array([w, xyz[0], xyz[1], xyz[2]])


class SimpleOSC:
    """
    단순화된 OSC (빠른 테스트용)

    자코비안만 사용하는 간단한 버전.
    관성 디커플링 없이 동작.
    """

    def __init__(
        self,
        stiffness: float = 150.0,
        damping_ratio: float = 1.0,
    ):
        """
        Args:
            stiffness: 위치/회전 공통 강성
            damping_ratio: 댐핑 비율
        """
        self.Kp = stiffness
        self.Kd = 2.0 * np.sqrt(stiffness) * damping_ratio

        self.target_pos = None
        self.target_quat = None

    def set_target_pose(self, position: np.ndarray, quaternion: np.ndarray):
        """목표 자세 설정"""
        self.target_pos = np.array(position)
        self.target_quat = np.array(quaternion)

    def compute(
        self,
        jacobian: np.ndarray,
        current_pos: np.ndarray,
        current_quat: np.ndarray,
        gravity: np.ndarray,
        current_vel: np.ndarray = None,
    ) -> np.ndarray:
        """
        토크 계산 (단순 버전)

        τ = J^T * (Kp * e_pos + Kd * e_vel) + g

        Args:
            jacobian: (6, 6) 자코비안
            current_pos: (3,) 현재 위치
            current_quat: (4,) 현재 쿼터니언
            gravity: (6,) 중력 토크
            current_vel: (6,) 현재 EE 속도

        Returns:
            (6,) 관절 토크
        """
        if self.target_pos is None:
            return gravity.copy()

        if current_vel is None:
            current_vel = np.zeros(6)

        # 위치 에러
        pos_error = self.target_pos - current_pos

        # 회전 에러 (간단히 0으로)
        rot_error = np.zeros(3)
        if self.target_quat is not None:
            rot_error = self._compute_rotation_error(current_quat, self.target_quat)

        pose_error = np.concatenate([pos_error, rot_error])

        # PD 제어
        force = self.Kp * pose_error - self.Kd * current_vel

        # 관절 토크
        torque = jacobian.T @ force + gravity

        return torque

    def _compute_rotation_error(self, current_quat, target_quat):
        """회전 에러 (axis-angle)"""
        # scipy 사용
        r_current = R.from_quat([current_quat[1], current_quat[2], current_quat[3], current_quat[0]])
        r_target = R.from_quat([target_quat[1], target_quat[2], target_quat[3], target_quat[0]])
        r_error = r_target * r_current.inv()
        return r_error.as_rotvec()


# =============================================================================
# 유틸리티 함수
# =============================================================================

def rotation_matrix_to_quat(R_mat: np.ndarray) -> np.ndarray:
    """
    회전 행렬 -> 쿼터니언

    Args:
        R_mat: (3, 3) 회전 행렬

    Returns:
        (4,) [w, x, y, z]
    """
    r = R.from_matrix(R_mat)
    q = r.as_quat()  # [x, y, z, w]
    return np.array([q[3], q[0], q[1], q[2]])  # [w, x, y, z]


def quat_to_rotation_matrix(quat: np.ndarray) -> np.ndarray:
    """
    쿼터니언 -> 회전 행렬

    Args:
        quat: (4,) [w, x, y, z]

    Returns:
        (3, 3) 회전 행렬
    """
    r = R.from_quat([quat[1], quat[2], quat[3], quat[0]])  # [x, y, z, w]
    return r.as_matrix()


def euler_zyz_to_quat(a: float, b: float, c: float) -> np.ndarray:
    """
    ZYZ Euler (Doosan 형식) -> 쿼터니언

    Args:
        a, b, c: ZYZ Euler 각도 (rad)

    Returns:
        (4,) [w, x, y, z]
    """
    r = R.from_euler('ZYZ', [a, b, c])
    q = r.as_quat()  # [x, y, z, w]
    return np.array([q[3], q[0], q[1], q[2]])  # [w, x, y, z]


# =============================================================================
# 테스트
# =============================================================================

def test_osc():
    """OSC 컨트롤러 테스트"""
    print("=" * 60)
    print("OSC 컨트롤러 테스트")
    print("=" * 60)

    # 컨트롤러 생성
    osc = OperationalSpaceController(
        stiffness=np.array([150.0, 150.0, 150.0, 50.0, 50.0, 50.0]),
        damping_ratio=1.0,
        inertial_dynamics_decoupling=True,
        gravity_compensation=True,
    )

    # 가상 데이터
    num_dof = 6
    jacobian = np.eye(6)  # 단순화된 자코비안
    mass_matrix = np.eye(6) * 10.0  # 단순화된 질량 행렬
    gravity = np.array([0, 50, 30, 0, 0, 0])  # 예시 중력 토크

    current_pos = np.array([0.4, 0.0, 0.3])
    current_quat = np.array([1.0, 0.0, 0.0, 0.0])  # 단위 쿼터니언
    current_vel = np.zeros(6)

    # 목표 설정
    target_pos = np.array([0.5, 0.1, 0.35])
    target_quat = np.array([1.0, 0.0, 0.0, 0.0])
    osc.set_target_pose(target_pos, target_quat)

    # 토크 계산
    torque = osc.compute(
        jacobian=jacobian,
        current_ee_pos=current_pos,
        current_ee_quat=current_quat,
        current_ee_vel=current_vel,
        mass_matrix=mass_matrix,
        gravity=gravity,
    )

    print(f"\n현재 위치: {current_pos}")
    print(f"목표 위치: {target_pos}")
    print(f"위치 에러: {target_pos - current_pos}")
    print(f"\n계산된 토크: {np.round(torque, 2)}")

    # 델타 명령 테스트
    print("\n--- 델타 명령 테스트 ---")
    osc2 = OperationalSpaceController()

    pos_delta = np.array([0.01, 0.02, -0.01])  # 1cm, 2cm, -1cm
    rot_delta = np.array([0.0, 0.0, 0.1])  # z축 회전

    osc2.set_target_delta(pos_delta, rot_delta, current_pos, current_quat)
    print(f"위치 델타: {pos_delta}")
    print(f"회전 델타 (axis-angle): {rot_delta}")
    print(f"계산된 목표 위치: {osc2.target_pos}")

    torque2 = osc2.compute(
        jacobian=jacobian,
        current_ee_pos=current_pos,
        current_ee_quat=current_quat,
        current_ee_vel=current_vel,
        mass_matrix=mass_matrix,
        gravity=gravity,
    )
    print(f"계산된 토크: {np.round(torque2, 2)}")

    print("\n" + "=" * 60)
    print("테스트 완료!")
    print("=" * 60)


if __name__ == "__main__":
    test_osc()
