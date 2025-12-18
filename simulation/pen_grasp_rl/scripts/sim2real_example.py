"""
Sim2Real 기본 예제

강화학습으로 학습된 .pt 파일을 실제 로봇에서 사용하는 방법을 보여줍니다.

=== 핵심 개념 ===
1. Policy (.pt 파일)에서 Actor 네트워크만 추출
2. 로봇 센서에서 관찰값(observation) 수집
3. Actor에 관찰값 입력 → 액션 출력
4. 액션을 로봇에 적용

=== 실제 로봇 적용 시 필요한 것 ===
1. 로봇 제어 인터페이스 (DRFL, ROS, etc.)
2. 관찰값 수집 함수 (센서 데이터 → observation)
3. 액션 적용 함수 (action → 로봇 명령)
"""

import torch
import torch.nn as nn
import numpy as np


# =============================================================================
# 1. Actor 네트워크 정의 (학습 시 사용한 것과 동일해야 함)
# =============================================================================
class ActorNetwork(nn.Module):
    """
    학습된 Actor 네트워크

    구조: observation → hidden layers → action
    """
    def __init__(self, obs_dim: int, action_dim: int, hidden_dims: list = [256, 256, 128]):
        super().__init__()

        # 네트워크 레이어 구성
        layers = []
        prev_dim = obs_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ELU())  # 활성화 함수
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, action_dim))

        self.actor = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """관찰값 → 액션"""
        return self.actor(obs)


# =============================================================================
# 2. Policy 로더
# =============================================================================
class PolicyLoader:
    """
    학습된 .pt 파일에서 Actor를 로드하고 추론하는 클래스
    """
    def __init__(self, checkpoint_path: str, obs_dim: int, action_dim: int,
                 hidden_dims: list = [256, 256, 128], device: str = "cpu"):
        """
        Args:
            checkpoint_path: .pt 파일 경로
            obs_dim: 관찰 공간 차원
            action_dim: 액션 공간 차원
            hidden_dims: 히든 레이어 크기
            device: 실행 디바이스 (cpu 또는 cuda)
        """
        self.device = device

        # Actor 네트워크 생성
        self.actor = ActorNetwork(obs_dim, action_dim, hidden_dims).to(device)

        # 체크포인트 로드
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model_state = checkpoint["model_state_dict"]

        # Actor 가중치만 추출
        actor_state = {}
        for key, value in model_state.items():
            if key.startswith("actor."):
                actor_state[key] = value

        # 가중치 로드
        self.actor.load_state_dict(actor_state)
        self.actor.eval()  # 추론 모드

        print(f"[PolicyLoader] 모델 로드 완료: {checkpoint_path}")
        print(f"[PolicyLoader] 관찰 차원: {obs_dim}, 액션 차원: {action_dim}")

    @torch.no_grad()
    def get_action(self, observation: np.ndarray) -> np.ndarray:
        """
        관찰값으로부터 액션 계산

        Args:
            observation: numpy 배열 (obs_dim,)

        Returns:
            action: numpy 배열 (action_dim,)
        """
        # numpy → tensor
        obs_tensor = torch.FloatTensor(observation).unsqueeze(0).to(self.device)

        # 추론
        action_tensor = self.actor(obs_tensor)

        # tensor → numpy
        action = action_tensor.squeeze(0).cpu().numpy()

        return action


# =============================================================================
# 3. 실제 로봇 인터페이스 (예시 - 실제 구현 필요)
# =============================================================================
class RobotInterface:
    """
    실제 로봇과 통신하는 인터페이스 (예시)

    실제 사용 시 아래 메서드들을 로봇 SDK에 맞게 구현해야 함:
    - Doosan: DRFL 라이브러리
    - ROS: rospy 또는 rclpy
    """
    def __init__(self):
        self.joint_names = ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"]
        self.current_joint_pos = np.zeros(6)
        self.current_joint_vel = np.zeros(6)

    def connect(self, robot_ip: str = "192.168.137.100"):
        """로봇 연결"""
        print(f"[Robot] 연결 시도: {robot_ip}")
        # 실제 구현: DRFL.open_connection(robot_ip) 등
        print("[Robot] 연결 성공 (시뮬레이션)")

    def get_joint_positions(self) -> np.ndarray:
        """현재 관절 위치 읽기 (라디안)"""
        # 실제 구현: DRFL.get_current_posj() 등
        return self.current_joint_pos.copy()

    def get_joint_velocities(self) -> np.ndarray:
        """현재 관절 속도 읽기 (라디안/초)"""
        # 실제 구현: 로봇 SDK에서 속도 읽기
        return self.current_joint_vel.copy()

    def set_joint_position_target(self, target_pos: np.ndarray, duration: float = 0.1):
        """관절 위치 명령 (라디안)"""
        # 실제 구현: DRFL.movej(target_pos, vel, acc) 등
        print(f"[Robot] 관절 명령: {np.rad2deg(target_pos).round(1)} deg")
        self.current_joint_pos = target_pos.copy()

    def disconnect(self):
        """로봇 연결 해제"""
        print("[Robot] 연결 해제")


# =============================================================================
# 4. 관찰값 구성 함수
# =============================================================================
def build_observation(robot: RobotInterface, target_pos: np.ndarray) -> np.ndarray:
    """
    로봇 상태로부터 관찰값 구성

    주의: 학습 시 사용한 관찰값 구조와 동일해야 함!

    E0509 Reach 환경의 관찰값 (33차원):
    - joint_pos: 6 (관절 위치)
    - joint_vel: 6 (관절 속도)
    - ee_pos: 3 (엔드이펙터 위치)
    - ee_quat: 4 (엔드이펙터 방향)
    - target_pos: 3 (목표 위치)
    - actions: 6 (이전 액션)
    - 기타...

    Args:
        robot: 로봇 인터페이스
        target_pos: 목표 위치 (x, y, z)

    Returns:
        observation: 관찰값 배열
    """
    joint_pos = robot.get_joint_positions()
    joint_vel = robot.get_joint_velocities()

    # 실제로는 Forward Kinematics로 계산해야 함
    ee_pos = np.array([0.4, 0.0, 0.3])  # 예시 값
    ee_quat = np.array([1.0, 0.0, 0.0, 0.0])  # 예시 값

    # 관찰값 구성 (학습 환경과 동일해야 함!)
    observation = np.concatenate([
        joint_pos,      # 6
        joint_vel,      # 6
        ee_pos,         # 3
        ee_quat,        # 4
        target_pos,     # 3
        # ... 나머지 관찰값
    ])

    return observation


# =============================================================================
# 5. 메인 제어 루프
# =============================================================================
def main():
    """Sim2Real 메인 루프 예제"""

    print("=" * 60)
    print("Sim2Real 예제")
    print("=" * 60)

    # ----- 설정 -----
    CHECKPOINT_PATH = "/home/fhekwn549/e0509_reach/model_4999.pt"
    OBS_DIM = 33        # 학습 환경의 관찰 차원
    ACTION_DIM = 6      # 학습 환경의 액션 차원
    ACTION_SCALE = 0.1  # 학습 시 사용한 액션 스케일

    # ----- Policy 로드 -----
    print("\n[1] Policy 로드")
    policy = PolicyLoader(
        checkpoint_path=CHECKPOINT_PATH,
        obs_dim=OBS_DIM,
        action_dim=ACTION_DIM,
        hidden_dims=[256, 256, 128],
        device="cpu"  # 실제 로봇 PC에서는 CPU 사용이 일반적
    )

    # ----- 로봇 연결 -----
    print("\n[2] 로봇 연결")
    robot = RobotInterface()
    robot.connect()

    # ----- 목표 설정 -----
    target_pos = np.array([0.4, 0.1, 0.3])  # 목표 위치
    print(f"\n[3] 목표 위치: {target_pos}")

    # ----- 제어 루프 -----
    print("\n[4] 제어 루프 시작")
    print("-" * 40)

    control_freq = 30  # Hz
    dt = 1.0 / control_freq
    max_steps = 100

    for step in range(max_steps):
        # 1. 관찰값 수집
        # 주의: 실제로는 학습 환경과 동일한 구조로 구성해야 함
        observation = np.zeros(OBS_DIM)  # 예시 (실제로는 build_observation 사용)
        observation[:6] = robot.get_joint_positions()
        observation[6:12] = robot.get_joint_velocities()

        # 2. Policy로 액션 계산
        action = policy.get_action(observation)

        # 3. 액션 적용 (delta position control)
        current_pos = robot.get_joint_positions()
        target_joint_pos = current_pos + action * ACTION_SCALE

        # 관절 한계 클램핑 (예시)
        joint_limits_lower = np.deg2rad([-360, -95, -135, -360, -135, -360])
        joint_limits_upper = np.deg2rad([360, 95, 135, 360, 135, 360])
        target_joint_pos = np.clip(target_joint_pos, joint_limits_lower, joint_limits_upper)

        # 4. 로봇에 명령
        robot.set_joint_position_target(target_joint_pos, duration=dt)

        if step % 10 == 0:
            print(f"Step {step}: action = {action.round(3)}")

        # 실제로는 time.sleep(dt) 또는 rate.sleep()

    print("-" * 40)
    print("[5] 제어 완료")

    # ----- 정리 -----
    robot.disconnect()
    print("\n완료!")


# =============================================================================
# 6. 간단한 테스트: Policy 추론만 테스트
# =============================================================================
def test_policy_inference():
    """Policy 추론 테스트 (로봇 연결 없이)"""

    print("=" * 60)
    print("Policy 추론 테스트")
    print("=" * 60)

    CHECKPOINT_PATH = "/home/fhekwn549/e0509_reach/model_4999.pt"

    # Policy 로드
    policy = PolicyLoader(
        checkpoint_path=CHECKPOINT_PATH,
        obs_dim=33,
        action_dim=6,
        hidden_dims=[256, 256, 128],
        device="cpu"
    )

    # 랜덤 관찰값으로 테스트
    print("\n랜덤 관찰값으로 추론 테스트:")
    for i in range(5):
        obs = np.random.randn(33).astype(np.float32)
        action = policy.get_action(obs)
        print(f"  테스트 {i+1}: action = {action.round(4)}")

    print("\n추론 성공!")


if __name__ == "__main__":
    # 추론 테스트만 실행
    test_policy_inference()

    # 전체 예제 (로봇 연결 포함)
    # main()
