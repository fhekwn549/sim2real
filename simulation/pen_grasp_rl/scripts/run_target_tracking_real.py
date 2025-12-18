"""
Target Tracking Sim2Real 실행 스크립트

학습된 Policy + RealSense로 실제 로봇에서 펜 추적

=== 사용법 ===
python run_target_tracking_real.py --checkpoint /path/to/model.pt

=== 필요한 것 ===
1. 학습된 모델 (.pt 파일)
2. RealSense 카메라 (그리퍼에 부착)
3. Hand-eye calibration 파일 (.npz)
4. Doosan 로봇 연결
"""

import argparse
import numpy as np
import torch
import torch.nn as nn
import time
import math

# =============================================================================
# 설정
# =============================================================================
HOME_JOINT_DEG = [0, 0, 90, 0, 90, 0]  # Home 자세 (도)
HOME_JOINT_RAD = [math.radians(d) for d in HOME_JOINT_DEG]

ACTION_SCALE = 0.05  # 학습 시 사용한 값과 동일해야 함
CONTROL_FREQ = 30    # Hz


# =============================================================================
# Actor 네트워크 (학습과 동일한 구조)
# =============================================================================
class ActorNetwork(nn.Module):
    def __init__(self, obs_dim: int = 18, action_dim: int = 6, hidden_dims: list = [128, 128]):
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
    def __init__(self, checkpoint_path: str, device: str = "cpu"):
        self.device = device
        self.actor = ActorNetwork(obs_dim=18, action_dim=6, hidden_dims=[128, 128]).to(device)

        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model_state = checkpoint["model_state_dict"]

        actor_state = {k: v for k, v in model_state.items() if k.startswith("actor.")}
        self.actor.load_state_dict(actor_state)
        self.actor.eval()

        print(f"[Policy] 모델 로드: {checkpoint_path}")

    @torch.no_grad()
    def get_action(self, observation: np.ndarray) -> np.ndarray:
        obs_tensor = torch.FloatTensor(observation).unsqueeze(0).to(self.device)
        action_tensor = self.actor(obs_tensor)
        return action_tensor.squeeze(0).cpu().numpy()


# =============================================================================
# RealSense 펜 인식
# =============================================================================
class PenDetector:
    """RealSense로 펜 위치 인식"""

    def __init__(self, calibration_file: str = None):
        try:
            import pyrealsense2 as rs
            self.rs = rs
        except ImportError:
            print("[Warning] pyrealsense2 not installed. Using dummy mode.")
            self.rs = None
            self.dummy_mode = True
            return

        self.dummy_mode = False
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

        # Hand-eye calibration 로드
        self.T_robot_camera = np.eye(4)
        if calibration_file:
            try:
                calib_data = np.load(calibration_file)
                self.T_robot_camera = calib_data['T_robot_camera']
                print(f"[Calibration] 로드: {calibration_file}")
            except:
                print(f"[Warning] Calibration 파일 로드 실패: {calibration_file}")

        self.pipeline.start(config)
        self.align = rs.align(rs.stream.color)
        print("[RealSense] 초기화 완료")

    def get_pen_position(self) -> np.ndarray:
        """펜 위치 반환 (로봇 좌표계)"""
        if self.dummy_mode:
            # 더미 모드: 고정 위치 반환
            return np.array([0.35, 0.0, 0.35])

        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)

        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        if not depth_frame or not color_frame:
            return None

        # 이미지 중앙 영역에서 가장 가까운 점 찾기
        depth_image = np.asanyarray(depth_frame.get_data())
        h, w = depth_image.shape

        # 중앙 ROI
        roi = depth_image[h//2-50:h//2+50, w//2-50:w//2+50]
        valid_depths = roi[roi > 0]

        if len(valid_depths) == 0:
            return None

        min_depth = np.min(valid_depths)

        # 가장 가까운 점의 픽셀 좌표
        cy, cx = np.where(depth_image == min_depth)
        if len(cx) == 0:
            cx, cy = w // 2, h // 2
        else:
            cx, cy = cx[0], cy[0]

        # 픽셀 → 3D 좌표 (카메라 좌표계)
        depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics
        point_camera = self.rs.rs2_deproject_pixel_to_point(
            depth_intrin, [cx, cy], min_depth * 0.001  # mm → m
        )

        # 카메라 좌표 → 로봇 좌표
        point_camera_homo = np.append(point_camera, 1)
        point_robot = self.T_robot_camera @ point_camera_homo

        return point_robot[:3]

    def close(self):
        if not self.dummy_mode:
            self.pipeline.stop()


# =============================================================================
# Doosan 로봇 인터페이스
# =============================================================================
class DoosanRobot:
    """Doosan 로봇 제어 (DRFL)"""

    def __init__(self, ip: str = "192.168.137.100"):
        self.ip = ip
        self.connected = False
        self.current_joint_pos = np.array(HOME_JOINT_RAD)
        self.current_joint_vel = np.zeros(6)

        # DRFL import 시도
        try:
            # import DRFL  # 실제 Doosan SDK
            self.drfl = None  # 더미 모드
            print(f"[Robot] 더미 모드 (DRFL 없음)")
        except ImportError:
            self.drfl = None
            print(f"[Robot] 더미 모드 (DRFL 없음)")

    def connect(self):
        if self.drfl:
            # self.drfl.open_connection(self.ip)
            pass
        self.connected = True
        print(f"[Robot] 연결됨: {self.ip}")

    def get_joint_positions(self) -> np.ndarray:
        """현재 관절 위치 (라디안)"""
        if self.drfl:
            # pos_deg = self.drfl.get_current_posj()
            # return np.radians(pos_deg)
            pass
        return self.current_joint_pos.copy()

    def get_joint_velocities(self) -> np.ndarray:
        """현재 관절 속도"""
        return self.current_joint_vel.copy()

    def get_tcp_position(self) -> np.ndarray:
        """현재 TCP 위치 (Forward Kinematics)"""
        # 실제로는 FK 계산 필요
        # 간단한 추정값 사용
        return np.array([0.3, 0.0, 0.4])

    def set_joint_position(self, target_rad: np.ndarray, duration: float = 0.1):
        """관절 위치 명령"""
        target_deg = np.degrees(target_rad)

        if self.drfl:
            # self.drfl.movej(target_deg.tolist(), vel=30, acc=30)
            pass
        else:
            # 더미 모드
            self.current_joint_pos = target_rad.copy()

        print(f"[Robot] 관절 명령: {target_deg.round(1)} deg")

    def move_to_home(self):
        """Home 위치로 이동"""
        print("[Robot] Home 위치로 이동...")
        self.set_joint_position(np.array(HOME_JOINT_RAD))

    def disconnect(self):
        if self.drfl:
            # self.drfl.close_connection()
            pass
        self.connected = False
        print("[Robot] 연결 해제")


# =============================================================================
# Observation 구성
# =============================================================================
def build_observation(robot: DoosanRobot, target_pos: np.ndarray) -> np.ndarray:
    """
    로봇 상태 + target으로 관찰값 구성

    학습 환경과 동일한 구조:
    - joint_pos: 6
    - joint_vel: 6
    - grasp_pos: 3 (로컬, 로봇 base 기준)
    - target_pos: 3 (로컬, 로봇 base 기준)
    """
    joint_pos = robot.get_joint_positions()
    joint_vel = robot.get_joint_velocities()
    grasp_pos = robot.get_tcp_position()  # 실제로는 손가락 끝 + 10cm

    observation = np.concatenate([
        joint_pos,   # 6
        joint_vel,   # 6
        grasp_pos,   # 3
        target_pos,  # 3
    ]).astype(np.float32)

    return observation


# =============================================================================
# 메인 제어 루프
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="Target Tracking Sim2Real")
    parser.add_argument("--checkpoint", type=str, required=True, help="모델 경로")
    parser.add_argument("--calibration", type=str, default=None, help="Hand-eye calibration 파일")
    parser.add_argument("--robot_ip", type=str, default="192.168.137.100", help="로봇 IP")
    parser.add_argument("--duration", type=float, default=30.0, help="실행 시간 (초)")
    args = parser.parse_args()

    print("=" * 60)
    print("Target Tracking Sim2Real")
    print("=" * 60)

    # Policy 로드
    policy = PolicyLoader(args.checkpoint)

    # RealSense 초기화
    detector = PenDetector(args.calibration)

    # 로봇 연결
    robot = DoosanRobot(args.robot_ip)
    robot.connect()

    # Home 위치로 이동
    robot.move_to_home()
    time.sleep(2.0)

    print("=" * 60)
    print("제어 시작 (Ctrl+C로 종료)")
    print("=" * 60)

    # 관절 한계 (라디안)
    joint_limits_lower = np.radians([-360, -95, -135, -360, -135, -360])
    joint_limits_upper = np.radians([360, 95, 135, 360, 135, 360])

    dt = 1.0 / CONTROL_FREQ
    start_time = time.time()

    try:
        step = 0
        while time.time() - start_time < args.duration:
            loop_start = time.time()

            # 1. 펜 위치 인식
            target_pos = detector.get_pen_position()
            if target_pos is None:
                print("[Warning] 펜 인식 실패, 이전 위치 유지")
                time.sleep(dt)
                continue

            # 2. 관찰값 구성
            obs = build_observation(robot, target_pos)

            # 3. Policy로 액션 계산
            action = policy.get_action(obs)

            # 4. 액션 적용
            current_pos = robot.get_joint_positions()
            target_joint = current_pos + action * ACTION_SCALE

            # 관절 한계 클램핑
            target_joint = np.clip(target_joint, joint_limits_lower, joint_limits_upper)

            # 5. 로봇 명령
            robot.set_joint_position(target_joint, duration=dt)

            # 로그
            if step % 30 == 0:
                distance = np.linalg.norm(robot.get_tcp_position() - target_pos)
                print(f"Step {step}: target={target_pos.round(3)}, distance={distance:.3f}m")

            step += 1

            # 제어 주기 유지
            elapsed = time.time() - loop_start
            if elapsed < dt:
                time.sleep(dt - elapsed)

    except KeyboardInterrupt:
        print("\n[중단됨]")

    # 정리
    print("=" * 60)
    print("종료 중...")
    robot.move_to_home()
    time.sleep(2.0)
    robot.disconnect()
    detector.close()
    print("완료!")


if __name__ == "__main__":
    main()
