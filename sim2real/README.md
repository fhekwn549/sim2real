# Sim2Real 실행 가이드

## 아키텍처

```
[ROS2 환경 - Python 3.10]              [일반/Isaac Sim 환경 - Python 3.11]

┌─────────────────────────┐            ┌─────────────────────────┐
│   bringup.launch.py     │            │   run_sim2real.py       │
│   (로봇 드라이버)        │            │   (Policy 추론)          │
└───────────┬─────────────┘            └───────────┬─────────────┘
            │                                      │
            ▼                                      ▼
┌─────────────────────────┐            ┌─────────────────────────┐
│   sim2real_bridge.py    │◄──────────►│   RobotStateReader      │
│   (ROS2 브릿지)          │  JSON파일   │   RobotCommandWriter    │
└─────────────────────────┘            └─────────────────────────┘
            │                                      │
            ▼                                      │
    /tmp/sim2real_state.json                       │
    /tmp/sim2real_command.json ◄───────────────────┘
```

## 실행 순서

### 터미널 1: 로봇 드라이버 (ROS2)
```bash
source /opt/ros/humble/setup.bash
source ~/doosan_ws/install/setup.bash
ros2 launch e0509_gripper_description bringup.launch.py mode:=real host:=192.168.137.100
```

### 터미널 2: Sim2Real 브릿지 (ROS2)
```bash
source /opt/ros/humble/setup.bash
source ~/doosan_ws/install/setup.bash
cd ~/doosan_ws/src/e0509_gripper_description/scripts/sim2real
python3 sim2real_bridge.py
```

### 터미널 3: Policy 실행 (Isaac Sim 또는 일반 Python)
```bash
# Isaac Sim 환경
source ~/.local/share/ov/pkg/isaac-sim-4.2.0/setup_conda_env.sh
conda activate isaacsim_env

# 또는 일반 환경
# (torch, numpy만 있으면 됨)

cd ~/doosan_ws/src/e0509_gripper_description/scripts/sim2real
python3 run_sim2real.py --checkpoint /home/fhekwn549/simple_move/model_1999.pt
```

## 옵션

```bash
# 고정 타겟 위치 지정 (미터)
python3 run_sim2real.py --checkpoint /path/to/model.pt --fixed_target 0.4 0.0 0.3

# 실행 시간 변경 (초)
python3 run_sim2real.py --checkpoint /path/to/model.pt --duration 30.0

# 제어 주파수 변경 (Hz)
python3 run_sim2real.py --checkpoint /path/to/model.pt --freq 20.0

# RealSense 카메라로 펜 감지
python3 run_sim2real.py --checkpoint /path/to/model.pt --use_camera
```

## 파일 설명

| 파일 | 설명 | Python 버전 |
|------|------|-------------|
| `sim2real_bridge.py` | ROS2 ↔ JSON 파일 브릿지 | 3.10 (ROS2) |
| `run_sim2real.py` | Policy 추론 및 명령 생성 | 3.10/3.11 |
| `policy_loader.py` | PyTorch 모델 로드 | 3.10/3.11 |
| `pen_detector.py` | RealSense 펜 감지 | 3.10/3.11 |
| `coordinate_transformer.py` | 카메라-로봇 좌표 변환 | 3.10/3.11 |

## 통신 파일 형식

### /tmp/sim2real_state.json (로봇 상태)
```json
{
  "timestamp": 1234567890.123,
  "joint_pos_deg": [0.0, 0.0, 90.0, 0.0, 90.0, 0.0],
  "joint_pos_rad": [0.0, 0.0, 1.57, 0.0, 1.57, 0.0],
  "tcp_pos_mm": [400.0, 0.0, 300.0],
  "tcp_pos_m": [0.4, 0.0, 0.3]
}
```

### /tmp/sim2real_command.json (로봇 명령)
```json
{
  "type": "move_joint",
  "target_deg": [0.0, 0.0, 90.0, 0.0, 90.0, 0.0],
  "vel": 60,
  "acc": 60,
  "timestamp": 1234567890.456
}
```

## 트러블슈팅

### 브릿지 연결 타임아웃
- `sim2real_bridge.py`가 실행 중인지 확인
- ROS2 서비스가 준비되었는지 확인: `ros2 service list | grep dsr01`

### Policy 로드 실패
- checkpoint 경로 확인
- env_type이 학습 환경과 일치하는지 확인

### 로봇이 움직이지 않음
- `/tmp/sim2real_command.json` 파일이 생성되는지 확인
- 브릿지 로그에서 명령 수신 확인
