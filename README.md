# Sim2Real

Doosan E0509 + RH-P12-RN-A 그리퍼를 이용한 펜 잡기 Sim2Real 프로젝트

## 개요

Isaac Lab 기반 강화학습으로 로봇이 펜을 적절한 자세로 잡는 방법을 학습하고, 이를 실제 로봇에 적용하는 Sim2Real 파이프라인입니다.

## 프로젝트 구조

```
sim2real/
├── simulation/                      # Isaac Sim 기반 시뮬레이션 & RL
│   ├── pen_grasp_rl/               # 강화학습 환경 및 학습 스크립트
│   │   ├── envs/                   # RL 환경 정의
│   │   ├── scripts/                # train.py, play.py 등
│   │   └── models/                 # USD 모델 파일
│   └── e0509_gripper_isaac/        # Isaac Sim 로봇 설정
│
├── robot/                           # ROS2 기반 실제 로봇 제어
│   └── e0509_gripper_description/  # 로봇 URDF, launch, 제어 스크립트
│       ├── urdf/                   # URDF/Xacro 파일
│       ├── launch/                 # ROS2 launch 파일
│       └── scripts/                # 로봇 제어 스크립트
│
└── sim2real/                        # Sim2Real 브릿지 (공통)
    ├── policy_loader.py            # 학습된 정책 로드
    ├── robot_observation.py        # 로봇 상태 관측
    ├── sim2real_bridge.py          # Sim-Real 브릿지
    └── pen_detector.py             # 펜 감지 (카메라)
```

## 의존성

### 시뮬레이션
- Isaac Sim 4.5+
- Isaac Lab
- RSL-RL

### 실제 로봇
- ROS2 Humble
- Doosan Robot SDK (dsr_control2)
- RH-P12-RN-A 그리퍼 패키지

## 빠른 시작

### 1. 시뮬레이션 학습
```bash
source ~/isaacsim_env/bin/activate
cd sim2real/simulation
python pen_grasp_rl/scripts/train.py --headless --num_envs 4096 --max_iterations 5000
```

### 2. 학습된 정책 테스트 (시뮬레이션)
```bash
python pen_grasp_rl/scripts/play.py --checkpoint ./logs/pen_grasp/model_5000.pt
```

### 3. 실제 로봇 실행
```bash
# 터미널 1: 로봇 bringup
ros2 launch e0509_gripper_description bringup.launch.py mode:=real

# 터미널 2: Sim2Real 실행
cd sim2real/sim2real
python run_sim2real.py --checkpoint ../simulation/logs/pen_grasp/model_5000.pt
```

## 세부 가이드

- [시뮬레이션 README](simulation/README.md)
- [로봇 패키지 README](robot/e0509_gripper_description/README.md)
- [Sim2Real 브릿지 README](sim2real/README.md)

## 하드웨어

- **로봇**: Doosan E0509
- **그리퍼**: Robotis RH-P12-RN-A
- **카메라**: Intel RealSense D455F

## License

Apache-2.0
