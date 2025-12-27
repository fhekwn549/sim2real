# Sim2Real

Doosan E0509 + RH-P12-RN-A 그리퍼를 이용한 펜 잡기 Sim2Real 프로젝트

## 개요

Isaac Lab 기반 강화학습으로 로봇이 펜을 적절한 자세로 잡는 방법을 학습하고, 이를 실제 로봇에 적용하는 Sim2Real 파이프라인입니다.

## 관련 레포지토리

| 레포지토리 | 설명 |
|-----------|------|
| **[CoWriteBotRL](https://github.com/fhekwn549/CoWriteBotRL)** | Isaac Lab 기반 강화학습 환경 및 학습 스크립트 |
| **[e0509_gripper_description](https://github.com/fhekwn549/e0509_gripper_description)** | ROS2 로봇 패키지 (URDF, launch, 그리퍼 제어) |

## 프로젝트 구조

```
sim2real/
└── sim2real/                        # Sim2Real 실행 코드
    ├── run_sim2real.py              # 메인 실행 스크립트
    ├── jacobian_ik.py               # Differential IK (DLS 방식)
    ├── pen_detector_yolo.py         # YOLO 펜 감지
    ├── coordinate_transformer.py    # 카메라 → 로봇 좌표 변환
    ├── robot_interface.py           # 로봇 인터페이스 (ROS2)
    ├── gripper_interface.py         # 그리퍼 인터페이스
    ├── policy_loader.py             # 학습된 정책 로드
    ├── config/                      # 캘리브레이션 설정
    ├── SIM2REAL_GUIDE.md            # 상세 가이드
    └── deprecated/                  # 이전 버전 파일들
```

## 의존성

### 시뮬레이션 (CoWriteBotRL)
- Isaac Sim 4.5+
- Isaac Lab
- RSL-RL

### 실제 로봇 (e0509_gripper_description)
- ROS2 Humble
- Doosan Robot SDK (dsr_control2)
- RH-P12-RN-A 그리퍼 패키지

### 펜 감지
- YOLOv8 Segmentation
- Intel RealSense SDK

## 빠른 시작

### 1. 시뮬레이션 학습 (CoWriteBotRL)
```bash
source ~/isaacsim_env/bin/activate
cd ~/CoWriteBotRL
python pen_grasp_rl/scripts/train_v7.py --headless --num_envs 4096 --max_iterations 100000
```

### 2. 학습된 정책 테스트 (시뮬레이션)
```bash
python pen_grasp_rl/scripts/play_v7.py --checkpoint ~/ikv7/model_99999.pt
```

### 3. 실제 로봇 실행
```bash
# 터미널 1: 로봇 bringup
ros2 launch e0509_gripper_description bringup.launch.py mode:=real host:=192.168.137.100

# 터미널 2: Sim2Real 실행
cd ~/sim2real/sim2real
python run_sim2real.py --checkpoint ~/ikv7/model_99999.pt
```

## 상세 가이드

- [Sim2Real 가이드](sim2real/SIM2REAL_GUIDE.md) - 캘리브레이션, 펜 감지, 실행 방법

## 하드웨어

- **로봇**: Doosan E0509
- **그리퍼**: Robotis RH-P12-RN-A
- **카메라**: Intel RealSense D455F (Eye-to-Hand 구성)

## License

Apache-2.0
