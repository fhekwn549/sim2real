# Deprecated Files

이 폴더에는 더 이상 사용하지 않는 이전 버전의 파일들이 있습니다.

## 아키텍처 변경 (2024.12)

### 이전 방식 (JSON 브릿지)
```
[ROS2 환경 - Python 3.10]              [Isaac Sim 환경 - Python 3.11]

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
            │                                      
            ▼                                      
    /tmp/sim2real_state.json                       
    /tmp/sim2real_command.json
```

- Python 버전 분리 필요 (Isaac Sim이 Python 3.11 사용)
- JSON 파일을 통한 프로세스 간 통신
- 두 개의 프로세스 실행 필요

### 현재 방식 (직접 ROS2 통신)
```
┌─────────────────────────────────────────┐
│         run_sim2real.py                 │
│         (Python 3.10, ROS2 환경)         │
│                                         │
│  ┌─────────────┐    ┌────────────────┐  │
│  │ Policy 추론  │    │ RobotInterface │  │
│  │ (PyTorch)   │───►│ (rclpy 직접)   │  │
│  └─────────────┘    └───────┬────────┘  │
└─────────────────────────────┼───────────┘
                              │ ROS2 서비스 직접 호출
                              ▼
                    ┌─────────────────────┐
                    │   Doosan Driver     │
                    │   (bringup.launch)  │
                    └─────────────────────┘
```

- 단일 프로세스에서 실행
- rclpy로 ROS2 서비스 직접 호출
- Isaac Sim 없이 PyTorch만 사용하므로 Python 버전 충돌 없음

## Deprecated 파일 목록

### 아키텍처 관련
- `sim2real_bridge.py`: ROS2 ↔ JSON 파일 브릿지 (더 이상 사용 안 함)
- `robot_observation.py`: 이전 observation 구성 방식 (run_sim2real.py에 통합됨)

### 캘리브레이션 관련
- `auto_hand_eye_calibration.py`: 자동 hand-eye 캘리브레이션 (calibrate_eye_to_hand.py로 대체)
- `hand_eye_calibration.py`: 이전 캘리브레이션 방식
- `manual_hand_eye_calibration.py`: 수동 캘리브레이션
- `verify_calibration.py`: 캘리브레이션 검증 (test_pen_detection_calibrated.py로 대체)
- `generate_checkerboard.py`: 체커보드 생성 (ArUco 마커 사용으로 불필요)

### 펜 감지/제어 관련
- `pen_detector.py`: HSV 기반 펜 감지 (pen_detector_yolo.py로 대체)
- `pen_grasp_controller.py`: 이전 펜 잡기 컨트롤러 (run_sim2real.py로 통합)
- `run_pen_tracking.py`: 이전 tracking 스크립트 (run_sim2real.py로 대체)
- `action_processor.py`: 이전 action 처리 방식 (run_sim2real.py로 통합)

### 테스트 관련
- `test_ik_move.py`: IK 이동 테스트 (jacobian_ik.py에서 직접 테스트)
- `test_observation.py`: observation 테스트 (run_sim2real.py에서 직접 확인)
