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

- `sim2real_bridge.py`: ROS2 ↔ JSON 파일 브릿지 (더 이상 사용 안 함)
- `robot_observation.py`: 이전 observation 구성 방식 (run_sim2real.py에 통합됨)
