# E0509 Sim2Real 가이드

Doosan E0509 로봇 + RH-P12-RN-A 그리퍼를 사용한 펜 잡기 강화학습의 Sim2Real 전이 과정을 설명합니다.

## 목차
1. [시스템 구성](#1-시스템-구성)
2. [캘리브레이션](#2-캘리브레이션)
3. [YOLO 펜 감지](#3-yolo-펜-감지)
4. [Sim2Real 실행](#4-sim2real-실행)
5. [주요 파라미터](#5-주요-파라미터)

---

## 1. 시스템 구성

### 하드웨어
- **로봇**: Doosan E0509 (6축 협동로봇)
- **그리퍼**: Robotis RH-P12-RN-A
- **카메라**: Intel RealSense D455F (Eye-to-Hand 구성)
- **PC**: Ubuntu 22.04, ROS2 Humble

### 소프트웨어
- **시뮬레이션**: NVIDIA Isaac Lab
- **강화학습**: RSL-RL (PPO)
- **펜 감지**: YOLOv8 Segmentation
- **로봇 제어**: Doosan ROS2 Driver

### 학습 환경 (V7)
- **Observation (27차원)**:
  - joint_pos (6): 관절 각도 [rad]
  - joint_vel (6): 관절 속도 [rad/s]
  - grasp_pos (3): 그리퍼 grasp point 위치 [m]
  - cap_pos (3): 펜 캡 위치 [m]
  - rel_pos (3): 상대 위치 (cap - grasp) [m]
  - pen_z (3): 펜 방향 벡터 (정규화)
  - dist_to_cap (1): 펜까지 거리 [m]
  - perp_dist (1): 펜 축 수직 거리 [m]
  - gripper_state (1): 그리퍼 상태

- **Action (3차원)**: TCP 델타 [Δx, Δy, Δz]
  - action_scale: 0.03m (3cm)
  - Differential IK (DLS 방식)로 관절 각도 변환

---

## 2. 캘리브레이션

### 2.1 Eye-to-Hand 캘리브레이션

카메라 좌표계 → 로봇 베이스 좌표계 변환 행렬을 구합니다.

#### 준비물
- ArUco 마커 (ID: 0, 크기: 3cm)
- 마커를 그리퍼에 부착

#### 실행
```bash
cd ~/doosan_ws/src/e0509_sim2real/scripts
python3 calibrate_eye_to_hand.py
```

#### 조작
- `c`: 현재 포즈에서 샘플 수집 (로봇을 다양한 위치로 이동하며 30개 이상 수집)
- `s`: 캘리브레이션 계산 및 저장
- `q`: 종료

#### 결과 파일
- `scripts/config/calibration_eye_to_hand.npz`
  - `R_axes`: 회전 행렬 (3x3)
  - `t_offset`: 이동 벡터 (3)
  - `T_cam_to_base`: 변환 행렬 (4x4)

### 2.2 마커-TCP 오프셋 보정

ArUco 마커 중심과 실제 TCP 사이의 오프셋을 보정합니다.

#### 측정 방법
1. `test_pen_detection_calibrated.py` 실행
2. `g` 키로 로봇을 펜 위치로 이동
3. 실제 TCP 위치와 예상 위치의 차이 측정

#### 오프셋 적용 (예시)
```bash
python3 -c "
import numpy as np
path = 'config/calibration_eye_to_hand.npz'
data = dict(np.load(path))

# 마커가 TCP에서 X방향으로 2cm, Z방향으로 4.5cm 떨어진 경우
data['t_offset'][0] -= 0.02  # X 오프셋
data['t_offset'][2] -= 0.045  # Z 오프셋

np.savez(path, **data)
"
```

---

## 3. YOLO 펜 감지

### 3.1 데이터 수집
```bash
python3 collect_yolo_data.py
```
- `s`: 이미지 저장
- 다양한 각도, 조명에서 펜 이미지 수집 (최소 100장 권장)

### 3.2 라벨링
- Roboflow 또는 CVAT 사용
- 펜 마스크 세그멘테이션 라벨링

### 3.3 학습
```bash
yolo segment train data=pen_dataset model=yolov8n-seg.pt epochs=100 imgsz=640
```

### 3.4 모델 경로
- 학습된 모델: `/home/fhekwn549/runs/segment/train/weights/best.pt`

---

## 4. Sim2Real 실행

### 4.1 사전 요구사항
1. Doosan 로봇 전원 ON 및 ROS2 드라이버 실행
2. RealSense 카메라 연결
3. 캘리브레이션 완료

### 4.2 실행
```bash
cd ~/doosan_ws/src/e0509_sim2real/scripts
python3 run_sim2real.py --checkpoint /path/to/model.pt
```

### 4.3 조작
- `g`: Policy 실행 시작
- `h`: Home 위치로 이동
- `q`: 종료

### 4.4 동작 흐름
1. **펜 감지**: YOLO로 펜 캡 위치 감지
2. **좌표 변환**: 카메라 좌표 → 로봇 좌표
3. **Observation 구성**: 관절 각도, grasp point, 펜 위치 등
4. **Policy 추론**: 학습된 모델로 action [Δx, Δy, Δz] 출력
5. **IK 계산**: Differential IK로 관절 각도 변화량 계산
6. **로봇 제어**: MoveJoint로 관절 이동
7. **반복**: 목표 도달까지 2-7 반복

---

## 5. 주요 파라미터

### 5.1 캘리브레이션 (`config/calibration_eye_to_hand.npz`)
| 파라미터 | 설명 | 예시 값 |
|---------|------|--------|
| R_axes | 축 매핑 회전 행렬 | [[0,0,-1],[1,0,0],[0,-1,0]] |
| t_offset | 이동 오프셋 [m] | [0.898, 0.025, 0.274] |

### 5.2 Sim2Real (`run_sim2real.py`)
| 파라미터 | 설명 | 기본값 |
|---------|------|--------|
| gripper_offset_z | 플랜지→grasp point 오프셋 | 0.08m (8cm) |
| action_scale | action → TCP 델타 스케일 | 0.03m (3cm) |
| max_delta_rad | 관절 각도 최대 변화량 | 0.026rad (1.5°) |
| approach_stop_dist | 자동 정지 거리 | 0.02m (2cm) |

### 5.3 IK (`jacobian_ik.py`)
| 파라미터 | 설명 | 기본값 |
|---------|------|--------|
| lambda_val | DLS damping factor | 0.05 |

### 5.4 펜 감지 (`pen_detector_yolo.py`)
| 파라미터 | 설명 | 기본값 |
|---------|------|--------|
| confidence_threshold | YOLO 신뢰도 임계값 | 0.2 |
| pixel_ema_alpha | 좌표 EMA 필터 계수 | 0.15 |

---

## 6. 트러블슈팅

### 6.1 로봇이 펜에서 멀리 멈춤
- **원인**: 거리 계산 오류 또는 캘리브레이션 오프셋 문제
- **해결**:
  1. `t_offset` 조정
  2. `gripper_offset_z` 확인 (플랜지→grasp point 거리)

### 6.2 펜 인식이 깜빡임
- **원인**: YOLO confidence가 경계선
- **해결**: `confidence_threshold` 낮추기 (0.3 → 0.2)

### 6.3 Cap/Tip 좌표 진동
- **원인**: 세그멘테이션 마스크 끝점 불안정
- **해결**: `pixel_ema_alpha` 낮추기 (0.3 → 0.15)

### 6.4 로봇이 진동하며 움직임
- **원인**: 펜 감지 좌표 변동 또는 IK 불안정
- **해결**:
  1. 펜 위치 고정 모드 사용 (g 누르면 고정)
  2. `max_delta_rad` 줄이기

---

## 7. 파일 구조

```
e0509_sim2real/
├── config/
│   ├── config.yaml              # 설정 파일
│   └── calibration_eye_to_hand.npz  # 캘리브레이션 결과
├── scripts/
│   ├── run_sim2real.py          # 메인 실행 스크립트
│   ├── jacobian_ik.py           # Differential IK
│   ├── pen_detector_yolo.py     # YOLO 펜 감지
│   ├── coordinate_transformer.py # 좌표 변환
│   ├── robot_interface.py       # 로봇 인터페이스
│   ├── policy_loader.py         # Policy 로더
│   ├── calibrate_eye_to_hand.py # 캘리브레이션
│   └── test_pen_detection_calibrated.py  # 감지 테스트
└── SIM2REAL_GUIDE.md            # 이 문서
```

---

## 8. 참고

- Isaac Lab V7 학습 환경: `IsaacLab/pen_grasp_rl/envs/e0509_ik_env_v7.py`
- 학습된 모델: `/home/fhekwn549/ikv7/model_99999.pt`
- Doosan ROS2: `doosan-robot2` 패키지
