# Sim2Real 작업 가이드

## 개요

학습된 Policy를 실제 Doosan E0509 로봇에 적용하여 RealSense로 인식한 펜을 추적하는 Visual Servoing 시스템

```
[시뮬레이션]                    [실제 환경]
Target Tracking 환경    →    RealSense + Doosan 로봇
  (Isaac Lab)                   (run_target_tracking_real.py)
```

---

## 파일 구조

```
pen_grasp_rl/
├── envs/
│   └── target_tracking_env.py    # 학습 환경
├── scripts/
│   ├── train_target_tracking.py  # 학습 스크립트
│   └── run_target_tracking_real.py  # Sim2Real 실행
```

---

## 1단계: Target Tracking 학습

### 환경 설명

| 항목 | 값 |
|------|-----|
| 관찰 공간 | 18차원 (joint_pos, joint_vel, grasp_pos, target_pos) |
| 액션 공간 | 6차원 (6 DOF 팔 delta position) |
| 목표 | grasp point를 랜덤 target으로 이동 |
| 성공 조건 | 거리 < 2cm |
| 예상 학습 시간 | ~20분 (2048 envs, 1000 iterations) |

### Grasp Point Offset

```
RealSense가 인식할 수 있도록 손가락 끝에서 10cm 앞으로 offset

  [Gripper]
     ||
     ||
     \/
  [Fingertip] ----10cm----> [Grasp Point = Target]
                               ↑
                         RealSense가 이 위치의
                         펜캡을 인식
```

### 학습 명령어

```bash
# 1. 가상환경 활성화
source ~/isaacsim_env/bin/activate

# 2. IsaacLab 폴더로 이동
cd ~/IsaacLab

# 3. 학습 실행 (별도 터미널에서!)
python pen_grasp_rl/scripts/train_target_tracking.py \
    --headless \
    --num_envs 2048 \
    --max_iterations 1000

# 모델 저장 위치: ./pen_grasp_rl/logs/target_tracking/
```

### 학습 테스트 (시뮬레이션)

```bash
# 학습된 모델로 시뮬레이션 테스트
python pen_grasp_rl/scripts/play.py \
    --task TargetTracking \
    --checkpoint ./pen_grasp_rl/logs/target_tracking/<run_name>/model_XXX.pt
```

---

## 2단계: Hand-Eye Calibration

RealSense 카메라가 그리퍼에 부착되어 있으므로 Hand-Eye Calibration 필요

### Calibration 방법

1. **ChArUco 보드 준비**
2. **여러 로봇 자세에서 촬영**
3. **OpenCV로 calibration 수행**

### Calibration 스크립트 예시

```python
import cv2
import numpy as np

# ... calibration 코드 ...

# 결과 저장
np.savez('hand_eye_calibration.npz', T_robot_camera=T_robot_camera)
```

### Calibration 파일 형식

```python
# hand_eye_calibration.npz
{
    'T_robot_camera': np.array([  # 4x4 변환 행렬
        [r11, r12, r13, tx],
        [r21, r22, r23, ty],
        [r31, r32, r33, tz],
        [0,   0,   0,   1 ]
    ])
}
```

---

## 3단계: Sim2Real 실행

### 필요 조건

1. **학습된 모델** (.pt 파일)
2. **RealSense 카메라** (그리퍼에 부착, 연결됨)
3. **Hand-Eye Calibration 파일** (.npz)
4. **Doosan 로봇** (네트워크 연결)

### 실행 명령어

```bash
# 기본 실행
python pen_grasp_rl/scripts/run_target_tracking_real.py \
    --checkpoint ./pen_grasp_rl/logs/target_tracking/<run_name>/model_XXX.pt

# 전체 옵션
python pen_grasp_rl/scripts/run_target_tracking_real.py \
    --checkpoint /path/to/model.pt \
    --calibration /path/to/hand_eye_calibration.npz \
    --robot_ip 192.168.137.100 \
    --duration 30.0
```

### 실행 흐름

```
1. Policy 로드 (학습된 .pt 파일)
2. RealSense 초기화
3. 로봇 연결 및 Home 위치로 이동
4. 제어 루프 시작 (30Hz):
   ├─ RealSense로 펜 위치 인식
   ├─ 관찰값 구성 (joint + grasp_pos + target_pos)
   ├─ Policy로 액션 계산
   ├─ 로봇에 관절 명령 전송
   └─ 반복
5. Ctrl+C로 종료 → Home 위치로 복귀
```

---

## 네트워크 설정

### 로봇 IP 설정

```
Doosan 로봇 기본 IP: 192.168.137.100
PC 이더넷 설정:
  - IP: 192.168.137.xxx (100 제외)
  - Subnet: 255.255.255.0
```

### 연결 확인

```bash
# 로봇 ping 테스트
ping 192.168.137.100

# RealSense 연결 확인
realsense-viewer
```

---

## 안전 주의사항

### 로봇 안전

1. **비상 정지 버튼** 위치 확인
2. **작업 영역** 확보 (로봇 팔 반경 내 장애물 제거)
3. **속도 제한** 확인 (코드에서 velocity_limit 설정)
4. **관절 한계** 확인 (joint_limits_lower/upper)

### 처음 실행 시

1. **더미 모드로 먼저 테스트** (로봇 연결 없이)
2. **느린 속도로 시작** (ACTION_SCALE 줄이기)
3. **짧은 duration으로 테스트** (--duration 5.0)

### 코드 내 안전 설정

```python
# run_target_tracking_real.py
ACTION_SCALE = 0.05  # 작게 시작, 점진적 증가
CONTROL_FREQ = 30    # Hz

# 관절 한계 (라디안)
joint_limits_lower = np.radians([-360, -95, -135, -360, -135, -360])
joint_limits_upper = np.radians([360, 95, 135, 360, 135, 360])
```

---

## 트러블슈팅

### RealSense 관련

| 문제 | 해결 |
|------|------|
| 카메라 인식 안됨 | USB 재연결, `realsense-viewer`로 확인 |
| 깊이값 없음 | 최소 거리 확인 (D455: ~20cm) |
| 프레임 드랍 | USB 3.0 포트 사용 |

### 로봇 관련

| 문제 | 해결 |
|------|------|
| 연결 안됨 | IP 설정 확인, 방화벽 확인 |
| 움직임 없음 | 로봇 모드 확인 (Auto 모드) |
| 갑작스런 움직임 | ACTION_SCALE 줄이기 |

### Policy 관련

| 문제 | 해결 |
|------|------|
| 이상한 동작 | 학습 충분히 했는지 확인 |
| 진동 | ACTION_SCALE 줄이기, 제어 주파수 조절 |

---

## 체크리스트

### 학습 전

- [ ] IsaacLab 환경 설정 완료
- [ ] target_tracking_env.py 동기화 확인

### 학습 후

- [ ] 모델 파일 (.pt) 위치 확인
- [ ] 시뮬레이션에서 테스트 완료

### Sim2Real 전

- [ ] RealSense 연결 및 테스트
- [ ] Hand-Eye Calibration 완료
- [ ] 로봇 네트워크 연결 확인
- [ ] 비상 정지 버튼 위치 확인
- [ ] 작업 영역 안전 확인

### Sim2Real 실행

- [ ] 더미 모드로 먼저 테스트
- [ ] 느린 속도 (ACTION_SCALE=0.02)로 시작
- [ ] 짧은 시간 (duration=5.0)으로 테스트
- [ ] 점진적으로 속도/시간 증가

---

## 관련 파일 경로

| 파일 | 경로 |
|------|------|
| Target Tracking 환경 | `pen_grasp_rl/envs/target_tracking_env.py` |
| 학습 스크립트 | `pen_grasp_rl/scripts/train_target_tracking.py` |
| Sim2Real 스크립트 | `pen_grasp_rl/scripts/run_target_tracking_real.py` |
| 학습 로그 | `./pen_grasp_rl/logs/target_tracking/` |

---

## 내일 할 일

1. **Target Tracking 학습** (~20분)
   ```bash
   python pen_grasp_rl/scripts/train_target_tracking.py --headless --num_envs 2048 --max_iterations 1000
   ```

2. **시뮬레이션 테스트**
   - 학습된 모델로 시뮬레이션에서 동작 확인

3. **Hand-Eye Calibration** (필요시)
   - RealSense-그리퍼 변환 행렬 계산

4. **Sim2Real 테스트**
   - 더미 모드 → 실제 로봇 순서로 진행
   - 안전하게 점진적으로 테스트

---

## 참고: Actor 네트워크 구조

```python
# 학습과 동일한 구조로 로드해야 함
ActorNetwork(
    obs_dim=18,
    action_dim=6,
    hidden_dims=[128, 128],
    activation=ELU
)

# Checkpoint에서 actor 가중치 추출
checkpoint = torch.load(path)
actor_state = {k: v for k, v in checkpoint["model_state_dict"].items()
               if k.startswith("actor.")}
```
