# Pen Grasp RL 학습 기록 (V2 환경)

## 개요

V2 환경은 Isaac Lab의 **reach 예제**를 기반으로 재설계된 단순화된 환경입니다.

### V1 vs V2 비교

| 항목 | V1 (기존) | V2 (신규) |
|------|-----------|-----------|
| 보상 개수 | 7개 | **4개** |
| 관찰 차원 | 36 | **27** |
| 구조 | 직접 구현 | **reach 예제 기반** |
| 목표 | 여러 조건 혼합 | **위치 + 방향만** |

### 학습 목표 (2가지만)

1. **위치**: `gripper_grasp_point` → `pen_cap_point` 거리 최소화
2. **방향**: `gripper_z` · `pen_z` → -1 (반대 방향 정렬)

---

## 환경 구조

### 관찰 공간 (27차원)

| 관찰 | 차원 | 설명 |
|------|------|------|
| `joint_pos` | 6 | 팔 관절 위치 |
| `joint_vel` | 6 | 팔 관절 속도 |
| `grasp_point` | 3 | 그리퍼 잡기 포인트 위치 |
| `pen_cap` | 3 | 펜 캡 위치 |
| `relative_pos` | 3 | 그리퍼→캡 상대 위치 (핵심!) |
| `pen_z_axis` | 3 | 펜 Z축 방향 |
| `gripper_z_axis` | 3 | 그리퍼 Z축 방향 |

### 보상 함수 (4개)

| 보상 | weight | 형태 | 설명 |
|------|--------|------|------|
| `position_error` | -0.5 | L2 거리 | 거리 페널티 |
| `position_fine` | +1.0 | 1 - tanh(d/0.1) | 정밀 위치 보상 |
| `orientation_error` | -0.3 | 1 + dot | 방향 오차 페널티 |
| `action_rate` | -0.001 | action² | 행동 페널티 |

### 보상 형태 (reach 예제 스타일)

**위치 보상:**
```python
# L2 거리 (페널티)
distance = ||grasp_pos - cap_pos||
position_error = distance  # weight: -0.5

# tanh 커널 (보상)
position_fine = 1 - tanh(distance / 0.1)  # weight: +1.0
```

**방향 보상:**
```python
# dot product 오차
dot = pen_z · gripper_z
orientation_error = 1 + dot  # dot=-1이면 0, dot=+1이면 2
# weight: -0.3
```

---

## 학습 실행 방법

```bash
source ~/isaacsim_env/bin/activate
cd ~/IsaacLab

# V2 환경으로 학습 (기본값)
python pen_grasp_rl/scripts/train.py --num_envs 64

# V1 환경으로 학습하려면
python pen_grasp_rl/scripts/train.py --num_envs 64 --env_version v1
```

---

## TensorBoard 확인 지표

```bash
tensorboard --logdir=~/IsaacLab/logs/pen_grasp
# 브라우저에서 http://localhost:6006 접속
```

| 지표 | 좋은 신호 |
|------|-----------|
| `Episode_Reward/position_error` | 📉 감소 (거리 줄어듦) |
| `Episode_Reward/position_fine` | 📈 증가 (가까워짐) |
| `Episode_Reward/orientation_error` | 📉 감소 (정렬됨) |
| `Train/mean_reward` | 📈 증가 |

---

## 학습 기록

### 2025-12-16 V2 환경 생성

#### 배경
- V1 환경에서 보상 함수를 여러 번 수정했으나 학습이 잘 안됨
- 측면에서 접근하여 펜과 충돌하는 문제 발생
- 기존 예제 없이 직접 구현한 것이 문제의 원인

#### 해결책
- Isaac Lab의 **reach 예제** 구조를 기반으로 재설계
- 보상 함수를 **4개로 단순화** (기존 7개)
- 검증된 보상 형태 사용 (L2, tanh)

#### 변경 사항

**1. 새 파일 생성**
- `pen_grasp_rl/envs/pen_grasp_env_v2.py`

**2. 보상 구조 단순화**
```python
# V1: 7개 보상 (복잡)
distance_to_cap, z_axis_alignment, base_orientation,
approach_from_above, alignment_success, floor_collision, action_rate

# V2: 4개 보상 (단순)
position_error, position_fine, orientation_error, action_rate
```

**3. 관찰 공간 정리**
- 36차원 → 27차원
- 불필요한 관찰 제거
- 핵심 관찰만 유지 (relative_pos, z_axis 등)

**4. train.py 수정**
- `--env_version` 인자 추가
- 기본값: v2

#### 다음 단계
- [x] V2 환경으로 학습 실행
- [x] position_fine 보상 증가 확인
- [x] orientation_error 감소 확인
- [x] play.py로 동작 확인

---

### 2025-12-16 첫 번째 학습 결과 (5,800 iteration)

#### 학습 결과

**로그 위치**: `/home/fhekwn549/pen_grasp/events.out.tfevents.*.21.0`
**모델**: `model_5800.pt`

| 지표 | 초기 | 최종 | 변화 |
|------|------|------|------|
| position_fine | 0.006 | **0.72** | +11,254% ✅ |
| position_error | -0.08 | -0.014 | +82% ✅ |
| orientation_error | -0.112 | **-0.016** | +86% ✅ |
| mean_reward | -2.03 | **6.90** | +440% ✅ |
| noise_std | 0.31 | **0.32** | 안정적 ✅ |
| value_loss | 0.006 | **0.000** | 수렴 ✅ |
| time_out | 40% | 100% | (성공 종료 없음) |

#### 분석

**위치 학습 ✅**
- position_fine = 0.72 → 약 **3cm** 거리까지 접근
- V1과 비슷한 수준 달성

**방향 학습 ✅ (V1 대비 큰 개선!)**
- orientation_error = -0.016 → dot ≈ **-0.95**
- 약 18도 오차 (거의 완벽한 정렬)
- V1에서는 전혀 학습 안 됐던 부분!

**학습 안정성 ✅**
- noise_std: 0.31 → 0.32 (V1: 0.31 → 0.73)
- V1은 노이즈 2배 증가 (불안정), V2는 안정적

#### play.py 확인 결과

- ✅ 펜 캡 쪽으로 접근함
- ⚠️ 측면에서 충돌하는 경우 종종 있음
- ❌ 위에서 내려오는 동작이 거의 없음

#### 결론
- 보상 구조는 잘 작동함
- 초기 자세 때문에 측면 접근이 쉬워서 그쪽으로 학습된 것으로 추정

---

### 2025-12-16 초기 자세 변경 (직립 자세)

#### 문제점
- 기존 초기 자세 (joint_3=1.57, joint_5=1.57)에서 그리퍼가 이미 아래를 향함
- 펜에 측면에서 접근하기 쉬운 자세
- 위에서 내려오는 동작을 학습할 동기 부족

#### 변경 사항

**로봇 초기 자세 변경 (`pen_grasp_env_v2.py`)**

| 관절 | 이전 | 변경 |
|------|------|------|
| joint_1 | 0.0 | 0.0 |
| joint_2 | 0.0 | 0.0 |
| joint_3 | **1.57** | **0.0** |
| joint_4 | 0.0 | 0.0 |
| joint_5 | **1.57** | **0.0** |
| joint_6 | 0.0 | 0.0 |

```
이전 자세:              변경된 자세:
   ┌─[그리퍼↓]              │
   │                       │
   └─[베이스]              [그리퍼↑]
                           │
                          [베이스]
```

#### 기대 효과
- 그리퍼가 위를 향한 상태에서 시작
- 펜(앞쪽 공중)에 접근하려면 팔을 뻗어서 내려가야 함
- 자연스럽게 **위에서 접근하는 동작** 학습 유도

#### 추가 변경: play.py
- `--env_version` 인자 추가 (v1/v2 선택 가능)

#### 다음 단계
- [x] 새 초기 자세로 학습 실행
- [ ] 위에서 접근하는 동작 학습 확인
- [ ] 측면 충돌 감소 확인

---

### 2025-12-16 두 번째 학습 결과 (3,000 iteration, 직립 자세)

#### 학습 결과

| 지표 | 초기 | 최종 | 비교 (이전) |
|------|------|------|-------------|
| position_fine | 0.0001 | **0.69** | 0.72 (비슷) |
| orientation_error | -0.116 | **-0.015** | -0.016 (비슷) |
| mean_reward | -3.25 | **6.41** | 6.90 (비슷) |
| noise_std | 0.31 | **0.37** | 0.32 |

#### play.py 확인 결과

- ✅ 위에서 내려오긴 함 (초기 자세 효과)
- ❌ **joint_2만 움직여서 바로 아래로 꼬라박음**
- ❌ z축 정렬, 펜 캡 위치 찾기 예상보다 안됨

#### 문제 분석

1. **초기 거리 문제**
   - 직립 자세에서 그리퍼 위치 ≈ (0, 0, ~0.8m)
   - 펜 위치: (0.5, 0, 0.3)
   - 거리 약 60-70cm로 탐험이 어려움

2. **Local Minimum**
   - joint_2만 움직이면 어느 정도 보상 받음
   - 더 복잡한 동작(joint_1 회전 등) 안 배움

---

### 2025-12-16 중간 자세 + 위에서 접근 보상 추가

#### 변경 사항

**1. 중간 자세로 시작 (`pen_grasp_env_v2.py`)**

| 관절 | 이전 | 변경 | 설명 |
|------|------|------|------|
| joint_1 | 0.0 | 0.0 | |
| joint_2 | 0.0 | **0.3** | 약간 앞으로 |
| joint_3 | 0.0 | **0.5** | 약간 구부림 |
| joint_4 | 0.0 | 0.0 | |
| joint_5 | 0.0 | **0.5** | 약간 아래로 |
| joint_6 | 0.0 | 0.0 | |

→ 펜에 더 가까우면서도 측면 접근은 어렵게

**2. 위에서 접근 보상 추가**

```python
def approach_from_above_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    """그리퍼가 캡보다 위에 있을 때 보상"""
    grasp_pos = get_grasp_point(robot)
    cap_pos = get_pen_cap_pos(pen)

    # 그리퍼 Z > 캡 Z 이면 보상
    height_diff = grasp_pos[:, 2] - cap_pos[:, 2]
    return torch.clamp(height_diff, min=0.0, max=0.1) * 10.0  # 최대 1.0
```

```python
# RewardsCfg
approach_bonus = RewTerm(func=approach_from_above_reward, weight=0.3)
```

#### 현재 보상 구조 (5개)

| 보상 | weight | 설명 |
|------|--------|------|
| position_error | -0.5 | 거리 페널티 |
| position_fine | +1.0 | 정밀 위치 보상 |
| orientation_error | -0.3 | 방향 오차 페널티 |
| **approach_bonus** | **+0.3** | **위에서 접근 보상 (신규)** |
| action_rate | -0.001 | 행동 페널티 |

#### 기대 효과

1. 중간 자세 → 펜에 가까워서 탐험 쉬움
2. approach_bonus → 위에서 접근하는 동작 유도
3. 측면 접근 시 approach_bonus 못 받음

#### 다음 단계
- [ ] 새 설정으로 학습 실행
- [ ] approach_bonus 보상 증가 확인
- [ ] 위에서 접근하는 동작 학습 확인

---
