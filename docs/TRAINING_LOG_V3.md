# Pen Grasp RL V3 - Curriculum Learning

## 개요

V2 환경에서 37,000 iteration 학습 결과, 로그 상의 수치는 좋아 보였으나 실제 동작에서는:
- 펜캡 마커 부분에 가까이 가는 경우가 소수에 불과
- Z축 정렬이 거의 안 됨

이 문제를 해결하기 위해 **Curriculum Learning**을 도입한 V3 환경 개발.

---

## V2 문제점 분석

### 1. 보상 함수가 너무 관대함

```python
# V2: distance_ee_cap_reward
return torch.exp(-distance * 10.0)
# 거리 10cm → 보상 0.37
# 거리 5cm → 보상 0.61
```

로봇이 정확히 캡에 도달하지 않아도 적당히 가까우면 높은 보상을 받음.

### 2. 로그 수치의 함정

- `Position Fine = 0.71` → `exp(-distance * 10) = 0.71` → 평균 거리 약 3.4cm
- 하지만 이건 **평균값**. 일부 환경만 가깝고, 나머지는 멀리서 머무를 수 있음.

### 3. 정렬 보상이 거리와 분리됨

```python
# V2: z_axis_alignment_reward
distance_factor = torch.clamp(1.0 - distance_to_cap / 0.30, min=0.0)
```

30cm까지 보상을 주기 때문에 멀리서 방향만 맞춰도 보상을 받음.

### 4. 로컬 최적화 함정

로봇이 안전한 거리에서 방향만 맞추면 꽤 높은 리워드(~7.6)를 받을 수 있음.
더 가까이 가려면 복잡한 움직임이 필요한데, 일시적으로 보상이 감소할 수 있어서 로컬 최적값에 갇힘.

---

## V3 설계: Curriculum Learning

### Stage 구성

| Stage | 거리 조건 | Z축 조건 | 전환 조건 |
|-------|-----------|----------|-----------|
| 1 | < 10cm | dot < -0.70 | 성공률 85% |
| 2 | < 5cm | dot < -0.85 | 성공률 90% |
| 3 | < 2cm | dot < -0.95 | 최종 목표 95% |

### 왜 이런 기준인가?

**실제 Sim2Real 적용 조건:**
- 펜 지름: 19.8mm
- 잡을 수 있는 위치 오차: 1~2cm 이내
- 잡을 수 있는 각도 오차: ~18도 (dot < -0.95)

10cm는 펜 길이(12cm)에 가까운 수준으로 전혀 쓸 수 없음.

### 보상 함수 변경

```python
# V3: 현재 stage threshold를 std로 사용
dist_thresh, _ = get_stage_thresholds()
std = dist_thresh  # Stage 1: 0.1, Stage 2: 0.05, Stage 3: 0.02
return 1.0 - torch.tanh(distance / std)
```

Stage가 진행될수록 더 가까이 가야만 높은 보상을 받음.

### 성공 보상

```python
def success_bonus_reward(env) -> torch.Tensor:
    success = compute_success(env)  # 현재 stage 조건 만족 여부
    return success.float() * 10.0   # 성공 시 10.0
```

Stage 조건을 만족하면 큰 보상 (+50.0 = 10.0 * weight 5.0).

---

## 파일 구조

```
pen_grasp_rl/
├── envs/
│   ├── pen_grasp_env.py       # V1: 기존 환경
│   ├── pen_grasp_env_v2.py    # V2: reach 기반 단순화
│   └── pen_grasp_env_v3.py    # V3: Curriculum Learning
└── scripts/
    ├── train.py               # Curriculum Learning 지원
    └── play.py                # 테스트 스크립트
```

---

## 사용법

### 학습 실행

```bash
# 가상환경 활성화
source ~/isaacsim_env/bin/activate

# IsaacLab 폴더로 이동
cd ~/IsaacLab

# V3 환경으로 Curriculum Learning 학습
python pen_grasp_rl/scripts/train.py \
    --headless \
    --num_envs 4096 \
    --max_iterations 100000 \
    --env_version v3 \
    --check_interval 1000
```

### 주요 옵션

| 옵션 | 설명 | 기본값 |
|------|------|--------|
| `--env_version` | 환경 버전 (v1, v2, v3) | v3 |
| `--check_interval` | 성공률 체크 간격 | 1000 |
| `--max_iterations` | 최대 학습 횟수 | 100000 |

### 학습 재개 (Resume)

```bash
python pen_grasp_rl/scripts/train.py \
    --headless \
    --num_envs 4096 \
    --max_iterations 100000 \
    --env_version v3 \
    --resume \
    --checkpoint ./logs/pen_grasp/model_50000.pt
```

Resume 시 `curriculum_state.json`에서 현재 stage와 성공률을 자동으로 로드.

---

## Curriculum State 저장

학습 중 `logs/pen_grasp/curriculum_state.json`에 상태 저장:

```json
{
  "current_stage": 2,
  "success_rates": {
    "1": 0.87,
    "2": 0.45,
    "3": 0.0
  },
  "last_iteration": 75000
}
```

### 저장 시점
- 매 `check_interval` (1000 iteration)마다
- Stage 전환 시
- 학습 종료 시

---

## 예상 학습 과정

1. **Stage 1** (0 ~ 약 30,000 iteration)
   - 대략적인 접근 및 방향 학습
   - 성공 조건: 거리 < 10cm, dot < -0.7
   - 목표 성공률: 85%

2. **Stage 2** (약 30,000 ~ 70,000 iteration)
   - 정밀 접근 학습
   - 성공 조건: 거리 < 5cm, dot < -0.85
   - 목표 성공률: 90%

3. **Stage 3** (약 70,000 ~ 100,000+ iteration)
   - 최종 정밀 제어
   - 성공 조건: 거리 < 2cm, dot < -0.95
   - 목표 성공률: 95%

---

## 학습 모니터링

### TensorBoard

```bash
tensorboard --logdir ./logs/pen_grasp
```

### 출력 로그 예시

```
============================================================
[Iteration 50000] Stage 2
  성공률: 72.3%
  Stage 조건: 거리 < 5cm, dot < -0.85
============================================================

[Curriculum] Stage 2 성공률: 72.3% (목표: 90%)
```

---

## 성공 기준

| Stage | 성공률 기준 | 다음 Stage 전환 |
|-------|-------------|-----------------|
| 1 | 85% | → Stage 2 |
| 2 | 90% | → Stage 3 |
| 3 | 95% | 학습 완료 (조기 종료) |

---

## 주의사항

1. **학습은 별도 터미널에서 실행** (Claude 터미널은 타임아웃 있음)
2. **GPU 메모리에 따라 num_envs 조절** (4096 → 2048 등)
3. **Resume 시 동일한 env_version 사용** (v3로 학습한 것은 v3로 resume)

---

## 다음 단계

V3 학습 완료 후:
1. 성공률 95% 달성한 모델로 play.py 테스트
2. 실제 동작 확인 (마커 위치, Z축 정렬)
3. Sim2Real 테스트 준비
