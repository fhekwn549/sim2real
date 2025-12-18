# Pen Grasp RL 학습 기록

## TensorBoard 실행 방법
```bash
tensorboard --logdir=~/IsaacLab/logs/pen_grasp
# 브라우저에서 http://localhost:6006 접속
```

## 주요 지표 해석

| 지표 | 의미 | 좋은 신호 |
|------|------|-----------|
| Episode_Reward/iteration_wise | 전체 보상 | 📈 증가 |
| Episode_Reward/distance | 펜과의 거리 보상 | 📈 증가 |
| Episode_Reward/pen_lifted | 펜 들어올리기 보상 | 📈 증가 (0보다 커야 함) |
| Episode_Termination/time_out | 시간 초과 종료 비율 | 학습 초기엔 높음 |
| Episode_Termination/pen_dropped | 펜 떨어짐 종료 비율 | 낮을수록 좋음 |

---

## 학습 기록

### 2025-12-11 첫 번째 학습 (1000 iteration)
- **설정**: num_envs=4096, max_iterations=1000
- **소요 시간**: 약 15분
- **결과**:
  - distance 보상: 증가 추세 → 로봇이 펜 쪽으로 이동 학습 중
  - pen_lifted 보상: 거의 0 → 아직 펜 잡기 미성공
- **결론**: 더 많은 iteration 필요

### 2025-12-11 두 번째 학습 (3000 iteration)
- **설정**: num_envs=4096, max_iterations=3000
- **소요 시간**: 약 45분
- **결과**:
  - distance 보상: 지속 증가 → 로봇이 펜에 더 가까이 접근
  - pen_lifted 보상: 여전히 0 근처 → 펜 잡기 미성공
  - Episode_Termination: time_out이 대부분
- **분석**: play.py로 동작 확인 결과:
  - 펜이 z=0 평면(바닥)에서 소환됨
  - 그리퍼가 펜에 접근하나 잡는 동작 미완성
  - 펜을 들어올리는 것보다 잡는 것이 우선 필요

---

## 환경 수정 기록

### 2025-12-11 환경 개선 v2

#### 변경 목표
1. 펜을 공중에 띄워서 (사람이 손으로 들고 있는 상황 시뮬레이션)
2. 펜 자세를 랜덤하게 부여
3. 그리퍼가 펜의 cap 부분(point b)을 향해 접근하도록
4. pen_lifted 보상 제거 (잡기 먼저, 들기는 나중에)

#### 코드 수정 사항

**1. 펜 설정 변경 (`pen_grasp_env.py`)**
```python
# 이전: 바닥에서 소환, 중력 적용
pos=(0.4, 0.0, 0.0)

# 변경: 공중에서 소환, 중력 비활성화, kinematic
pen: RigidObjectCfg = RigidObjectCfg(
    spawn=sim_utils.CylinderCfg(
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            kinematic_enabled=True,
        ),
    ),
    init_state=RigidObjectCfg.InitialStateCfg(
        pos=(0.4, 0.0, 0.3),  # z=0.3m 공중
    ),
)
```

**2. 펜 랜덤 자세 (`_reset_idx` 함수)**
```python
# 랜덤 orientation 생성
roll = torch.rand(num_resets, device=self.device) * 1.0 - 0.5   # ±0.5 rad (약 ±30°)
pitch = torch.rand(num_resets, device=self.device) * 1.0 - 0.5  # ±0.5 rad
yaw = torch.rand(num_resets, device=self.device) * 6.28 - 3.14  # 360° 랜덤
```

**3. 새로운 관측 함수**
- `pen_orientation_obs`: 펜의 quaternion 자세
- `pen_cap_pos_obs`: 펜 cap(point b) 위치 계산
- `relative_ee_cap_obs`: 그리퍼와 cap 간의 상대 위치

**4. 보상 함수 변경**
```python
# 제거: pen_lifted_reward (잡기 전에 들기 보상은 불필요)

# 추가: distance_ee_cap_reward
# - 펜 중심이 아닌 cap(point b) 위치로 접근 유도
# - cap 위치 = pen_pos + pen_orientation * (0, 0, -PEN_LENGTH/2)
```

**5. ObservationGroup 업데이트**
```python
"policy": ObservationGroup(
    terms=[
        ObservationTerm("joint_pos", ...),
        ObservationTerm("joint_vel", ...),
        ObservationTerm("ee_pos", ...),
        ObservationTerm("pen_pos", ...),
        ObservationTerm("pen_orientation", ...),    # 추가
        ObservationTerm("relative_ee_pen", ...),
        ObservationTerm("relative_ee_cap", ...),    # 추가
        ObservationTerm("gripper_state", ...),
    ]
)
```

#### 다음 단계
- [x] 수정된 환경 테스트 (`play.py`)
- [x] Docker 환경 구축
- [ ] 새 환경으로 학습 (10000 iteration) - 진행 중

---

### 2025-12-11 Docker 환경 구축 및 새 노트북 학습

#### Docker 환경 구축
- Isaac Lab 공식 Docker 사용 (`nvcr.io/nvidia/isaac-sim`)
- `container.py` 스크립트로 관리 (docker compose 직접 사용 시 환경변수 오류)
- 볼륨 마운트: pen_grasp_rl, logs, e0509_gripper_isaac

#### USD 파일 참조 문제 해결
- `first_control.usd`가 `/workspace/e0509_gripper_isaac/e0509_gripper_isaac.usd` 참조
- `e0509_gripper_isaac` 폴더를 레포에 추가하고 Docker 볼륨 마운트로 해결

#### 펜 스폰 범위 수정 (실제 작업 공간 기준)
```python
# 실제 로봇 작업 범위 측정값 기준
"pose_range": {
    "x": (-0.2, 0.2),      # 로봇 기준 0.3~0.7m
    "y": (-0.3, 0.3),      # 좌우 ±30cm
    "z": (-0.2, 0.2),      # 높이 0.1~0.5m
}
```

#### play.py 마커 추가
- Tip (파란색): 필기 끝 (pen_pos + axis * half_len)
- Cap (빨간색): 그리퍼가 잡아야 할 곳 (pen_pos - axis * half_len)

#### 새 노트북 학습 시작
- **하드웨어**: RTX 5080 (16GB VRAM)
- **설정**: num_envs=8192, max_iterations=10000
- **상태**: 학습 진행 중
- **TensorBoard**: 컨테이너 내부에서 실행 권장

#### 관련 문서
- `DOCKER_GUIDE.md`: Docker 환경 설정 가이드
- `docker_setup.sh`: 컨테이너 내 의존성 설치 스크립트

---

### 2025-12-11 Grasp Point 및 보상함수 개선

#### 문제 분석
- 기존 gripper center가 손가락 끝 중앙이라 그리퍼 open/close 상태에 따라 이동
- 보상함수가 펜에 접근만 유도하고, 정렬(orientation)은 고려하지 않음

#### 변경 사항

**1. Grasp Point 계산 방식 변경 (`pen_grasp_env.py`)**
```python
def get_grasp_point(robot: Articulation) -> torch.Tensor:
    """Get ideal grasp point: (l1+r1)/2 center + 2cm along finger direction.

    This point is stable regardless of gripper open/close state.
    """
    # [7] l1, [8] r1 = 손가락 베이스
    # [9] l2, [10] r2 = 손가락 끝
    l1 = robot.data.body_pos_w[:, 7, :]
    r1 = robot.data.body_pos_w[:, 8, :]
    l2 = robot.data.body_pos_w[:, 9, :]
    r2 = robot.data.body_pos_w[:, 10, :]

    base_center = (l1 + r1) / 2.0
    tip_center = (l2 + r2) / 2.0
    finger_dir = tip_center - base_center
    finger_dir = finger_dir / (torch.norm(finger_dir, dim=-1, keepdim=True) + 1e-6)

    return base_center + finger_dir * 0.02  # 2cm along finger direction
```

**2. z축 정렬 보상함수 추가**
```python
def z_axis_alignment_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Reward for aligning gripper z-axis with pen z-axis.

    Only gives reward when:
    1. Gripper is close to pen cap (within 5cm)
    2. Z-axes are nearly parallel (dot product > 0.9)
    """
    # ... pen z-axis, gripper z-axis 계산 ...

    dot_product = torch.sum(pen_z_axis * gripper_z_axis, dim=-1)

    # Only reward when nearly parallel (dot > 0.9)
    alignment_reward = torch.clamp(dot_product - 0.9, min=0.0) * 10.0

    # Only apply when close to cap (within 5cm)
    distance_factor = torch.clamp(1.0 - distance_to_cap / 0.05, min=0.0)

    return alignment_reward * distance_factor
```

**3. 현재 보상함수 구성**
| 보상함수 | weight | 설명 |
|---------|--------|------|
| `distance_to_cap` | 1.0 | grasp point → 펜 캡 거리 |
| `z_axis_alignment` | 0.5 | z축 정렬 (캡 5cm 이내 + dot>0.9 일때만) |
| `action_rate` | 0.1 | 액션 크기 페널티 |

**4. play.py 마커 개선**
- Cap (빨강): 펜 캡 위치 (목표)
- Grasp Point (초록): 그리퍼 잡기 위치
- Pen z-axis (파랑): 펜 중심에서 z축 방향 (5개 점, 15cm)
- Gripper z-axis (노랑): grasp point에서 link_6 z축 방향 (5개 점, 15cm)

#### 커리큘럼 러닝 전략
- **Phase 1 (현재)**: 펜 kinematic, 위치+정렬 학습
- **Phase 2 (추후)**: 펜 dynamic, 잡기 동작 학습
- 기존 학습된 "접근+정렬" 정책이 Phase 2에서 fine-tuning으로 활용됨

---

### 2025-12-11 추가 개선 사항

#### 1. 로봇 USD에서 불필요한 펜 제거
- `first_control.usd` 내부에 펜 오브젝트가 포함되어 있었음
- Isaac Sim에서 USD 열어서 Robot/Pen 삭제 후 저장
- 이전 학습에서 이 펜이 물리적 노이즈로 작용했을 가능성 있음

#### 2. 펜 자세 랜덤화 범위 확대
```python
# 이전: 거의 수직으로만 스폰
"roll": (-0.5, 0.5),   # ±30°
"pitch": (-0.5, 0.5),  # ±30°

# 변경: 완전 랜덤 (뒤집힘 포함)
"roll": (-3.14, 3.14),   # ±180°
"pitch": (-3.14, 3.14),  # ±180°
```

#### 3. 바닥 충돌 페널티 추가 (실제 접촉력 기반)
```python
def floor_collision_penalty(env) -> torch.Tensor:
    """로봇 링크가 바닥에 닿으면 페널티."""
    # 접촉력 z성분 확인 (바닥이 위로 밀어올림)
    contact_forces_z = robot.data.net_contact_forces_w[:, 2:11, 2]
    link_z = robot.data.body_pos_w[:, 2:11, 2]

    # 바닥 충돌: 위쪽 접촉력 > 1N AND 링크 z < 0.1m
    floor_contact = ((contact_forces_z > 1.0) & (link_z < 0.1)).any(dim=-1)
    return -floor_contact.float()
```

#### 4. 현재 보상함수 구성
| 보상함수 | weight | 설명 |
|---------|--------|------|
| `distance_to_cap` | 1.0 | grasp point → 펜 캡 거리 |
| `z_axis_alignment` | 0.5 | z축 정렬 (5cm 이내 + dot>0.9) |
| `floor_collision` | 1.0 | 바닥 실제 충돌 시 -1 페널티 |
| `action_rate` | 0.1 | 액션 크기 페널티 |

---

### 2025-12-12 z_axis_alignment 보상함수 개선

#### 50,000 iteration 학습 결과 분석
- **distance_to_cap**: 0.96 (성공적으로 펜 캡 접근 학습)
- **z_axis_alignment**: ~0 (정렬 보상 거의 없음)
- **floor_collision**: -0.001 (바닥 충돌 거의 없음)

#### 문제점
기존 z_axis_alignment 조건이 너무 까다로움:
- 5cm 이내 접근 AND dot product > 0.9 일때만 보상
- 로봇이 접근은 하지만 정확한 각도로 정렬되는 순간이 거의 없어 보상을 못 받음

#### 해결책: 거리 기반 가중치 적용
```python
def z_axis_alignment_reward(env) -> torch.Tensor:
    # 기존: 5cm 이내 + dot > 0.9 일때만 보상
    # 변경: 거리와 무관하게 정렬 보상, 단 가까울수록 가중치 증가

    # dot product: 양수만 보상 (캡 방향만 허용, 팁 방향은 보상 0)
    dot_product = torch.sum(pen_z_axis * gripper_z_axis, dim=-1)
    alignment_score = torch.clamp(dot_product, min=0.0)  # 0 ~ 1

    # 거리 가중치: 가까울수록 높음
    # 5cm: weight = 10, 50cm: weight ≈ 1.8
    distance_weight = 1.0 / (distance_to_cap + 0.05)

    return alignment_score * distance_weight * 0.1
```

#### 개선 효과
- 멀리서도 방향 맞추면 작은 보상 (방향 학습 힌트)
- 가까이 가면서 정렬하면 큰 보상
- 접근 + 정렬 동시 학습 유도

#### 다음 단계
- [x] 새로운 보상함수로 학습 실행
- [ ] TensorBoard에서 z_axis_alignment 보상 증가 확인

---

### 2025-12-12 Phase 2 구현: 펜 충돌 및 그립 동작

#### 50,000 iteration 학습 결과 추가 분석
- play.py 실행 결과, 로봇이 펜 **팁** 방향으로 접근하고 있었음
- **원인**: z_axis_alignment에서 `torch.clamp(dot_product, min=0.0)` 사용
  - dot=+1.0 (같은 방향)일 때 보상 → 잘못된 방향
  - 실제로는 dot=-1.0 (반대 방향)일 때 보상해야 함 (그리퍼가 캡을 마주보며 접근)

#### z_axis_alignment 방향 수정
```python
# 이전: 같은 방향일 때 보상 (틀림)
alignment_score = torch.clamp(dot_product, min=0.0)

# 수정: 반대 방향일 때 보상 (올바름)
alignment_score = torch.clamp(-dot_product, min=0.0)
```

#### Phase 2 변경 사항

**1. 펜 모델 변경**
- 팀원이 모델링한 pen.usd 적용 (뚜껑 없는 상태, 117mm)
- PEN_LENGTH: 0.1207 → 0.117

**2. 펜 충돌 활성화**
```python
# 이전: kinematic_enabled=True (고정)
# 변경: kinematic_enabled=False (충돌 가능)
rigid_props=sim_utils.RigidBodyPropertiesCfg(
    disable_gravity=True,      # 공중에 떠있음
    kinematic_enabled=False,   # 그리퍼에 맞으면 밀림
)
```

**3. 새로운 Observation 추가**
```python
gripper_state = ObsTerm(func=gripper_state_obs)  # 그리퍼 열림/닫힘 상태 (0~1)
```

**4. 새로운 보상함수 추가**
| 보상함수 | weight | 설명 |
|---------|--------|------|
| `pen_displacement_penalty` | 1.0 | 펜을 치면 속도에 비례한 페널티 |
| `grasp_success_reward` | 2.0 | 3cm 이내 + 정렬 + 그리퍼 닫힘 시 큰 보상 |

```python
def pen_displacement_penalty(env) -> torch.Tensor:
    """펜 속도에 비례한 페널티 (펜을 함부로 치지 않도록)"""
    pen_vel = pen.data.root_lin_vel_w
    vel_magnitude = torch.norm(pen_vel, dim=-1)
    return -vel_magnitude * 0.5

def grasp_success_reward(env) -> torch.Tensor:
    """성공적인 그립 자세 달성 시 보상"""
    close_enough = (distance_to_cap < 0.03).float()  # 3cm 이내
    aligned = (dot_product < -0.8).float()           # 반대 방향 정렬
    gripper_closed = (gripper_pos > 0.5).all().float()  # 그리퍼 닫힘
    return close_enough * aligned * gripper_closed * 5.0
```

**5. Termination 조건 변경**
```python
# 이전: 펜 z < 0.01 (바닥에 떨어지면 종료)
# 변경: 펜이 초기 위치에서 15cm 이상 이탈하면 종료
def pen_dropped_termination(env) -> torch.Tensor:
    pen_pos = pen.data.root_pos_w - env.scene.env_origins
    init_pos = torch.tensor([0.5, 0.0, 0.3])
    displacement = torch.norm(pen_pos - init_pos, dim=-1)
    return displacement > 0.15  # 어느 방향이든 15cm 이상 밀리면 실패
```

**6. play.py cap 위치 수정**
```python
# 이전: cap_pos = pen_pos - pen_axis_world * half_len (틀림)
# 수정: cap_pos = pen_pos + pen_axis_world * half_len (올바름)
```

#### 현재 보상함수 구성
| 보상함수 | weight | 설명 |
|---------|--------|------|
| `distance_to_cap` | 1.0 | grasp point → 펜 캡 거리 |
| `z_axis_alignment` | 0.5 | z축 반대 방향 정렬 (거리 가중치) |
| `floor_collision` | 1.0 | 바닥 충돌 페널티 |
| `pen_displacement` | 1.0 | 펜 밀림 페널티 |
| `grasp_success` | 2.0 | 성공적 그립 보상 |
| `action_rate` | 0.1 | 액션 크기 페널티 |

#### 커리큘럼 러닝 진행 상황
- **Phase 1**: 펜 고정 (kinematic=True), 접근+정렬 학습 → 완료
- **Phase 2 (현재)**: 펜 충돌 활성화, 그립 동작 학습 → 준비 완료

#### 다음 단계
- [ ] Phase 2 학습 실행
- [ ] 펜을 밀지 않고 조심스럽게 접근하는지 확인
- [ ] grasp_success 보상이 발생하는지 확인

---

### 2025-12-12 그리퍼 회전 수정 및 정밀 펜 모델 적용

#### 문제 발견
- 강화학습 USD 내의 그리퍼가 실제 그리퍼와 90도 회전되어 있었음
- XACRO 파일의 `gripper_attach_joint` rpy 값이 잘못 설정됨

#### 그리퍼 USD 수정

**1. XACRO 파일 수정**
```xml
<!-- 이전 -->
<joint name="gripper_attach_joint" type="fixed">
  <origin xyz="0 0 0" rpy="0 0 0"/>
</joint>

<!-- 수정 -->
<joint name="gripper_attach_joint" type="fixed">
  <origin xyz="0 0 0" rpy="0 0 1.5708"/>  <!-- Z축 90도 회전 -->
</joint>
```

**2. USD 재생성**
```bash
# XACRO → URDF
xacro e0509_with_gripper.urdf.xacro > e0509_gripper_isaaclab_absolute.urdf

# package:// 경로를 절대 경로로 변환
sed -i 's|package://e0509_description|/home/.../e0509_description|g' ...

# URDF → USD (Isaac Lab 변환기 사용)
./isaaclab.sh -p scripts/tools/convert_urdf.py \
  e0509_gripper_isaaclab_absolute.urdf \
  e0509_gripper_isaaclab/e0509_gripper_isaaclab.usd \
  --merge-joints
```

#### 정밀 펜 USD 모델 생성

**펜 구조 (뚜껑 씌운 상태)**
```
[뒷캡] ─── [본체] ─── [펜촉 뚜껑]
  5mm      81.7mm       34mm
           ↓
        전체 120.7mm
```

| 부분 | 형태 | 치수 |
|------|------|------|
| 뒷캡 | 원통 | Ø13.5mm, 5mm |
| 본체 | 원뿔대 | Ø19.8mm → Ø17mm, 81.7mm |
| 펜촉 뚜껑 (원뿔대) | 원뿔대 | Ø17mm → Ø16mm, 29mm |
| 펜촉 뚜껑 (반구) | 반구 | Ø16mm, 5mm |
| 무게 | - | 16.3g |

**create_pen_usd.py 스크립트 작성**
- `create_truncated_cone_mesh()`: 원뿔대 메시 생성
- `create_hemisphere_mesh()`: 반구 메시 생성
- RigidBodyAPI, MassAPI, CollisionAPI 적용

#### 환경 설정 변경

**pen_grasp_env.py 수정**
```python
# 이전: CylinderCfg (단순 원통)
spawn=sim_utils.CylinderCfg(
    radius=0.005,
    height=PEN_LENGTH,
    ...
)

# 변경: UsdFileCfg (정밀 모델)
spawn=sim_utils.UsdFileCfg(
    usd_path=PEN_USD_PATH,
    rigid_props=sim_utils.RigidBodyPropertiesCfg(
        disable_gravity=True,
        kinematic_enabled=False,
    ),
    mass_props=sim_utils.MassPropertiesCfg(mass=PEN_MASS),
    collision_props=sim_utils.CollisionPropertiesCfg(),
)
```

**상수 변경**
```python
PEN_LENGTH = 0.1207  # 120.7mm (이전: 117mm)
```

#### test_env.py 마커 추가
- 빨간색: 펜 뒷캡 위치 (+Z 방향, 잡을 부분)
- 초록색: 그리퍼 잡기 포인트
- 파란색: 펜 Z축 방향
- 노란색: 그리퍼 Z축 방향

#### 검증 결과
- 그리퍼 수직 방향 확인 ✓
- 펜 모델 형상 정확 ✓
- 보상함수 목표 위치 (뒷캡) 확인 ✓

#### 다음 단계
- [x] 새 모델로 Phase 2 학습 실행
- [ ] 정밀 펜 모델에서 그립 동작 확인

---

### 2025-12-15 주말 학습 결과 분석 및 환경 대폭 수정

#### 학습 결과 (2025-12-12 ~ 2025-12-15, 약 200K iteration)

**로그 위치**: `/home/fhekwn549/pen_grasp_logs/pen_grasp/2025-12-12_11-38-23`

| 지표 | 시작 | 최종 | 비고 |
|------|------|------|------|
| Mean Reward | 0.05 | -1,600 | 발산 |
| Episode Length | 5.1 | 2.0 | 거의 즉시 종료 |
| action_rate | -0.00 | -27,793 | 폭발 |
| Value Loss | 0.01 | 12.4T | 완전 발산 |
| pen_dropped | - | 99.9% | 펜 항상 밀림 |

#### 문제 원인 분석

**1. 충돌 + 종료 조건의 악순환**
```
로봇 접근 → 펜 충돌 → 펜 15cm 이동 → 에피소드 즉시 종료
→ 학습할 시간 없음 → 정책 혼란 → 행동 값 폭발 → 완전 발산
```

**2. 시간대별 발산 추이**
| Step | Mean Reward | action_rate | Value Loss |
|------|-------------|-------------|------------|
| 0 | 0.05 | -0.00 | 0.01 |
| 10K | 0.52 | -0.02 | 0.02 |
| **50K** | **-21,032** | **-3,009** | **5.9B** |
| 150K | -3,873,878 | - | - |
| 200K | -1,600 | -27,793 | 12.4T |

Step 50K 부근에서 급격히 발산 시작.

#### 환경 대폭 수정

**1. 펜 물리 설정 변경**
```python
# 이전: 중력 없음 (공중에 고정)
disable_gravity=True

# 변경: 중력 있음 (떨어질 수 있음)
disable_gravity=False
```

**2. 종료 조건 변경**
```python
# 이전: 펜이 15cm 이상 밀리면 종료 (너무 민감)
def pen_dropped_termination(env):
    displacement = torch.norm(pen_pos - init_pos, dim=-1)
    return displacement > 0.15

# 변경: 펜이 바닥으로 떨어지면 종료 (현실적)
def pen_fell_termination(env):
    pen_z = pen.data.root_pos_w[:, 2]
    return pen_z < 0.05  # 5cm 이하
```

**3. 행동 공간 단순화 (7차원 → 6차원)**
```python
# 이전: 팔 6 + 그리퍼 1 = 7차원
class ArmGripperActionTerm:
    action_dim = 7
    # 그리퍼 제어 포함

# 변경: 팔만 6차원, 그리퍼 열린 상태 고정
class ArmActionTerm:
    action_dim = 6
    # 그리퍼 항상 0 (열림)
```

**4. 보상 함수 수정**
```python
# 이전: grasp_success (위치 + 정렬 + 그리퍼 닫힘)
grasp_success = RewTerm(func=grasp_success_reward, weight=2.0)

# 변경: alignment_success (위치 + 정렬만)
alignment_success = RewTerm(func=alignment_success_reward, weight=2.0)
```

**5. 관찰 공간 수정 (36차원 → 36차원, gripper_state 제거)**
```python
# 제거: gripper_state (1차원)
# 그리퍼가 항상 열려 있으므로 불필요
```

#### 수정 근거

| 기존 문제 | 해결책 | 이유 |
|----------|--------|------|
| 펜 밀리면 즉시 종료 | 떨어지면 종료 | 학습 시간 확보 |
| 펜이 우주로 날아감 | 중력 활성화 | 현실적 물리 |
| 그리퍼 학습 불필요 | 그리퍼 제거 | 위치+정렬만 학습 |
| 복잡한 성공 조건 | 그리퍼 조건 제거 | 학습 목표 단순화 |

#### 학습 목표 재정의

**Phase 1 (현재)**
- 목표: 그리퍼를 펜 캡 위치로 이동 + Z축 정렬
- 그리퍼: 항상 열린 상태
- 잡기: 학습하지 않음 (실제 로봇에서 수행)

**Phase 2 (추후)**
- Phase 1 학습된 정책 기반
- 그리퍼 닫기 동작 추가
- 또는 실제 로봇에서 단순 오므리기로 해결

#### 현재 보상 함수 구성
| 보상함수 | weight | 설명 |
|---------|--------|------|
| `distance_to_cap` | 1.0 | grasp point → 펜 캡 거리 |
| `z_axis_alignment` | 0.5 | z축 반대 방향 정렬 |
| `floor_collision` | 1.0 | 바닥 충돌 페널티 |
| `pen_displacement` | 1.0 | 펜 밀림 페널티 |
| `alignment_success` | 2.0 | 위치+정렬 성공 보상 |
| `action_rate` | 0.1 | 액션 크기 페널티 |

#### 다음 단계
- [x] 수정된 환경으로 학습 실행
- [x] pen_fell 종료 비율 모니터링

---

### 2025-12-15 중력 설정 재수정

#### 문제 발생
- pen_fell 종료 비율이 99.9%로 나옴
- 원인: 펜이 공중(z=0.3m)에서 스폰 → 중력으로 바로 떨어짐 → 즉시 종료
- 로봇이 뭔가 하기도 전에 에피소드 종료

#### 해결책: 중력 다시 비활성화
```python
# 변경: 중력 끄기 (펜 공중 고정)
disable_gravity=True

# 종료 조건: time_out만 (pen_fell 제거)
time_out = DoneTerm(func=mdp.time_out, time_out=True)
```

#### 현재 설정 요약
| 항목 | 설정 |
|------|------|
| 펜 중력 | 비활성화 (공중 고정) |
| 펜 충돌 | 활성화 (그리퍼가 밀 수 있음) |
| 종료 조건 | time_out만 (10초) |
| 학습 목표 | 접근 + 정렬 (pen_displacement 페널티로 밀기 억제) |

#### 다음 단계
- [ ] 수정된 환경으로 학습 재실행
- [ ] 발산 여부 확인 (action_rate, value_loss)
- [ ] alignment_success 보상 발생 확인

---

### 2025-12-15 학습 설정 최적화 (시험 학습 분석 기반)

#### 시험 학습 분석 (2025-12-15_02-55-00)
- **진행**: 5,900 iterations
- **문제**: 리워드가 +1.7에서 -3.9로 급락
- **지표**:
  | 지표 | 초반 | 최종 | 상태 |
  |------|------|------|------|
  | Mean Reward | 1.77 | -3.93 | 급락 |
  | Value Loss | 0.005 | 5.39 | 불안정 |
  | Entropy | 8.5 | 26.2 | 비정상 증가 |
  | Alignment Success | 0.0 | 0.0 | 성공 없음 |

#### 원인 분석
1. **init_noise_std=1.0이 너무 높음**: 액션 범위 [-1,1]에서 노이즈 1.0이면 거의 랜덤
2. **펜 방향 360도 랜덤화**: 학습 초기에 캡 위치 찾기가 너무 어려움
3. **alignment 조건 5cm 이내**: 조건이 너무 엄격해서 보상을 거의 못 받음
4. **action_rate 페널티 너무 작음**: 실제 페널티 = action² × 0.0001

#### 변경 사항

**1. PPO 하이퍼파라미터 (train.py)**
| 항목 | 이전 | 변경 |
|------|------|------|
| init_noise_std | 1.0 | **0.3** |

**2. 펜 방향 설정 (pen_grasp_env.py)**
```python
# 이전: 모든 방향 랜덤
"roll": (-3.14, 3.14),
"pitch": (-3.14, 3.14),
"yaw": (-3.14, 3.14),

# 변경: 수직 고정 (캡이 위를 향함)
# roll, pitch, yaw 모두 제거
# 학습 성공 후 점진적으로 각도 추가 예정
```

**3. 로봇 초기 자세**
```python
# 이전
"joint_5": -1.57,  # -90° (그리퍼가 위를 향함)

# 변경
"joint_5": 1.57,   # +90° (그리퍼가 아래를 향함, 펜 캡 잡기 용이)
```

**4. 리워드 함수 수정**
| 항목 | 이전 | 변경 |
|------|------|------|
| distance_to_cap | `1/(1+d*10)` | **`exp(-d*10)`** (exponential) |
| z_axis_alignment 거리조건 | 5cm | **10cm** |
| action_rate 함수 | `action² * 0.001` | **`action²`** |
| action_rate weight | 0.1 | **0.01** |

**5. 펜 모델**
- 이전: CylinderCfg (단순 실린더)
- 변경: **pen.usd** (BackCap, Body, TipCone, TipSphere 포함)

#### 학습 실행 명령어
```bash
cd ~/IsaacLab
source ~/isaacsim_env/bin/activate
python pen_grasp_rl/scripts/train.py --headless --num_envs 4096 --max_iterations 5000
```

#### 다음 단계
- [x] 현재 설정으로 학습 진행
- [ ] 학습 성공 시 펜 각도 랜덤화 추가 (±30도부터 시작)
- [ ] 그리퍼 잡기 동작 추가

---

### 2025-12-15 10,000 iteration 학습 분석 및 z축 정렬 버그 수정

#### 학습 결과 분석 (10,000 iteration)

**로그 위치**: `/home/fhekwn549/pen_grasp/`

| 지표 | 초기 | 최종 | 상태 |
|------|------|------|------|
| distance_to_cap | 0.004 | 0.752 | 학습됨 (약 2.9cm 거리) |
| z_axis_alignment | 0.0 | **0.0** | 전혀 학습 안됨 |
| floor_collision | -0.68 | 0.0 | 해결됨 |
| mean_episode_length | 22 | 300 | 최대 길이까지 생존 |
| mean_reward | -6.57 | 7.41 | 증가 |

#### 문제 발견: z_axis_alignment가 항상 0인 이유

**play.py로 시각적 확인 결과:**
1. Blue 점들 (pen z-axis): 펜 중심에서 **캡 방향**으로 뻗어나감 (+Z)
2. Yellow 점들 (gripper z-axis): 그리퍼 안쪽에서 **손가락이 뻗어나가는 방향**

**결론:**
- 펜 +Z: 캡 방향 (위로 향함)
- 그리퍼 +Z: 손가락 끝 방향 (아래로 향해야 캡 잡기 가능)
- 따라서 **dot product = -1** 일 때 올바른 정렬

**기존 코드 버그:**
```python
# 이전 (잘못됨): dot > 0.9 일 때 보상 → 절대 달성 불가
alignment_reward = torch.clamp(dot_product - 0.9, min=0.0) * 10.0
```

#### 수정 사항

**1. z_axis_alignment_reward 방향 수정 (`pen_grasp_env.py`)**
```python
# 수정 (올바름): dot < -0.9 일 때 보상 (반대 방향)
alignment_reward = torch.clamp(-dot_product - 0.9, min=0.0) * 10.0
```

**2. play.py cap 마커 위치 수정**
```python
# 이전 (잘못됨)
cap_pos = pen_pos - pen_axis_world * half_len

# 수정 (올바름): pen +Z 방향이 캡
cap_pos = pen_pos + pen_axis_world * half_len
```

#### 다음 단계
- [x] 수정된 보상함수로 학습 재개 (resume)
- [ ] TensorBoard에서 z_axis_alignment 보상 증가 확인
- [ ] 그리퍼가 위에서 캡을 향해 내려오는지 play.py로 확인

---

### 2025-12-16 50,000 iteration 학습 분석 및 보상 함수 개선

#### 학습 결과 분석 (z축 수정 후 50,000 iteration)

**로그 위치**: `/home/fhekwn549/pen_grasp/`

| 지표 | 초기 | 최종 | 상태 |
|------|------|------|------|
| distance_to_cap | 0.73 | 0.72 | 안정 (약 3cm 거리) |
| z_axis_alignment | 0.22 | **0.26** | 소폭 상승 |
| floor_collision | 0.0 | 0.0 | 바닥 충돌 없음 |
| mean_episode_length | 300 | 300 | 모두 시간 초과 |
| mean_reward | 9.6 | 9.0 | 안정적 |
| time_out | 100% | 100% | 성공 종료 없음 |

#### 문제점 분석

1. **z_axis_alignment 보상이 여전히 낮음 (최대 0.26)**
   - weight가 0.5로 distance_to_cap(1.0)보다 낮아 우선순위가 밀림

2. **태스크 완료 조건 없음**
   - 성공해도 에피소드가 계속 진행됨
   - 성공 시 큰 보상 + 조기 종료가 필요

3. **모든 에피소드가 시간 초과로 종료**
   - 성공/실패 구분이 없음

#### 변경 사항

**1. z_axis_alignment weight 증가**
```python
# 이전
z_axis_alignment = RewTerm(func=z_axis_alignment_reward, weight=0.5)

# 변경
z_axis_alignment = RewTerm(func=z_axis_alignment_reward, weight=1.5)
```

**2. alignment_success_reward 함수 추가**
```python
def alignment_success_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    """
    위치 + 정렬 성공 보상

    성공 조건:
    1. 그리퍼-캡 거리 3cm 이내
    2. Z축 정렬 (dot product < -0.9)

    Returns:
        (num_envs,) - 성공 시 1.0, 아니면 0.0
    """
    # ... 거리 및 정렬 계산 ...
    close_enough = distance_to_cap < 0.03
    aligned = dot_product < -0.9
    return (close_enough & aligned).float()
```

**3. task_success_termination 종료 조건 추가**
```python
def task_success_termination(env: ManagerBasedRLEnv) -> torch.Tensor:
    """
    Task 성공 종료 조건

    성공 조건: 거리 < 3cm AND dot < -0.9
    """
    # ... alignment_success_reward와 동일한 로직 ...
    return close_enough & aligned
```

#### 현재 보상 함수 구성

| 보상함수 | weight | 설명 |
|---------|--------|------|
| `distance_to_cap` | 1.0 | grasp point → 펜 캡 거리 (exponential) |
| `z_axis_alignment` | **1.5** | z축 반대 방향 정렬 (기존 0.5) |
| `alignment_success` | **5.0** | 위치+정렬 성공 시 큰 보상 (신규) |
| `floor_collision` | 1.0 | 바닥 충돌 페널티 |
| `action_rate` | 0.01 | 액션 크기 페널티 |

#### 현재 종료 조건

| 종료 조건 | 설명 |
|----------|------|
| `time_out` | 에피소드 시간 초과 (10초) |
| `pen_dropped` | 펜 낙하 (현재 kinematic이라 미발동) |
| `task_success` | **위치+정렬 성공 시 조기 종료 (신규)** |

#### TensorBoard 확인 지표

학습 진행 시 다음 지표 모니터링:
- `Episode_Reward/alignment_success`: 0보다 커지면 성공 발생
- `Episode_Termination/task_success`: 성공률 (높을수록 좋음)
- `Episode_Termination/time_out`: 낮아져야 함 (성공이 늘면 감소)

#### 다음 단계
- [x] 수정된 환경으로 학습 재개 (--resume)
- [ ] task_success 종료 비율 모니터링
- [ ] 성공 시 그리퍼 닫기 동작 추가 검토

---

### 2025-12-16 보상 조건 완화 (학습 정체 해결)

#### 문제점
- 50,000 iter resume 후 추가 학습에서 reward 정체
- task_success: 0.3~0.5% (매우 낮음)
- 원인: checkpoint + reward 구조 변경 시 기존 policy 혼란

#### 발견된 문제: "절벽 보상" 구조

**기존 z_axis_alignment_reward:**
```python
alignment_reward = torch.clamp(-dot_product - 0.9, min=0.0) * 10.0
```

| dot product | 보상 |
|-------------|------|
| -1.0 (완벽) | 1.0 |
| -0.9 | **0** |
| -0.5 | **0** |
| 0.0 (수직) | **0** |

→ dot > -0.9 이면 보상 = 0 (학습 신호 없음!)

#### 수정 사항

**1. z_axis_alignment_reward: 절벽 → 점진적 보상**
```python
# 수정: 전체 범위에서 점진적 보상
# dot = -1 → 1.0, dot = 0 → 0.5, dot = +1 → 0.0
alignment_reward = (-dot_product + 1.0) / 2.0

# 거리 조건 완화: 10cm → 30cm
distance_factor = torch.clamp(1.0 - distance_to_cap / 0.30, min=0.0)
```

**2. alignment_success_reward: 조건 완화**
```python
close_enough = distance_to_cap < 0.05  # 3cm → 5cm
aligned = dot_product < -0.7           # -0.9 → -0.7 (약 45도)
```

**3. task_success_termination: 조건 완화**
```python
close_enough = distance_to_cap < 0.05  # 3cm → 5cm
aligned = dot_product < -0.7           # -0.9 → -0.7
```

#### 수정 요약

| 항목 | 수정 전 | 수정 후 |
|------|---------|---------|
| **정렬 보상 방식** | 절벽 (dot < -0.9만) | 점진적 (전체 범위) |
| **정렬 보상 거리** | 10cm 이내 | 30cm 이내 |
| **Success 거리** | 3cm | 5cm |
| **Success 정렬** | dot < -0.9 (~25도) | dot < -0.7 (~45도) |
| **학습 시작** | checkpoint 이어서 | **새로 시작** |

#### 학습 실행 명령어
```bash
# checkpoint 없이 새로 시작
python train.py --headless --num_envs 8192 --max_iterations 20000
```

#### 다음 단계
- [x] 새 환경으로 학습 진행
- [x] z_axis_alignment 보상 증가 확인
- [ ] task_success 비율 증가 확인
- [ ] 학습 성공 시 조건 점진적 강화 (5cm→3cm, -0.7→-0.85)

---

### 2025-12-16 접근 방향 보상 추가 (측면 충돌 문제 해결)

#### 학습 결과 분석 (model_8100.pt, 약 8,000 iteration)

**로그 위치**: `/home/fhekwn549/events.out.tfevents.*.0`

| 지표 | 초기 | 최종 | 상태 |
|------|------|------|------|
| distance_to_cap | 0.016 | **0.74** | ✅ 펜 캡 접근 학습됨 |
| z_axis_alignment | 0.018 | **1.11** | ✅ 정렬 보상 증가 |
| alignment_success | 0 | 0.0001 | ❌ 거의 성공 없음 |
| task_success | 0% | **0.66%** | ❌ 매우 낮음 |
| time_out | 40% | **99%** | ❌ 거의 모두 타임아웃 |
| mean_episode_length | 130 | **300** (최대) | 항상 최대 길이 |
| mean_noise_std | 0.31 | **0.73** | ⚠️ 탐색 노이즈 증가 |

#### play.py 시각적 확인 결과

**관찰된 동작:**
1. 그리퍼가 벌린 상태로 펜 캡 방향으로 잘 접근함
2. **문제**: 그리퍼가 **측면에서** 접근해서 펜과 충돌
3. 충돌 후 더 이상 진행 불가 (펜이 kinematic으로 고정)
4. z축 정렬이 안 되는 근본 원인: 접근 방향이 잘못됨

```
나쁜 접근 (현재):          좋은 접근 (목표):
  그리퍼 →                    ↓ 그리퍼
      |                       |
     [펜]                    [펜]
```

#### 해결책: 접근 방향 보상 추가

**1. base_orientation_reward (신규)**
- joint_1이 펜 방향을 향하도록 유도
- 로봇 베이스가 펜 쪽으로 회전하면 보상

```python
def base_orientation_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    """
    베이스(joint_1) 방향 보상
    joint_1이 펜 방향을 향하면 높은 보상
    """
    # 펜 방향 각도 계산
    to_pen_angle = torch.atan2(to_pen[:, 1], to_pen[:, 0])

    # joint_1 현재 각도
    joint_1_angle = robot.data.joint_pos[:, 0]

    # 각도 차이가 작을수록 높은 보상
    angle_diff = torch.abs(to_pen_angle - joint_1_angle)
    return torch.exp(-angle_diff * 2.0)
```

**2. approach_from_above_reward (신규)**
- 그리퍼가 펜 캡 위쪽에서 접근하도록 유도
- 측면 충돌 방지

```python
def approach_from_above_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    """
    위에서 접근 보상
    그리퍼→캡 방향이 펜 z축과 반대(위에서 내려오는)일수록 높은 보상
    """
    # 그리퍼에서 캡으로의 방향
    to_cap_normalized = (cap_pos - grasp_pos) / distance

    # 위에서 접근하면 to_cap ≈ -pen_z_axis
    alignment = -torch.sum(to_cap_normalized * pen_z_axis, dim=-1)

    # 30cm 이내에서 활성화
    distance_factor = torch.clamp(1.0 - distance / 0.30, min=0.0)

    return torch.clamp(alignment, min=0.0) * distance_factor
```

#### 현재 보상 함수 구성

| 보상함수 | weight | 설명 | 상태 |
|---------|--------|------|------|
| `distance_to_cap` | 1.0 | 캡에 가까워지기 | 기존 |
| `z_axis_alignment` | 1.5 | z축 반대 방향 정렬 | 기존 |
| `base_orientation` | **0.5** | joint_1이 펜 방향 향하기 | **신규** |
| `approach_from_above` | **1.0** | 위에서 접근하기 | **신규** |
| `alignment_success` | 5.0 | 위치+정렬 성공 보상 | 기존 |
| `floor_collision` | 1.0 | 바닥 충돌 페널티 | 기존 |
| `action_rate` | 0.01 | 액션 크기 페널티 | 기존 |

#### 기대 효과

새 보상 구조로 로봇이 다음 순서로 동작할 것으로 예상:

1. **joint_1 회전** → 펜 방향으로 팔 정렬 (base_orientation)
2. **캡 위로 이동** → 펜 캡 위쪽에 위치 (approach_from_above)
3. **위에서 하강** → 캡을 향해 내려옴 (distance_to_cap)
4. **z축 정렬** → 그리퍼 방향 맞추기 (z_axis_alignment)
5. **성공** → 위치+정렬 달성 (alignment_success + task_success)

#### 학습 실행 명령어

```bash
# 처음부터 새로 학습 (기존 모델은 측면 접근으로 학습됨)
source ~/isaacsim_env/bin/activate
cd ~/IsaacLab
python pen_grasp_rl/scripts/train.py --num_envs 64
```

#### TensorBoard 확인 지표

- `Episode_Reward/base_orientation`: 증가 확인
- `Episode_Reward/approach_from_above`: 증가 확인
- `Episode_Termination/task_success`: 성공률 증가 기대

#### 다음 단계
- [ ] 새 보상 구조로 학습 진행
- [ ] 로봇이 위에서 접근하는지 play.py로 확인
- [ ] task_success 비율 모니터링
