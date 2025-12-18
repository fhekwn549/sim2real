# E0509 Reach 환경 개발 로그

## 개요

펜 잡기 강화학습 프로젝트에서 Isaac Lab의 **reach 예제**를 기반으로 E0509 로봇용 환경을 구현했습니다.

### 배경
- v1~v3 환경: 커스텀 보상 함수로 구현했으나 학습 성공률 저조 (7000 iter에 0%)
- 원인: 성공률 계산 방식 오류, 성공 조건이 너무 엄격함
- 해결: Isaac Lab의 검증된 reach 예제 구조를 그대로 활용

### 핵심 변경사항
1. **목표 단순화**: 펜 z축 정렬 제거 → 위치 접근만 학습
2. **펜 USD 제거**: 목표 포인트만 사용 (RealSense로 실제 잡기 예정)
3. **검증된 보상 함수**: Isaac Lab reach 예제와 동일한 구조

---

## 파일 구조

```
pen_grasp_rl/
├── envs/
│   ├── e0509_reach_env.py    ★ 새로 생성 (Isaac Lab reach 기반)
│   ├── pen_grasp_env.py      # v1 (기존, 유지)
│   ├── pen_grasp_env_v2.py   # v2 (기존, 유지)
│   ├── pen_grasp_env_v3.py   # v3 (기존, 유지)
│   └── __init__.py           # 업데이트됨
├── scripts/
│   ├── train_reach.py        ★ 새로 생성
│   ├── play_reach.py         ★ 새로 생성
│   ├── train.py              # 기존 (v1~v3용)
│   └── play.py               # 기존
└── models/
    └── first_control.usd     # E0509 + RH-P12-RN-A 그리퍼
```

---

## 환경 설정 (E0509ReachEnvCfg)

### 로봇
- **모델**: Doosan E0509 + RH-P12-RN-A 그리퍼
- **엔드이펙터**: `link_6`
- **제어 관절**: `joint_[1-6]` (6DOF)
- **그리퍼**: 열린 상태 고정 (액션 없음)

### 목표 위치
- **범위**:
  - x: 0.3 ~ 0.5m
  - y: -0.2 ~ 0.2m
  - z: 0.2 ~ 0.4m
- **갱신 주기**: 4초마다 새로운 목표 생성
- **시각화**: 목표 위치 마커 표시 (debug_vis=True)

### 보상 함수
| 보상 | 함수 | Weight | 설명 |
|------|------|--------|------|
| 위치 추적 (L2) | `position_command_error` | -0.2 | 목표와의 L2 거리 페널티 |
| 위치 추적 (tanh) | `position_command_error_tanh` | 0.1 | 가까울수록 높은 보상 |
| 액션 변화율 | `action_rate_l2` | -0.0001 | 급격한 액션 변화 페널티 |
| 관절 속도 | `joint_vel_l2` | -0.0001 | 관절 속도 페널티 |

### 관찰 (Observation)
- `joint_pos_rel`: 관절 위치 (상대값, 노이즈 포함)
- `joint_vel_rel`: 관절 속도 (상대값, 노이즈 포함)
- `pose_command`: 목표 위치 명령
- `last_action`: 이전 액션

### 커리큘럼
- 4500 스텝에 걸쳐 `action_rate` 페널티 점진적 증가 (-0.0001 → -0.005)
- 4500 스텝에 걸쳐 `joint_vel` 페널티 점진적 증가 (-0.0001 → -0.001)

---

## 사용법

### 학습
```bash
# 가상환경 활성화
source ~/isaacsim_env/bin/activate
cd ~/IsaacLab

# 학습 실행 (headless)
python pen_grasp_rl/scripts/train_reach.py --headless --num_envs 4096 --max_iterations 5000

# GUI 모드로 학습 (디버깅용)
python pen_grasp_rl/scripts/train_reach.py --num_envs 64
```

### 테스트
```bash
# 학습된 모델 테스트
python pen_grasp_rl/scripts/play_reach.py --checkpoint logs/e0509_reach/model_XXXX.pt

# 환경 수 조절
python pen_grasp_rl/scripts/play_reach.py --checkpoint logs/e0509_reach/model_XXXX.pt --num_envs 16
```

### 체크포인트 이어서 학습
```bash
python pen_grasp_rl/scripts/train_reach.py --headless --num_envs 4096 --checkpoint logs/e0509_reach/model_XXXX.pt
```

---

## 기대 효과

1. **학습 성공률 향상**: Isaac Lab 검증된 구조 사용
2. **단순화된 목표**: 위치 접근만 → 빠른 학습
3. **확장성**: 나중에 eye-in-hand로 실제 잡기 연결 가능

---

## 다음 단계

1. [ ] reach 학습 실행 및 결과 확인
2. [ ] 학습된 정책 테스트
3. [ ] RealSense D455F eye-in-hand 연동
4. [ ] 실제 펜 위치로 목표 대체
5. [ ] 그리퍼 제어 추가 (잡기 동작)

---

## 참고

### Isaac Lab Reach 예제 위치
```
/home/fhekwn549/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/reach/
├── reach_env_cfg.py          # 기본 환경 설정
├── config/
│   ├── franka/               # Franka 로봇 설정
│   └── ur_10/                # UR10 로봇 설정
└── mdp/
    └── rewards.py            # 보상 함수 정의
```

### 폴더 동기화 명령어
```bash
# CoWriteBotRL → IsaacLab
cp /home/fhekwn549/CoWriteBotRL/pen_grasp_rl/envs/e0509_reach_env.py /home/fhekwn549/IsaacLab/pen_grasp_rl/envs/
cp /home/fhekwn549/CoWriteBotRL/pen_grasp_rl/scripts/train_reach.py /home/fhekwn549/IsaacLab/pen_grasp_rl/scripts/
cp /home/fhekwn549/CoWriteBotRL/pen_grasp_rl/scripts/play_reach.py /home/fhekwn549/IsaacLab/pen_grasp_rl/scripts/
```

---

*최종 업데이트: 2024-12-17*
