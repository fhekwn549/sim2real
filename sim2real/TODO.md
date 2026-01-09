# TODO - Sim2Real 테스트 및 검증

## OSC 토크 제어 테스트

### 1. 토크 제어 기본 테스트
```bash
cd ~/sim2real/sim2real
python3 test_torque_control.py --test read      # 상태 읽기 테스트
python3 test_torque_control.py --test gravity   # 중력 보상 테스트 (주의!)
```

**확인 사항:**
- [ ] 실시간 상태 읽기 (joint_pos, joint_vel, gravity_torque 등)
- [ ] 중력 보상 시 로봇이 현재 자세 유지하는지
- [ ] 제어 주파수 확인 (목표: 100Hz 이상)

### 2. OSC 컨트롤러 테스트
```bash
python3 test_osc_control.py --test hold     # 현재 위치 유지
python3 test_osc_control.py --test move     # 델타 이동 테스트
```

**확인 사항:**
- [ ] OSC 위치 유지 안정성
- [ ] 위치 델타 명령에 대한 반응
- [ ] 강성/댐핑 파라미터 튜닝 필요 여부

### 3. 통합 스크립트 테스트
```bash
# IK 모드 (안전, 먼저 테스트)
python3 run_sim2real_unified.py --checkpoint /path/to/model.pt --mode ik

# OSC 모드 (IK 테스트 후)
python3 run_sim2real_unified.py --checkpoint /path/to/model.pt --mode osc
```

**확인 사항:**
- [ ] IK 모드에서 기존과 동일하게 동작하는지
- [ ] OSC 모드에서 펜 접근 동작 확인
- [ ] IK vs OSC 성능 비교 (정확도, 속도, 안정성)

---

## 파라미터 튜닝

### OSC 파라미터
| 파라미터 | 현재값 | 범위 | 비고 |
|---------|--------|------|------|
| osc_stiffness | 150 | 50-300 | 높으면 빠르지만 진동 |
| osc_damping_ratio | 1.0 | 0.7-1.5 | 1.0 = 임계 감쇠 |
| freq_osc | 100 Hz | 50-1000 | 높을수록 안정 |

### 안전 파라미터
| 파라미터 | 현재값 | 설명 |
|---------|--------|------|
| safety_min_z | 0.05m | 최소 Z 높이 |
| max_tcp_delta | 0.05m | 스텝당 최대 이동 |

---

## 주의사항

1. **OSC 테스트 전 필수 확인**
   - 비상정지 버튼 준비
   - 로봇 주변 정리
   - 낮은 강성으로 시작 (stiffness=50)

2. **처음 테스트 순서**
   1. `--dry-run` 옵션으로 상태 읽기만 테스트
   2. 짧은 시간(2-3초)으로 중력 보상 테스트
   3. OSC hold 테스트
   4. OSC move 테스트 (작은 델타)
   5. 통합 스크립트 IK 모드
   6. 통합 스크립트 OSC 모드

---

## 예상 이슈

- [ ] Doosan RT 제어 권한 문제
- [ ] Jacobian/Mass matrix 정확도
- [ ] 실시간 제어 주기 지연
- [ ] 토크 제한으로 인한 동작 제한
