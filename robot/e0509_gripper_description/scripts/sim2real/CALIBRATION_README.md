# Hand-Eye Calibration 작업 현황

## 현재 상태
- **캘리브레이션 완료**: 26개 데이터 수집, TSAI 알고리즘 적용
- **검증 대기 중**: verify_calibration.py 실행 필요

## 캘리브레이션 결과
```
카메라 위치 (로봇 베이스 기준):
  x: 548.2 mm
  y: -34.3 mm
  z: 581.4 mm

실제 측정값 (줄자):
  x: ~670 mm
  y: ~0 mm
  z: ~690 mm

예상 오차: ~167mm
```

---

## 다음 단계: 검증 스크립트 실행

### 1. 로봇 bringup 실행 (터미널 1)
```bash
ros2 launch e0509_gripper_description bringup.launch.py mode:=real host:=<로봇IP>
```

### 2. 그리퍼로 체커보드 잡기
```bash
# 그리퍼 열기
ros2 service call /dsr01/gripper/open std_srvs/srv/Trigger

# 체커보드 위치시킨 후 닫기
ros2 service call /dsr01/gripper/close std_srvs/srv/Trigger
```

### 3. 검증 스크립트 실행 (터미널 2)
```bash
cd ~/doosan_ws/src/e0509_gripper_description/scripts/sim2real
python3 verify_calibration.py
```

### 4. 검증 방법
- 로봇을 여러 자세로 움직이면서 **'s' 키**로 검증
- 최소 5-10개 자세에서 테스트 권장
- 평균 오차 확인

### 5. 조작 키
| 키 | 기능 |
|----|------|
| s | 현재 자세에서 검증 |
| r | 검증 결과 초기화 |
| q | 종료 |

---

## 오차가 클 경우 (50mm 이상)

### 방법 1: 데이터 추가 수집
```bash
python3 manual_hand_eye_calibration.py
```
- 기존 데이터에 추가로 수집 가능 (y 선택)
- 40-50개까지 늘리면 정확도 향상

### 방법 2: 재캘리브레이션
```bash
python3 manual_hand_eye_calibration.py
```
- 'r' 키로 초기화 후 새로 수집
- 더 다양한 각도에서 촬영

---

## 파일 구조
```
scripts/sim2real/
├── manual_hand_eye_calibration.py  # 수동 캘리브레이션 (수집 + 계산)
├── verify_calibration.py           # 검증 스크립트
├── auto_hand_eye_calibration.py    # 자동 캘리브레이션 (사용 안 함)
├── calibration_data.npz            # 수집된 데이터 (26개)
├── calibration_result.npz          # 최종 캘리브레이션 결과
├── calibration_images/             # 캘리브레이션 이미지들
└── CALIBRATION_README.md           # 이 파일
```

---

## 로봇 조인트 이동 명령어
```bash
# 형식
ros2 service call /dsr01/motion/move_joint dsr_msgs2/srv/MoveJoint "{pos: [J1, J2, J3, J4, J5, J6], vel: 속도, acc: 가속도}"

# 예시: 홈 위치
ros2 service call /dsr01/motion/move_joint dsr_msgs2/srv/MoveJoint "{pos: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], vel: 20.0, acc: 20.0}"

# 마지막 작업 위치
ros2 service call /dsr01/motion/move_joint dsr_msgs2/srv/MoveJoint "{pos: [-45.0, 30.0, 60.0, 45.0, 100.0, 30.0], vel: 20.0, acc: 20.0}"
```

---

## 체커보드 정보
- 크기: 6x9 내부 코너
- 한 칸 크기: 25mm
- 재질: 포맥스 판에 부착

---

## 주의사항
- 로봇 이동 시 주변 충돌 주의 (카메라 프레임과 충돌 이력 있음)
- 체커보드가 카메라에 완전히 보여야 함
- 's' 키는 체커보드가 인식된 상태에서만 동작
