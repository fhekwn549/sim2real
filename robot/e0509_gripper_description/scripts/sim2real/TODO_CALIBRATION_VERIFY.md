# Eye-in-Hand 캘리브레이션 검증 작업

## 현재 상태
- [x] Eye-in-Hand 캘리브레이션 완료
- [x] 결과 파일 생성됨: `calibration_result.npz`, `T_cam2tcp.npy` 등
- [ ] 검증 작업 필요

## 캘리브레이션 결과 요약
```
T_cam2tcp 행렬 (카메라 → TCP 변환):
[[ 0.033  -0.152  -0.988  -0.034]
 [-0.815   0.568  -0.115   0.017]
 [ 0.579   0.809  -0.105  -0.040]
 [ 0.      0.      0.      1.   ]]

카메라 위치 (TCP 기준):
  X: -33.9 mm
  Y: +17.2 mm
  Z: -39.6 mm
```

---

## 검증 작업 순서

### 1. 터미널 1: 로봇 연결
```bash
ros2 launch e0509_gripper_description bringup.launch.py mode:=real host:=<로봇IP>
```

### 2. 터미널 2: 슬라이더 제어 (선택)
```bash
cd ~/doosan_ws/src/e0509_gripper_description/scripts
python3 robot_slider_control.py
```

### 3. 터미널 3: 검증 실행
```bash
cd ~/doosan_ws/src/e0509_gripper_description/scripts/sim2real
python3 verify_calibration.py
```

---

## 검증 방법

1. **체커보드를 고정 위치에 배치** (테이블 위, 움직이지 않게)

2. **로봇을 여러 자세로 이동**하면서 체커보드가 카메라에 보이게 함

3. 각 자세에서 **'s' 키로 기록**
   - 최소 5~10개 자세 권장
   - 다양한 각도에서 촬영

4. **일관성 확인**
   - 고정된 체커보드이므로 로봇 좌표계에서 항상 같은 위치여야 함
   - 변환 결과의 편차로 캘리브레이션 정확도 판단

---

## 검증 결과 해석

| 최대 편차 | 평가 | 조치 |
|----------|------|------|
| < 10mm | 우수 | 바로 사용 가능 |
| 10~30mm | 양호 | 일반 작업 가능 |
| > 30mm | 부족 | 재캘리브레이션 권장 |

---

## 키 조작 (verify_calibration.py)

| 키 | 기능 |
|---|---|
| s | 현재 자세에서 체커보드 위치 기록 |
| r | 기록 초기화 |
| q | 종료 (최종 결과 출력) |

---

## 문제 발생 시

### 카메라 연결 안 됨
- USB 3.0 포트(파란색)에 연결 확인
- `rs-enumerate-devices` 로 카메라 인식 확인

### 체커보드 인식 안 됨
- 조명 확인 (너무 어둡거나 반사 없도록)
- 체커보드 전체가 화면에 보여야 함
- 거리: 30~80cm 권장

### 편차가 큼
- 캘리브레이션 데이터 수집 시 회전 변화량 확인 (30° 이상 필요)
- 재캘리브레이션: `python3 manual_hand_eye_calibration.py`

---

## 다음 단계 (검증 완료 후)

1. `coordinate_transformer.py` 사용하여 실제 물체 좌표 변환 테스트
2. 펜 인식 + 좌표 변환 → 로봇 이동 파이프라인 구축
