"""
Pen Grasp RL Package for Isaac Lab

이 패키지는 Isaac Lab 시뮬레이션 환경에서 펜 잡기 강화학습을 수행하기 위한 모듈입니다.

=== 패키지 구성 ===
- envs/: 강화학습 환경 정의
  - pen_grasp_env.py: 펜 잡기 환경 (PenGraspEnv, PenGraspEnvCfg)
- scripts/: 실행 스크립트
  - train.py: 강화학습 훈련 스크립트
  - play.py: 학습된 모델 테스트 스크립트
  - test_env.py: 환경 설정 테스트 스크립트
- models/: USD 모델 파일
  - first_control.usd: Doosan E0509 로봇 + RH-P12-RN-A 그리퍼 모델

=== 프로젝트 목표 ===
1. Doosan E0509 로봇팔이 RH-P12-RN-A 그리퍼로 펜 캡을 잡는 것
2. 잡은 후 글쓰기 자세로 정렬하는 것
"""
