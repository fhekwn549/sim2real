"""
Sim-to-Real Transfer Module for Doosan E0509 + RH-P12-RN-A Gripper

이 모듈은 Isaac Lab에서 학습된 정책을 실제 로봇에서 실행하기 위한 도구를 제공합니다.

Components:
    - RobotObservationCollector: 로봇 상태를 Isaac Lab observation 형태로 수집
    - (TODO) PolicyRunner: 학습된 정책 실행
    - (TODO) CameraProcessor: 카메라에서 펜 pose 추정
"""

from .robot_observation import RobotObservationCollector

__all__ = ['RobotObservationCollector']
