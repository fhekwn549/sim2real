#!/usr/bin/env python3
"""
DRFL 기반 RH-P12-RN-A 그리퍼 인터페이스

Doosan 로봇의 Tool Flange Serial을 통해 Modbus RTU로 그리퍼를 제어합니다.
DRFL의 flange_serial_* 함수를 사용하여 직접 통신합니다.

Usage:
    from gripper_interface import Gripper

    gripper = Gripper(robot)  # robot은 연결된 DoosanRobot 인스턴스
    gripper.initialize()

    gripper.open()           # 열기 (position=0)
    gripper.close()          # 닫기 (position=700)
    gripper.set_position(350)  # 특정 위치 (0~700)
"""

import time
import numpy as np
from typing import Optional

# DRFL import 시도
try:
    import DRFL
    DRFL_AVAILABLE = True
except ImportError:
    DRFL_AVAILABLE = False


class ModbusRTU:
    """Modbus RTU 프레임 생성 유틸리티"""

    SLAVE_ID = 1
    REG_TORQUE_ENABLE = 256   # 0x0100
    REG_GOAL_POSITION = 282   # 0x011A
    REG_GOAL_CURRENT = 275    # 0x0113

    @staticmethod
    def crc16(data: bytes) -> int:
        """Modbus CRC16 계산"""
        crc = 0xFFFF
        for byte in data:
            crc ^= byte
            for _ in range(8):
                if crc & 0x0001:
                    crc = (crc >> 1) ^ 0xA001
                else:
                    crc >>= 1
        return crc

    @classmethod
    def make_frame(cls, data: bytes) -> list:
        """CRC 추가하여 list[int]로 반환"""
        crc = cls.crc16(data)
        frame = data + bytes([crc & 0xFF, (crc >> 8) & 0xFF])
        return list(frame)

    @classmethod
    def fc06_write_single(cls, register: int, value: int) -> list:
        """FC06: Write Single Register"""
        data = bytes([
            cls.SLAVE_ID, 0x06,
            (register >> 8) & 0xFF, register & 0xFF,
            (value >> 8) & 0xFF, value & 0xFF
        ])
        return cls.make_frame(data)

    @classmethod
    def fc06_torque_enable(cls, enable: bool = True) -> list:
        """토크 활성화/비활성화"""
        return cls.fc06_write_single(cls.REG_TORQUE_ENABLE, 1 if enable else 0)

    @classmethod
    def fc16_position(cls, position: int) -> list:
        """FC16: 위치 설정 (0~700)"""
        position = max(0, min(700, position))
        data = bytes([
            cls.SLAVE_ID, 0x10,      # FC16: Write Multiple Registers
            0x01, 0x1A,              # Start register 282 (0x011A)
            0x00, 0x02,              # Register count: 2
            0x04,                    # Byte count: 4
            (position >> 8) & 0xFF, position & 0xFF,  # Position value
            0x00, 0x00               # Reserved
        ])
        return cls.make_frame(data)

    @classmethod
    def fc16_current(cls, current: int = 400) -> list:
        """FC16: 전류 제한 설정"""
        current = max(0, min(820, current))
        data = bytes([
            cls.SLAVE_ID, 0x10,
            0x01, 0x13,              # Start register 275 (0x0113)
            0x00, 0x02,
            0x04,
            (current >> 8) & 0xFF, current & 0xFF,
            0x00, 0x00
        ])
        return cls.make_frame(data)


class Gripper:
    """RH-P12-RN-A 그리퍼 인터페이스 (DRFL 기반)"""

    # 시리얼 설정
    SERIAL_PORT = 1
    BAUDRATE = 57600
    BYTESIZE = 8
    PARITY = 0  # None
    STOPBITS = 1

    # 그리퍼 범위
    POSITION_MIN = 0    # 완전 열림
    POSITION_MAX = 700  # 완전 닫힘

    def __init__(self, simulation: bool = False):
        """
        Args:
            simulation: True면 DRFL 없이 시뮬레이션 모드
        """
        self.simulation = simulation or not DRFL_AVAILABLE
        self.serial_open = False
        self._current_position = 0

        if self.simulation:
            print("[Gripper] Simulation mode")
        else:
            print("[Gripper] DRFL mode")

    def initialize(self) -> bool:
        """그리퍼 초기화 (시리얼 열기 + 토크 활성화)"""
        if not self._open_serial():
            return False

        time.sleep(0.1)

        # 토크 활성화
        if not self._send_frame(ModbusRTU.fc06_torque_enable(True)):
            print("[Gripper] Torque enable failed")
            return False

        time.sleep(0.1)
        print("[Gripper] Initialized")
        return True

    def shutdown(self):
        """그리퍼 종료 (시리얼 닫기)"""
        self._close_serial()
        print("[Gripper] Shutdown")

    def open(self, wait: bool = True) -> bool:
        """그리퍼 열기"""
        print("[Gripper] Opening...")
        result = self.set_position(self.POSITION_MIN)
        if wait and result:
            time.sleep(1.0)  # 동작 완료 대기
        return result

    def close(self, wait: bool = True) -> bool:
        """그리퍼 닫기"""
        print("[Gripper] Closing...")
        result = self.set_position(self.POSITION_MAX)
        if wait and result:
            time.sleep(1.0)
        return result

    def set_position(self, position: int) -> bool:
        """
        그리퍼 위치 설정

        Args:
            position: 0 (완전 열림) ~ 700 (완전 닫힘)
        """
        position = max(self.POSITION_MIN, min(self.POSITION_MAX, position))

        if not self._send_frame(ModbusRTU.fc16_position(position)):
            print(f"[Gripper] Set position {position} failed")
            return False

        self._current_position = position
        print(f"[Gripper] Position set to {position}")
        return True

    def set_current(self, current: int = 400) -> bool:
        """
        그리퍼 전류 제한 설정

        Args:
            current: 전류 제한 (0~820, 기본값 400)
        """
        if not self._send_frame(ModbusRTU.fc16_current(current)):
            print(f"[Gripper] Set current {current} failed")
            return False

        print(f"[Gripper] Current limit set to {current}")
        return True

    def get_position(self) -> int:
        """현재 위치 반환 (시뮬레이션에서는 마지막 설정값)"""
        return self._current_position

    def get_normalized_position(self) -> float:
        """정규화된 위치 반환 (0.0=열림, 1.0=닫힘)"""
        return self._current_position / self.POSITION_MAX

    # =========================================================================
    # Private 메서드
    # =========================================================================

    def _open_serial(self) -> bool:
        """시리얼 포트 열기"""
        if self.simulation:
            self.serial_open = True
            print("[Gripper] Serial opened (simulation)")
            return True

        try:
            DRFL.flange_serial_open(
                self.SERIAL_PORT,
                self.BAUDRATE,
                self.BYTESIZE,
                self.PARITY,
                self.STOPBITS
            )
            self.serial_open = True
            print("[Gripper] Serial opened")
            return True
        except Exception as e:
            print(f"[Gripper] Serial open failed: {e}")
            return False

    def _close_serial(self):
        """시리얼 포트 닫기"""
        if self.simulation:
            self.serial_open = False
            return

        try:
            DRFL.flange_serial_close(self.SERIAL_PORT)
        except:
            pass
        self.serial_open = False

    def _send_frame(self, frame: list) -> bool:
        """Modbus 프레임 전송"""
        if self.simulation:
            print(f"[Gripper] Send: {bytes(frame).hex()}")
            return True

        if not self.serial_open:
            print("[Gripper] Serial not open")
            return False

        try:
            DRFL.flange_serial_write(self.SERIAL_PORT, frame)
            time.sleep(0.05)  # 응답 대기
            return True
        except Exception as e:
            print(f"[Gripper] Send failed: {e}")
            return False


# =============================================================================
# 테스트
# =============================================================================
def test_gripper():
    """그리퍼 테스트 (시뮬레이션 모드)"""
    print("=" * 60)
    print("Gripper 테스트 (Simulation Mode)")
    print("=" * 60)

    gripper = Gripper(simulation=True)

    if not gripper.initialize():
        print("초기화 실패")
        return

    print("\n열기 테스트...")
    gripper.open(wait=False)

    print("\n닫기 테스트...")
    gripper.close(wait=False)

    print("\n위치 설정 테스트...")
    gripper.set_position(350)

    print(f"\n현재 위치: {gripper.get_position()}")
    print(f"정규화 위치: {gripper.get_normalized_position():.2f}")

    gripper.shutdown()
    print("\n테스트 완료!")


if __name__ == "__main__":
    test_gripper()
