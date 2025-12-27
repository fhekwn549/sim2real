#!/usr/bin/env python3
"""
Sim2Real ROS2 브릿지

ROS2와 sim2real 스크립트 간 파일 기반 통신을 담당합니다.
Python 버전 충돌을 해결하기 위해 별도 프로세스로 실행합니다.

역할:
1. 로봇 상태 읽기 → /tmp/sim2real_state.json
2. /tmp/sim2real_command.json → 로봇 명령 전송

사용법 (ROS2 환경에서):
    source /opt/ros/humble/setup.bash
    source ~/doosan_ws/install/setup.bash
    python3 sim2real_bridge.py
"""

import json
import os
import time
import numpy as np

import rclpy
from rclpy.node import Node
from dsr_msgs2.srv import MoveJoint, GetCurrentPosj, GetCurrentPosx
from std_srvs.srv import Trigger

# 펜 위치 서비스 (CoWriteBot)
try:
    from cowritebot_interfaces.srv import GetPenPosition
    PEN_SERVICE_AVAILABLE = True
except ImportError:
    PEN_SERVICE_AVAILABLE = False
    print("[Warning] cowritebot_interfaces 없음. 펜 감지 비활성화.")

# 공유 파일 경로
STATE_FILE = '/tmp/sim2real_state.json'
COMMAND_FILE = '/tmp/sim2real_command.json'


class Sim2RealBridge(Node):
    """Sim2Real ROS2 브릿지 노드"""

    HOME_JOINT_DEG = [0.0, 0.0, 90.0, 0.0, 90.0, 0.0]

    def __init__(self, namespace='dsr01'):
        super().__init__('sim2real_bridge')
        self.namespace = namespace

        # 서비스 클라이언트
        prefix = f'/{namespace}'
        self.cli_move_joint = self.create_client(MoveJoint, f'{prefix}/motion/move_joint')
        self.cli_get_posj = self.create_client(GetCurrentPosj, f'{prefix}/aux_control/get_current_posj')
        self.cli_get_posx = self.create_client(GetCurrentPosx, f'{prefix}/aux_control/get_current_posx')
        self.cli_gripper_open = self.create_client(Trigger, f'{prefix}/gripper/open')
        self.cli_gripper_close = self.create_client(Trigger, f'{prefix}/gripper/close')

        # 펜 감지 서비스 (CoWriteBot find_marker)
        self.cli_pen_position = None
        self._pen_position = None  # 캐시된 펜 위치
        self._pen_update_interval = 0.1  # 100ms
        self._last_pen_update = 0

        if PEN_SERVICE_AVAILABLE:
            self.cli_pen_position = self.create_client(GetPenPosition, 'get_pen_position')
            self.get_logger().info('펜 감지 서비스 활성화')

        # 상태
        self._last_command_time = 0
        self._command_count = 0

        self.get_logger().info(f'Sim2Real Bridge 시작 (namespace: {namespace})')

    def wait_for_services(self, timeout=10.0) -> bool:
        """서비스 연결 대기"""
        services = [
            (self.cli_move_joint, 'move_joint'),
            (self.cli_get_posj, 'get_current_posj'),
            (self.cli_get_posx, 'get_current_posx'),
        ]

        for client, name in services:
            self.get_logger().info(f'서비스 대기: {name}')
            if not client.wait_for_service(timeout_sec=timeout):
                self.get_logger().error(f'서비스 {name} 연결 실패')
                return False

        self.get_logger().info('모든 서비스 연결됨')
        return True

    def _call_service(self, client, request, timeout=2.0):
        """동기 서비스 호출"""
        future = client.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=timeout)
        return future.result()

    # ==================== 로봇 상태 읽기 ====================

    def get_joint_positions_deg(self) -> list:
        """현재 관절 위치 (도)"""
        req = GetCurrentPosj.Request()
        result = self._call_service(self.cli_get_posj, req)
        if result and result.success:
            return list(result.pos)
        return [0.0] * 6

    def get_tcp_position(self) -> list:
        """현재 TCP 위치 (mm)"""
        req = GetCurrentPosx.Request()
        req.ref = 0
        result = self._call_service(self.cli_get_posx, req)
        if result and result.success and len(result.task_pos_info) > 0:
            return list(result.task_pos_info[0].data[:3])
        return [0.0, 0.0, 0.0]

    def get_tcp_pose(self) -> tuple:
        """현재 TCP pose (위치 mm, 회전 deg)"""
        req = GetCurrentPosx.Request()
        req.ref = 0
        result = self._call_service(self.cli_get_posx, req)
        if result and result.success and len(result.task_pos_info) > 0:
            data = result.task_pos_info[0].data
            pos = list(data[:3])   # x, y, z (mm)
            rot = list(data[3:6])  # rx, ry, rz (deg)
            return pos, rot
        return [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]

    def get_pen_position(self) -> dict:
        """펜 위치 가져오기 (캐시 사용)"""
        if self.cli_pen_position is None:
            return None

        # 업데이트 간격 체크
        now = time.time()
        if now - self._last_pen_update < self._pen_update_interval:
            return self._pen_position

        # 서비스 호출
        if not self.cli_pen_position.service_is_ready():
            return self._pen_position

        try:
            req = GetPenPosition.Request()
            result = self._call_service(self.cli_pen_position, req, timeout=0.5)

            if result and result.success:
                # mm → m 변환
                self._pen_position = {
                    'cap_end_3d': [x / 1000.0 for x in result.cap_end_3d],
                    'tip_end_3d': [x / 1000.0 for x in result.tip_end_3d],
                    'center': list(result.center),
                    'angle': result.angle,
                    'detected': True,
                }
                self._last_pen_update = now
        except Exception as e:
            self.get_logger().warn(f'펜 감지 실패: {e}')

        return self._pen_position

    def write_state(self):
        """로봇 상태를 파일에 저장"""
        joint_pos_deg = self.get_joint_positions_deg()
        tcp_pos_mm, tcp_rot_deg = self.get_tcp_pose()

        state = {
            'timestamp': time.time(),
            'joint_pos_deg': joint_pos_deg,
            'joint_pos_rad': [np.radians(d) for d in joint_pos_deg],
            'tcp_pos_mm': tcp_pos_mm,
            'tcp_pos_m': [p / 1000.0 for p in tcp_pos_mm],
            'tcp_rot_deg': tcp_rot_deg,
            'tcp_rot_rad': [np.radians(d) for d in tcp_rot_deg],
        }

        # 펜 위치 추가
        pen_info = self.get_pen_position()
        if pen_info:
            state['pen'] = pen_info

        try:
            with open(STATE_FILE, 'w') as f:
                json.dump(state, f)
        except Exception as e:
            self.get_logger().error(f'상태 저장 실패: {e}')

    # ==================== 로봇 명령 처리 ====================

    def read_command(self) -> dict:
        """명령 파일 읽기"""
        if not os.path.exists(COMMAND_FILE):
            return None

        try:
            mtime = os.path.getmtime(COMMAND_FILE)
            if mtime <= self._last_command_time:
                return None

            with open(COMMAND_FILE, 'r') as f:
                command = json.load(f)

            self._last_command_time = mtime
            return command

        except (json.JSONDecodeError, IOError):
            return None

    def execute_command(self, command: dict):
        """명령 실행"""
        cmd_type = command.get('type', '')

        if cmd_type == 'move_joint':
            target_deg = command.get('target_deg', [])
            vel = command.get('vel', 30)
            acc = command.get('acc', 30)

            req = MoveJoint.Request()
            req.pos = [float(d) for d in target_deg]  # float 변환 필수
            req.vel = float(vel)
            req.acc = float(acc)
            req.time = 0.0
            req.radius = 0.0
            req.mode = 0
            req.blend_type = 0
            req.sync_type = 0

            result = self._call_service(self.cli_move_joint, req)

            self._command_count += 1
            if self._command_count % 30 == 0:
                self.get_logger().info(f'명령 {self._command_count}: joint={[round(d,1) for d in target_deg]}')

        elif cmd_type == 'gripper_open':
            if self.cli_gripper_open.service_is_ready():
                self._call_service(self.cli_gripper_open, Trigger.Request())
                self.get_logger().info('그리퍼 열기')

        elif cmd_type == 'gripper_close':
            if self.cli_gripper_close.service_is_ready():
                self._call_service(self.cli_gripper_close, Trigger.Request())
                self.get_logger().info('그리퍼 닫기')

        elif cmd_type == 'home':
            self.get_logger().info('Home 위치로 이동')
            req = MoveJoint.Request()
            req.pos = self.HOME_JOINT_DEG
            req.vel = 30.0
            req.acc = 30.0
            req.time = 0.0
            req.radius = 0.0
            req.mode = 0
            req.blend_type = 0
            req.sync_type = 0
            self._call_service(self.cli_move_joint, req, timeout=10.0)

    # ==================== 메인 루프 ====================

    def run(self):
        """메인 루프"""
        rate = 30  # Hz
        dt = 1.0 / rate

        self.get_logger().info(f'브릿지 루프 시작 ({rate}Hz)')

        while rclpy.ok():
            loop_start = time.time()

            # 1. 로봇 상태 → 파일
            self.write_state()

            # 2. 파일 → 로봇 명령
            command = self.read_command()
            if command:
                self.execute_command(command)

            # 주기 유지
            elapsed = time.time() - loop_start
            if elapsed < dt:
                time.sleep(dt - elapsed)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--namespace', '-n', default='dsr01')
    args = parser.parse_args()

    rclpy.init()
    node = Sim2RealBridge(namespace=args.namespace)

    print('=' * 60)
    print('  Sim2Real ROS2 Bridge')
    print('=' * 60)
    print(f'  Namespace: {args.namespace}')
    print(f'  State file: {STATE_FILE}')
    print(f'  Command file: {COMMAND_FILE}')
    print('=' * 60)

    if not node.wait_for_services():
        print('서비스 연결 실패. bringup이 실행 중인지 확인하세요.')
        rclpy.shutdown()
        return

    print('  Bridge 시작! Ctrl+C로 종료')
    print('=' * 60)

    try:
        node.run()
    except KeyboardInterrupt:
        print('\n종료 중...')
    finally:
        node.destroy_node()
        rclpy.shutdown()

        # 파일 정리
        for f in [STATE_FILE, COMMAND_FILE]:
            if os.path.exists(f):
                os.remove(f)


if __name__ == '__main__':
    main()
