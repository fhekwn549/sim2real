#!/usr/bin/env python3
"""
IK 기반 TCP 이동 테스트

두산 로봇의 MoveLine 서비스를 사용하여 TCP 기준으로 이동합니다.
강화학습 없이 직접 목표 위치로 이동하는 방식입니다.

사용법:
    # 터미널 1: 로봇 bringup
    ros2 launch dsr_bringup2 dsr_bringup2_gazebo.launch.py (시뮬) 또는
    ros2 launch e0509_gripper_description bringup_dsr.launch.py (실제)

    # 터미널 2: 이 스크립트 실행
    python3 test_ik_move.py

동작:
    1. 현재 TCP 위치 확인
    2. TCP 기준 Z축 5cm 위로 이동
    3. 잠시 대기
    4. 원래 위치로 복귀
"""

import rclpy
from rclpy.node import Node
from dsr_msgs2.srv import MoveLine, MoveJoint, GetCurrentPosx, GetCurrentPosj
import time
import argparse


class IKMoveTest(Node):
    """IK 기반 TCP 이동 테스트 노드"""

    HOME_JOINT_DEG = [0.0, 0.0, 90.0, 0.0, 90.0, 0.0]

    def __init__(self, namespace='dsr01'):
        super().__init__('ik_move_test')
        self.namespace = namespace

        prefix = f'/{namespace}'

        # 서비스 클라이언트
        self.cli_move_line = self.create_client(MoveLine, f'{prefix}/motion/move_line')
        self.cli_move_joint = self.create_client(MoveJoint, f'{prefix}/motion/move_joint')
        self.cli_get_posx = self.create_client(GetCurrentPosx, f'{prefix}/aux_control/get_current_posx')
        self.cli_get_posj = self.create_client(GetCurrentPosj, f'{prefix}/aux_control/get_current_posj')

        self.get_logger().info(f'IK Move Test 노드 시작 (namespace: {namespace})')

    def wait_for_services(self, timeout=10.0) -> bool:
        """서비스 연결 대기"""
        services = [
            (self.cli_move_line, 'move_line'),
            (self.cli_move_joint, 'move_joint'),
            (self.cli_get_posx, 'get_current_posx'),
            (self.cli_get_posj, 'get_current_posj'),
        ]

        for client, name in services:
            self.get_logger().info(f'서비스 대기 중: {name}')
            if not client.wait_for_service(timeout_sec=timeout):
                self.get_logger().error(f'서비스 연결 실패: {name}')
                return False

        self.get_logger().info('모든 서비스 연결됨!')
        return True

    def _call_service(self, client, request, timeout=5.0):
        """동기 서비스 호출"""
        future = client.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=timeout)
        return future.result()

    def get_current_tcp(self) -> tuple:
        """현재 TCP 위치 반환 (x, y, z, rx, ry, rz) in mm, deg"""
        req = GetCurrentPosx.Request()
        req.ref = 0  # DR_BASE
        result = self._call_service(self.cli_get_posx, req)
        if result and result.success and len(result.task_pos_info) > 0:
            pos = result.task_pos_info[0].data
            return tuple(pos[:6])
        return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    def get_current_joint(self) -> tuple:
        """현재 관절 위치 반환 (deg)"""
        req = GetCurrentPosj.Request()
        result = self._call_service(self.cli_get_posj, req)
        if result and result.success:
            return tuple(result.pos)
        return (0.0,) * 6

    def move_to_home(self, vel=30.0, acc=30.0):
        """Home 위치로 이동"""
        self.get_logger().info('Home 위치로 이동 중...')
        req = MoveJoint.Request()
        req.pos = self.HOME_JOINT_DEG
        req.vel = vel
        req.acc = acc
        req.time = 0.0
        req.radius = 0.0
        req.mode = 0
        req.blend_type = 0
        req.sync_type = 0  # SYNC - 이동 완료까지 대기

        result = self._call_service(self.cli_move_joint, req, timeout=30.0)
        if result and result.success:
            self.get_logger().info('Home 도착!')
            return True
        self.get_logger().error('Home 이동 실패')
        return False

    def move_tcp_relative(self, dx=0.0, dy=0.0, dz=0.0, drx=0.0, dry=0.0, drz=0.0,
                          vel=50.0, acc=50.0, ref='tool'):
        """
        TCP 상대 이동

        Args:
            dx, dy, dz: 위치 변화량 (mm)
            drx, dry, drz: 회전 변화량 (deg)
            vel: 속도 (mm/sec)
            acc: 가속도 (mm/sec2)
            ref: 'tool' (TCP 기준) 또는 'base' (베이스 기준)
        """
        ref_code = 1 if ref == 'tool' else 0  # DR_TOOL=1, DR_BASE=0

        self.get_logger().info(
            f'TCP 상대 이동: [{dx:.1f}, {dy:.1f}, {dz:.1f}] mm, '
            f'회전: [{drx:.1f}, {dry:.1f}, {drz:.1f}] deg ({ref} 기준)'
        )

        req = MoveLine.Request()
        req.pos = [dx, dy, dz, drx, dry, drz]
        req.vel = [vel, vel]  # [mm/sec, deg/sec]
        req.acc = [acc, acc]  # [mm/sec2, deg/sec2]
        req.time = 0.0
        req.radius = 0.0
        req.ref = ref_code
        req.mode = 1  # DR_MV_MOD_REL - 상대 이동
        req.blend_type = 0
        req.sync_type = 0  # SYNC - 이동 완료까지 대기

        result = self._call_service(self.cli_move_line, req, timeout=30.0)
        if result and result.success:
            self.get_logger().info('이동 완료!')
            return True
        self.get_logger().error('이동 실패')
        return False

    def move_tcp_absolute(self, x, y, z, rx, ry, rz, vel=50.0, acc=50.0):
        """
        TCP 절대 이동 (베이스 좌표계)

        Args:
            x, y, z: 목표 위치 (mm)
            rx, ry, rz: 목표 회전 (deg)
        """
        self.get_logger().info(
            f'TCP 절대 이동: [{x:.1f}, {y:.1f}, {z:.1f}] mm, '
            f'회전: [{rx:.1f}, {ry:.1f}, {rz:.1f}] deg'
        )

        req = MoveLine.Request()
        req.pos = [x, y, z, rx, ry, rz]
        req.vel = [vel, vel]
        req.acc = [acc, acc]
        req.time = 0.0
        req.radius = 0.0
        req.ref = 0  # DR_BASE
        req.mode = 0  # DR_MV_MOD_ABS - 절대 이동
        req.blend_type = 0
        req.sync_type = 0  # SYNC

        result = self._call_service(self.cli_move_line, req, timeout=30.0)
        if result and result.success:
            self.get_logger().info('이동 완료!')
            return True
        self.get_logger().error('이동 실패')
        return False


def main():
    parser = argparse.ArgumentParser(description='IK 기반 TCP 이동 테스트')
    parser.add_argument('--namespace', '-n', default='dsr01', help='로봇 namespace')
    parser.add_argument('--height', '-z', type=float, default=50.0, help='Z축 이동 거리 (mm)')
    args = parser.parse_args()

    rclpy.init()
    node = IKMoveTest(namespace=args.namespace)

    print('=' * 60)
    print('  IK 기반 TCP 이동 테스트')
    print('=' * 60)
    print(f'  목표: TCP 기준 Z축 {args.height}mm 위로 이동 후 복귀')
    print('=' * 60)

    if not node.wait_for_services():
        print('\n서비스 연결 실패!')
        print('로봇 bringup이 실행 중인지 확인하세요.')
        rclpy.shutdown()
        return

    try:
        # 1. 현재 위치 확인
        tcp_pos = node.get_current_tcp()
        joint_pos = node.get_current_joint()
        print(f'\n현재 TCP: [{tcp_pos[0]:.2f}, {tcp_pos[1]:.2f}, {tcp_pos[2]:.2f}] mm')
        print(f'현재 Joint: [{", ".join(f"{j:.1f}" for j in joint_pos)}] deg')

        # 2. Home으로 이동
        input('\nEnter를 누르면 Home 위치로 이동합니다...')
        node.move_to_home()

        time.sleep(1.0)
        tcp_home = node.get_current_tcp()
        print(f'\nHome TCP: [{tcp_home[0]:.2f}, {tcp_home[1]:.2f}, {tcp_home[2]:.2f}] mm')

        # 3. TCP 기준 Z축 위로 이동
        input(f'\nEnter를 누르면 TCP 기준 Z축 {args.height}mm 위로 이동합니다...')
        node.move_tcp_relative(dz=args.height, ref='tool')

        time.sleep(0.5)
        tcp_up = node.get_current_tcp()
        print(f'\n이동 후 TCP: [{tcp_up[0]:.2f}, {tcp_up[1]:.2f}, {tcp_up[2]:.2f}] mm')
        print(f'Z축 변화: {tcp_up[2] - tcp_home[2]:.2f} mm')

        # 4. 잠시 대기
        input('\nEnter를 누르면 원래 위치로 복귀합니다...')

        # 5. 원래 위치로 복귀 (역방향 이동)
        node.move_tcp_relative(dz=-args.height, ref='tool')

        time.sleep(0.5)
        tcp_back = node.get_current_tcp()
        print(f'\n복귀 후 TCP: [{tcp_back[0]:.2f}, {tcp_back[1]:.2f}, {tcp_back[2]:.2f}] mm')

        # 6. 결과 비교
        print('\n' + '=' * 60)
        print('  결과 비교')
        print('=' * 60)
        print(f'  Home TCP:     [{tcp_home[0]:.2f}, {tcp_home[1]:.2f}, {tcp_home[2]:.2f}]')
        print(f'  이동 후 TCP:  [{tcp_up[0]:.2f}, {tcp_up[1]:.2f}, {tcp_up[2]:.2f}]')
        print(f'  복귀 후 TCP:  [{tcp_back[0]:.2f}, {tcp_back[1]:.2f}, {tcp_back[2]:.2f}]')
        print('=' * 60)

        error = ((tcp_back[0] - tcp_home[0])**2 +
                 (tcp_back[1] - tcp_home[1])**2 +
                 (tcp_back[2] - tcp_home[2])**2) ** 0.5
        print(f'  복귀 오차: {error:.2f} mm')
        print('=' * 60)

    except KeyboardInterrupt:
        print('\n\n사용자 중단')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
