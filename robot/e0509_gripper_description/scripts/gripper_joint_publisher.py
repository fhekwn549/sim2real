#!/usr/bin/env python3
"""
RViz 시각화용 그리퍼 조인트 상태 발행 노드

역할:
    - /dsr01/gripper/stroke 토픽을 구독하여 stroke 값 수신
    - stroke 값을 조인트 각도로 변환하여 joint_states 발행
    - RViz에서 그리퍼 움직임 시각화

Note:
    그리퍼 제어는 gripper_service_node.py가 담당합니다.
    이 노드는 시각화만 담당합니다.
"""
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Int32


class GripperJointPublisher(Node):
    def __init__(self):
        super().__init__('gripper_joint_publisher')

        # Publisher for joint states
        self.publisher = self.create_publisher(JointState, 'joint_states', 10)
        self.timer = self.create_timer(0.02, self.publish_joint_states)  # 50Hz

        # Gripper joint names
        self.joint_names = [
            'gripper_rh_r1',
            'gripper_rh_r2',
            'gripper_rh_l1',
            'gripper_rh_l2'
        ]

        # Gripper position control
        # stroke: 0 = open, 700 = fully closed (real gripper)
        # joint angle: 0.0 = open, ~1.0 rad = closed
        self.stroke = 0  # Current stroke value (0~700)
        self.target_stroke = 0
        self.stroke_speed = 50  # stroke units per cycle (faster for visualization)

        # Stroke to joint angle conversion
        # stroke 700 → ~1.0 rad
        self.stroke_to_rad = 1.0 / 700.0

        # Stroke 토픽 구독 (gripper_service_node가 발행)
        self.stroke_sub = self.create_subscription(
            Int32, 'gripper/stroke', self.stroke_callback, 10)

        self.get_logger().info('========================================')
        self.get_logger().info('Gripper Joint Publisher Ready!')
        self.get_logger().info('----------------------------------------')
        self.get_logger().info('Subscribing to:')
        self.get_logger().info('  gripper/stroke - Int32 (0~700)')
        self.get_logger().info('Publishing to:')
        self.get_logger().info('  joint_states - Gripper joint angles')
        self.get_logger().info('========================================')

    def stroke_callback(self, msg):
        """토픽으로 stroke 값 수신"""
        stroke = max(0, min(700, msg.data))
        self.target_stroke = stroke
        self.get_logger().debug(f'Received stroke: {stroke}')

    def publish_joint_states(self):
        # Smooth movement towards target
        if abs(self.stroke - self.target_stroke) > 1:
            if self.stroke < self.target_stroke:
                self.stroke = min(self.stroke + self.stroke_speed, self.target_stroke)
            else:
                self.stroke = max(self.stroke - self.stroke_speed, self.target_stroke)
        else:
            self.stroke = self.target_stroke

        # Convert stroke to joint angle
        joint_angle = self.stroke * self.stroke_to_rad

        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = self.joint_names
        msg.position = [joint_angle] * 4
        msg.velocity = [0.0] * 4
        msg.effort = [0.0] * 4
        self.publisher.publish(msg)


def main():
    rclpy.init()
    node = GripperJointPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
