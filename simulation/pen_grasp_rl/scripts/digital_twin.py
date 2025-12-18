#!/usr/bin/env python3
"""
Isaac Sim Digital Twin for Doosan E0509 + RH-P12-RN-A Gripper

실제 로봇의 joint_states를 Isaac Sim에서 실시간 동기화합니다.
ROS2와 Isaac Sim의 Python 버전이 다르므로 별도 프로세스로 통신합니다.

사용법:
    # 터미널 1: 로봇 실행
    ros2 launch e0509_gripper_description bringup.launch.py mode:=virtual

    # 터미널 2: ROS2 Bridge 실행 (ROS2 환경)
    source /opt/ros/humble/setup.bash
    cd ~/IsaacLab/pen_grasp_rl/scripts
    python3 digital_twin_bridge.py

    # 터미널 3: Isaac Sim 디지털 트윈 (Isaac Sim 환경)
    source ~/isaacsim_env/bin/activate
    cd ~/IsaacLab
    python pen_grasp_rl/scripts/digital_twin.py
"""

import argparse
import json
import numpy as np
import os
import time

# Isaac Sim 초기화 (다른 import 전에 반드시 먼저!)
from isaacsim import SimulationApp

CONFIG = {
    "headless": False,
    "width": 1280,
    "height": 720,
    "title": "Doosan E0509 Digital Twin",
}

simulation_app = SimulationApp(CONFIG)

# Isaac Sim 5.x API
from isaacsim.core.api import World
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.prims import SingleArticulation

# 공유 파일 경로 (digital_twin_bridge.py와 동일)
SHARED_FILE = '/tmp/doosan_joint_states.json'


class JointStateReader:
    """파일에서 joint states 읽기"""

    def __init__(self):
        self._last_data = None
        self._last_mtime = 0

    def read(self):
        """파일에서 joint states 읽기"""
        if not os.path.exists(SHARED_FILE):
            return None

        try:
            # 파일 수정 시간 확인 (불필요한 읽기 방지)
            mtime = os.path.getmtime(SHARED_FILE)
            if mtime == self._last_mtime:
                return self._last_data

            with open(SHARED_FILE, 'r') as f:
                data = json.load(f)

            self._last_data = data
            self._last_mtime = mtime
            return data

        except (json.JSONDecodeError, IOError):
            return self._last_data

    def get_joint_positions(self):
        """Joint positions dict 반환"""
        data = self.read()
        if data is None:
            return {}

        positions = {}
        for i, name in enumerate(data.get('names', [])):
            if i < len(data.get('positions', [])):
                positions[name] = data['positions'][i]
        return positions

    def has_data(self):
        """데이터 존재 여부"""
        return os.path.exists(SHARED_FILE)


class DigitalTwinApp:
    """Isaac Sim 디지털 트윈 애플리케이션"""

    def __init__(self, usd_path=None):
        # USD 경로 설정 (그리퍼 자세 수정된 버전)
        if usd_path is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            # 수정된 USD 파일 사용 (그리퍼 90도 회전 수정)
            usd_path = os.path.join(script_dir, '..', '..', 'e0509_gripper_isaac', 'e0509_gripper_isaac.usd')
        self.usd_path = os.path.abspath(usd_path)

        # Joint state reader
        self.reader = JointStateReader()

        # Isaac Sim
        self.world = None
        self.robot = None

    def setup_scene(self):
        """씬 설정"""
        print(f"Loading USD: {self.usd_path}")

        # World 생성
        self.world = World(stage_units_in_meters=1.0)

        # 로봇 USD 추가
        add_reference_to_stage(
            usd_path=self.usd_path,
            prim_path="/World/Robot"
        )

        # Ground plane 추가
        self.world.scene.add_default_ground_plane()

        # Robot articulation 가져오기
        self.robot = self.world.scene.add(
            SingleArticulation(
                prim_path="/World/Robot",
                name="e0509_gripper"
            )
        )

        # Physics 초기화
        self.world.reset()

        # 디지털 트윈: 높은 stiffness로 position control
        from pxr import UsdPhysics, Usd
        stage = self.world.stage

        # 중력 비활성화
        physics_scene = stage.GetPrimAtPath("/World/physicsScene")
        if physics_scene.IsValid():
            gravity_attr = physics_scene.GetAttribute("physics:gravityMagnitude")
            if gravity_attr:
                gravity_attr.Set(0.0)
                print("  Gravity disabled")

        # 모든 joint에 높은 stiffness 설정
        root = stage.GetPrimAtPath("/World/Robot")
        joint_count = 0
        for prim in Usd.PrimRange(root):
            if prim.IsA(UsdPhysics.RevoluteJoint):
                drive = UsdPhysics.DriveAPI.Get(prim, "angular")
                if not drive:
                    drive = UsdPhysics.DriveAPI.Apply(prim, "angular")
                if drive:
                    # 높은 stiffness + 적절한 damping (빠른 반응 + 진동 억제)
                    drive.GetStiffnessAttr().Set(1e9)
                    drive.GetDampingAttr().Set(1e6)  # 낮춰서 빠른 반응
                    drive.GetMaxForceAttr().Set(1e12)
                    joint_count += 1
        print(f"  {joint_count} joints configured with high stiffness")

        # ArticulationAction 사용을 위한 플래그
        self.use_action = True
        print("Scene setup complete!")
        print(f"Robot DOF: {self.robot.num_dof}")
        print(f"Joint names: {self.robot.dof_names}")

    def update_robot_state(self):
        """파일에서 읽은 joint state로 로봇 업데이트"""
        joint_positions = self.reader.get_joint_positions()

        if not joint_positions:
            return

        # USD joint 순서에 맞게 배열 생성
        positions = []
        for joint_name in self.robot.dof_names:
            if joint_name in joint_positions:
                pos = joint_positions[joint_name]
                positions.append(pos)
            else:
                positions.append(0.0)

        # ArticulationAction으로 position target 설정
        if positions:
            from isaacsim.core.utils.types import ArticulationAction
            pos_array = np.array(positions)
            action = ArticulationAction(joint_positions=pos_array)
            self.robot.apply_action(action)

    def run(self):
        """메인 루프"""
        print("\n" + "=" * 60)
        print("  Isaac Sim Digital Twin - Doosan E0509 + Gripper")
        print("=" * 60)
        print(f"  USD Path: {self.usd_path}")
        print(f"  Data Source: {SHARED_FILE}")
        print("=" * 60)
        print("  Waiting for joint states from bridge...")
        print("  (Run digital_twin_bridge.py in another terminal)")
        print("  Ctrl+C or close window to exit")
        print("=" * 60 + "\n")

        self.setup_scene()

        # joint_states 대기
        wait_count = 0
        while not self.reader.has_data() and simulation_app.is_running():
            self.world.step(render=True)
            wait_count += 1
            if wait_count % 100 == 0:
                print(f"  Waiting for {SHARED_FILE}...")

        if self.reader.has_data():
            print("  Joint states received! Starting sync...")

        # 메인 루프
        while simulation_app.is_running():
            # 파일 → Isaac Sim 동기화
            self.update_robot_state()

            # Physics + Render
            self.world.step(render=True)

        self.cleanup()

    def cleanup(self):
        """정리"""
        print("\nShutting down...")
        simulation_app.close()


def main():
    parser = argparse.ArgumentParser(description='Isaac Sim Digital Twin')
    parser.add_argument('--usd', type=str, default=None,
                        help='USD file path (default: models/first_control.usd)')
    args = parser.parse_args()

    app = DigitalTwinApp(usd_path=args.usd)
    app.run()


if __name__ == '__main__':
    main()
