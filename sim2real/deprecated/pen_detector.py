#!/usr/bin/env python3
"""
RealSense 펜 인식 모듈

RealSense D455F로 펜 위치를 인식합니다.
coordinate_transformer와 함께 사용하여 로봇 좌표계로 변환합니다.

Usage:
    from pen_detector import PenDetector
    from coordinate_transformer import CoordinateTransformer

    detector = PenDetector()
    transformer = CoordinateTransformer()

    detector.start()

    # 카메라 좌표
    pos_cam = detector.get_pen_position_camera()

    # 로봇 좌표 (TCP pose 필요)
    tcp_pos, tcp_rot = robot.get_tcp_pose()
    pos_robot = transformer.camera_to_robot(pos_cam, tcp_pos, tcp_rot)

    detector.stop()
"""

import numpy as np
import cv2
import time
from typing import Optional, Tuple
from dataclasses import dataclass

try:
    import pyrealsense2 as rs
    REALSENSE_AVAILABLE = True
except ImportError:
    REALSENSE_AVAILABLE = False
    print("[Warning] pyrealsense2 not installed")


@dataclass
class DetectionConfig:
    """인식 설정"""
    width: int = 640
    height: int = 480
    fps: int = 30

    # HSV 색상 필터 (파란색 펜 캡)
    hue_low: int = 100
    hue_high: int = 130
    sat_low: int = 100
    sat_high: int = 255
    val_low: int = 50
    val_high: int = 255

    # 깊이 필터 (미터)
    min_depth: float = 0.1
    max_depth: float = 1.0


class PenDetector:
    """RealSense 펜 인식기"""

    def __init__(self, config: DetectionConfig = None):
        self.config = config or DetectionConfig()
        self.pipeline = None
        self.align = None
        self.intrinsics = None
        self.running = False

        self._last_position = None

        if not REALSENSE_AVAILABLE:
            print("[PenDetector] RealSense 미설치, 더미 모드")

    def start(self) -> bool:
        """카메라 시작"""
        if not REALSENSE_AVAILABLE:
            self.running = True
            return True

        try:
            self.pipeline = rs.pipeline()
            config = rs.config()

            config.enable_stream(rs.stream.color,
                self.config.width, self.config.height, rs.format.bgr8, self.config.fps)
            config.enable_stream(rs.stream.depth,
                self.config.width, self.config.height, rs.format.z16, self.config.fps)

            profile = self.pipeline.start(config)
            self.align = rs.align(rs.stream.color)

            depth_stream = profile.get_stream(rs.stream.depth)
            self.intrinsics = depth_stream.as_video_stream_profile().get_intrinsics()

            self.running = True
            print("[PenDetector] 시작됨")
            time.sleep(0.5)
            return True

        except Exception as e:
            print(f"[PenDetector] 시작 실패: {e}")
            return False

    def stop(self):
        """카메라 정지"""
        if self.pipeline and REALSENSE_AVAILABLE:
            self.pipeline.stop()
        self.running = False
        print("[PenDetector] 정지됨")

    def get_frames(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """현재 프레임"""
        if not REALSENSE_AVAILABLE:
            return None, None

        if not self.running:
            return None, None

        try:
            frames = self.pipeline.wait_for_frames(timeout_ms=1000)
            aligned = self.align.process(frames)

            color = aligned.get_color_frame()
            depth = aligned.get_depth_frame()

            if not color or not depth:
                return None, None

            return np.asanyarray(color.get_data()), np.asanyarray(depth.get_data())

        except Exception as e:
            print(f"[PenDetector] 프레임 획득 실패: {e}")
            return None, None

    def detect_pen_pixel(self, color: np.ndarray, depth: np.ndarray) -> Optional[Tuple[int, int, float]]:
        """펜 위치 (픽셀 좌표)"""
        h, w = color.shape[:2]

        # HSV 변환 및 색상 필터
        hsv = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)
        lower = np.array([self.config.hue_low, self.config.sat_low, self.config.val_low])
        upper = np.array([self.config.hue_high, self.config.sat_high, self.config.val_high])
        mask = cv2.inRange(hsv, lower, upper)

        # 깊이 필터
        depth_mask = (depth > self.config.min_depth * 1000) & (depth < self.config.max_depth * 1000)
        mask = mask & depth_mask.astype(np.uint8) * 255

        # 모폴로지
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # 컨투어
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return None

        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) < 100:
            return None

        M = cv2.moments(largest)
        if M["m00"] == 0:
            return None

        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        # 깊이 값
        depth_roi = depth[max(0, cy-5):cy+5, max(0, cx-5):cx+5]
        valid = depth_roi[depth_roi > 0]

        if len(valid) == 0:
            return None

        depth_m = np.median(valid) / 1000.0

        return cx, cy, depth_m

    def pixel_to_camera(self, cx: int, cy: int, depth_m: float) -> np.ndarray:
        """픽셀 → 카메라 좌표"""
        if not REALSENSE_AVAILABLE or self.intrinsics is None:
            fx, fy = 600, 600
            ppx, ppy = self.config.width / 2, self.config.height / 2
            x = (cx - ppx) * depth_m / fx
            y = (cy - ppy) * depth_m / fy
            return np.array([x, y, depth_m])

        point = rs.rs2_deproject_pixel_to_point(self.intrinsics, [cx, cy], depth_m)
        return np.array(point)

    def get_pen_position_camera(self) -> Optional[np.ndarray]:
        """펜 위치 (카메라 좌표계)"""
        if not REALSENSE_AVAILABLE:
            # 더미 데이터
            return np.array([0.0, 0.0, 0.4])

        color, depth = self.get_frames()
        if color is None:
            return self._last_position

        result = self.detect_pen_pixel(color, depth)
        if result is None:
            return self._last_position

        cx, cy, depth_m = result
        position = self.pixel_to_camera(cx, cy, depth_m)

        self._last_position = position
        return position.copy()

    def visualize(self, wait_key: int = 1) -> bool:
        """시각화 (디버깅용)"""
        color, depth = self.get_frames()
        if color is None:
            return True

        display = color.copy()
        result = self.detect_pen_pixel(color, depth)

        if result:
            cx, cy, depth_m = result
            cv2.circle(display, (cx, cy), 10, (0, 255, 0), 2)
            cv2.putText(display, f"{depth_m:.2f}m", (cx+15, cy),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        cv2.imshow("Pen Detector", display)
        return cv2.waitKey(wait_key) & 0xFF != ord('q')

    def set_color_filter(self, hue_low: int, hue_high: int):
        """색상 필터 설정"""
        self.config.hue_low = hue_low
        self.config.hue_high = hue_high


def main():
    """테스트"""
    print("PenDetector 테스트")

    detector = PenDetector()
    if not detector.start():
        return

    print("'q'로 종료")

    try:
        while detector.visualize(30):
            pos = detector.get_pen_position_camera()
            if pos is not None:
                print(f"\rPos: {pos.round(3)}", end="")
    except KeyboardInterrupt:
        pass

    detector.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
