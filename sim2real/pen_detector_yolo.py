#!/usr/bin/env python3
"""
YOLO 세그멘테이션 기반 펜 감지기

YOLOv8-seg 모델을 사용하여 펜을 감지합니다.
세그멘테이션 마스크로 정확한 펜 끝점(캡/팁)을 추출합니다.

Usage:
    from pen_detector_yolo import YOLOPenDetector

    detector = YOLOPenDetector(model_path="best.pt")  # 세그멘테이션 모델
    detector.start()

    result = detector.detect()
    if result:
        print(f"Pen at: {result.position}")
        print(f"Cap: {result.cap_3d}, Tip: {result.tip_3d}")
"""

import numpy as np
import cv2
import pyrealsense2 as rs
from dataclasses import dataclass
from typing import Optional, Tuple
from enum import Enum
import time


class KalmanFilter3D:
    """3D 좌표용 칼만 필터 (위치 + 속도 모델)"""

    def __init__(self, process_noise: float = 0.01, measurement_noise: float = 0.1):
        """
        Args:
            process_noise: 프로세스 노이즈 (작을수록 부드러움, 반응 느림)
            measurement_noise: 측정 노이즈 (클수록 부드러움, 반응 느림)
        """
        # 상태: [x, y, z, vx, vy, vz]
        self.state = np.zeros(6)
        self.initialized = False

        # 상태 전이 행렬 (dt=1 가정, 실제로는 시간 기반 업데이트)
        self.F = np.array([
            [1, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 1],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1],
        ], dtype=np.float64)

        # 측정 행렬 (위치만 측정)
        self.H = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
        ], dtype=np.float64)

        # 공분산 행렬
        self.P = np.eye(6) * 1.0  # 초기 불확실성

        # 프로세스 노이즈
        self.Q = np.eye(6) * process_noise

        # 측정 노이즈
        self.R = np.eye(3) * measurement_noise

        self.last_time = None

    def reset(self):
        """필터 초기화"""
        self.state = np.zeros(6)
        self.P = np.eye(6) * 1.0
        self.initialized = False
        self.last_time = None

    def predict(self, dt: float = None):
        """예측 단계"""
        if dt is not None and dt > 0:
            # 시간 기반 상태 전이 행렬 업데이트
            self.F[0, 3] = dt
            self.F[1, 4] = dt
            self.F[2, 5] = dt

        # 상태 예측
        self.state = self.F @ self.state

        # 공분산 예측
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, measurement: np.ndarray) -> np.ndarray:
        """
        측정값으로 상태 업데이트

        Args:
            measurement: [x, y, z] 측정값

        Returns:
            필터링된 [x, y, z]
        """
        current_time = time.time()

        # 첫 측정이면 초기화
        if not self.initialized:
            self.state[:3] = measurement
            self.state[3:] = 0  # 초기 속도 0
            self.initialized = True
            self.last_time = current_time
            return measurement.copy()

        # 시간 간격 계산
        dt = current_time - self.last_time if self.last_time else 0.033
        self.last_time = current_time

        # 예측
        self.predict(dt)

        # 칼만 이득 계산
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)

        # 상태 업데이트
        y = measurement - self.H @ self.state  # 잔차
        self.state = self.state + K @ y

        # 공분산 업데이트
        I = np.eye(6)
        self.P = (I - K @ self.H) @ self.P

        return self.state[:3].copy()

    def get_position(self) -> np.ndarray:
        """현재 필터링된 위치 반환"""
        return self.state[:3].copy()

    def get_velocity(self) -> np.ndarray:
        """현재 추정 속도 반환"""
        return self.state[3:].copy()


class DetectionState(Enum):
    DETECTED = "detected"
    LOST = "lost"


@dataclass
class PenDetectionResult:
    """펜 감지 결과"""
    state: DetectionState
    position: np.ndarray          # 3D 위치 (카메라 좌표계)
    pixel: Tuple[int, int]        # 픽셀 좌표 (cx, cy)
    confidence: float             # 감지 신뢰도
    bbox: Optional[Tuple[int, int, int, int]] = None  # (x, y, w, h)
    grasp_point: Optional[np.ndarray] = None  # 잡기 좋은 위치 (3D)
    grasp_pixel: Optional[Tuple[int, int]] = None  # 잡기 좋은 위치 (픽셀)
    cap_pixel: Optional[Tuple[int, int]] = None     # 펜 캡 끝점 (픽셀)
    tip_pixel: Optional[Tuple[int, int]] = None     # 펜 반대쪽 끝점 (픽셀)
    cap_3d: Optional[np.ndarray] = None   # 펜 캡 끝점 (3D)
    tip_3d: Optional[np.ndarray] = None   # 펜 반대쪽 끝점 (3D)
    direction_3d: Optional[np.ndarray] = None  # 펜 방향 벡터 (정규화, 필터링됨)
    pitch_deg: float = 0.0        # 3D 기울기 - pitch (앞뒤, 도)
    yaw_deg: float = 0.0          # 3D 기울기 - yaw (좌우, 도)
    orientation_2d_deg: float = 0.0  # 2D 기울기 (화면상, 도)
    is_vertical: bool = True      # 펜이 세로 방향인지
    mask: Optional[np.ndarray] = None  # 세그멘테이션 마스크


@dataclass
class YOLODetectorConfig:
    """YOLO 감지기 설정"""
    # 카메라 설정
    width: int = 640
    height: int = 480
    fps: int = 30
    flip_image: bool = False

    # YOLO 설정
    confidence_threshold: float = 0.2  # 낮춰서 감지 안정성 향상
    iou_threshold: float = 0.45

    # 깊이 설정
    min_depth_m: float = 0.1
    max_depth_m: float = 2.0

    # 펜 설정 (grasp point 계산용)
    pen_length: float = 0.14  # 14cm

    # 칼만 필터 설정
    use_kalman_filter: bool = True
    kalman_process_noise: float = 0.001  # 작을수록 부드러움
    kalman_measurement_noise: float = 0.05  # 클수록 부드러움


class YOLOPenDetector:
    """YOLO 기반 펜 감지기"""

    def __init__(self, model_path: str, config: YOLODetectorConfig = None):
        """
        Args:
            model_path: YOLOv8 모델 경로 (.pt)
            config: 감지기 설정
        """
        self.config = config or YOLODetectorConfig()
        self.model_path = model_path
        self.model = None

        # RealSense
        self.pipeline = None
        self.align = None
        self.intrinsics = None

        # 상태
        self.running = False
        self._last_result = None
        self._last_color_image = None
        self._last_depth_image = None

        # 칼만 필터 (cap, tip, center 각각)
        if self.config.use_kalman_filter:
            self.kf_cap = KalmanFilter3D(
                process_noise=self.config.kalman_process_noise,
                measurement_noise=self.config.kalman_measurement_noise
            )
            self.kf_tip = KalmanFilter3D(
                process_noise=self.config.kalman_process_noise,
                measurement_noise=self.config.kalman_measurement_noise
            )
            self.kf_center = KalmanFilter3D(
                process_noise=self.config.kalman_process_noise,
                measurement_noise=self.config.kalman_measurement_noise
            )
        else:
            self.kf_cap = None
            self.kf_tip = None
            self.kf_center = None

        self._lost_frames = 0  # 연속 미감지 프레임 수

        # 방향 필터 (Exponential Moving Average)
        self._direction_ema = None
        self._direction_alpha = 0.3  # 낮을수록 부드러움

        # Cap/Tip 픽셀 좌표 EMA 필터
        self._cap_pixel_ema = None
        self._tip_pixel_ema = None
        self._pixel_ema_alpha = 0.15  # 낮을수록 부드러움

    def start(self) -> bool:
        """카메라 및 모델 시작"""
        try:
            # YOLO 모델 로드
            from ultralytics import YOLO
            self.model = YOLO(self.model_path)
            print(f"[YOLOPenDetector] Model loaded: {self.model_path}")
        except ImportError:
            print("[Error] ultralytics not installed. Run: pip install ultralytics")
            return False
        except Exception as e:
            print(f"[Error] Failed to load model: {e}")
            return False

        try:
            # RealSense 시작
            self.pipeline = rs.pipeline()
            config = rs.config()
            config.enable_stream(rs.stream.color, self.config.width,
                               self.config.height, rs.format.bgr8, self.config.fps)
            config.enable_stream(rs.stream.depth, self.config.width,
                               self.config.height, rs.format.z16, self.config.fps)

            profile = self.pipeline.start(config)

            # Depth → Color 정렬
            self.align = rs.align(rs.stream.color)

            # 카메라 내부 파라미터
            color_stream = profile.get_stream(rs.stream.color)
            self.intrinsics = color_stream.as_video_stream_profile().get_intrinsics()

            self.running = True
            print("[YOLOPenDetector] Started")
            return True

        except Exception as e:
            print(f"[YOLOPenDetector] Camera start failed: {e}")
            return False

    def stop(self):
        """정지"""
        if self.pipeline:
            self.pipeline.stop()
        self.running = False
        print("[YOLOPenDetector] Stopped")

    def reset_tracking(self):
        """Cap/Tip EMA 및 칼만 필터 리셋 (cap/tip 바뀌었을 때 사용)"""
        self._cap_pixel_ema = None
        self._tip_pixel_ema = None
        self._direction_ema = None
        if self.kf_cap:
            self.kf_cap.reset()
        if self.kf_tip:
            self.kf_tip.reset()
        if self.kf_center:
            self.kf_center.reset()
        print("[YOLOPenDetector] Tracking reset")

    def swap_cap_tip(self):
        """Cap과 Tip을 수동으로 swap (잘못 인식된 경우 사용)"""
        if self._cap_pixel_ema is not None and self._tip_pixel_ema is not None:
            # EMA 값 swap
            self._cap_pixel_ema, self._tip_pixel_ema = self._tip_pixel_ema.copy(), self._cap_pixel_ema.copy()

            # 방향 벡터 반전
            if self._direction_ema is not None:
                self._direction_ema = -self._direction_ema

            # 칼만 필터의 cap/tip 상태도 swap
            if self.kf_cap and self.kf_tip and self.kf_cap.initialized and self.kf_tip.initialized:
                cap_state = self.kf_cap.state.copy()
                tip_state = self.kf_tip.state.copy()
                self.kf_cap.state = tip_state
                self.kf_tip.state = cap_state

            print("[YOLOPenDetector] Cap/Tip swapped!")
            return True
        else:
            print("[YOLOPenDetector] Cannot swap - no tracking data yet")
            return False

    def get_last_frames(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """마지막 detect()에서 사용한 프레임 반환 (동기화용)"""
        return self._last_color_image, self._last_depth_image

    def get_frames(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """프레임 가져오기"""
        if not self.running:
            return None, None

        try:
            frames = self.pipeline.wait_for_frames(timeout_ms=1000)
            aligned_frames = self.align.process(frames)

            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()

            if not color_frame or not depth_frame:
                return None, None

            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())

            # Flip
            if self.config.flip_image:
                color_image = cv2.flip(color_image, -1)
                depth_image = cv2.flip(depth_image, -1)

            return color_image, depth_image

        except Exception as e:
            print(f"[YOLOPenDetector] Get frames failed: {e}")
            return None, None

    def detect(self) -> Optional[PenDetectionResult]:
        """펜 감지 (세그멘테이션 마스크 기반)"""
        color_image, depth_image = self.get_frames()
        if color_image is None:
            return None

        # 사용한 이미지 저장 (시각화용)
        self._last_color_image = color_image
        self._last_depth_image = depth_image

        # YOLO 세그멘테이션 추론
        results = self.model(color_image,
                            conf=self.config.confidence_threshold,
                            iou=self.config.iou_threshold,
                            verbose=False)

        # 결과 파싱 - 세그멘테이션 마스크 확인
        if len(results) == 0 or results[0].masks is None or len(results[0].masks) == 0:
            self._lost_frames += 1
            # 최근 5프레임 이내면 이전 결과 유지 (깜빡임 방지)
            if self._lost_frames <= 5 and self._last_result is not None and self._last_result.state == DetectionState.DETECTED:
                return self._last_result

            self._last_result = PenDetectionResult(
                state=DetectionState.LOST,
                position=np.zeros(3),
                pixel=(0, 0),
                confidence=0.0
            )
            return self._last_result

        # 감지 성공 - lost_frames 리셋
        self._lost_frames = 0

        # 가장 confidence 높은 감지 결과 사용
        boxes = results[0].boxes
        masks = results[0].masks
        best_idx = boxes.conf.argmax().item()

        box = boxes.xyxy[best_idx].cpu().numpy()  # x1, y1, x2, y2
        conf = boxes.conf[best_idx].item()

        x1, y1, x2, y2 = map(int, box)
        w, h = x2 - x1, y2 - y1
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        # 세그멘테이션 마스크 추출
        mask = masks.data[best_idx].cpu().numpy()  # (H, W) 또는 모델 출력 크기

        # 마스크를 원본 이미지 크기로 리사이즈
        if mask.shape != (self.config.height, self.config.width):
            mask = cv2.resize(mask, (self.config.width, self.config.height),
                            interpolation=cv2.INTER_NEAREST)

        # 마스크 이진화
        mask_binary = (mask > 0.5).astype(np.uint8) * 255

        # 마스크에서 펜 끝점, 중심, 방향 추출
        cap_pixel, tip_pixel, center_pixel, orientation_2d_deg = self._find_pen_endpoints_from_mask(mask_binary)

        # 펜 방향 추정 (2D 기울기로)
        is_vertical = abs(orientation_2d_deg) > 45

        # === 펜 내부 점 기반 깊이 계산 (안정적) ===
        # center에서 cap/tip 방향으로 30% 지점 계산 (펜 내부라 깊이 안정적)
        center_arr = np.array(center_pixel, dtype=np.float32)
        cap_arr = np.array(cap_pixel, dtype=np.float32)
        tip_arr = np.array(tip_pixel, dtype=np.float32)

        # 내부 점 계산 (끝점에서 30% 안쪽)
        inner_cap_pixel = (center_arr + 0.3 * (cap_arr - center_arr)).astype(int)
        inner_tip_pixel = (center_arr + 0.3 * (tip_arr - center_arr)).astype(int)

        # 내부 점들의 3D 좌표 (깊이 안정적)
        inner_cap_3d = self._pixel_to_3d(int(inner_cap_pixel[0]), int(inner_cap_pixel[1]), depth_image)
        inner_tip_3d = self._pixel_to_3d(int(inner_tip_pixel[0]), int(inner_tip_pixel[1]), depth_image)

        # 중심점 3D
        position = self._pixel_to_3d(center_pixel[0], center_pixel[1], depth_image)

        # 펜 방향 벡터 계산 (내부 점 기반)
        pen_direction = np.zeros(3)
        if np.linalg.norm(inner_cap_3d) > 0.01 and np.linalg.norm(inner_tip_3d) > 0.01:
            pen_direction = inner_tip_3d - inner_cap_3d
            pen_length_3d = np.linalg.norm(pen_direction)
            if pen_length_3d > 0.01:
                pen_direction = pen_direction / pen_length_3d  # 정규화

                # 픽셀 거리 계산
                pixel_dist_cap_to_inner = np.linalg.norm(cap_arr - inner_cap_pixel)
                pixel_dist_tip_to_inner = np.linalg.norm(tip_arr - inner_tip_pixel)
                pixel_dist_inner = np.linalg.norm(inner_tip_pixel - inner_cap_pixel)

                # 3D 거리 비율로 실제 끝점 위치 계산
                if pixel_dist_inner > 0:
                    ratio_cap = pixel_dist_cap_to_inner / pixel_dist_inner
                    ratio_tip = pixel_dist_tip_to_inner / pixel_dist_inner
                    extend_cap = pen_length_3d * ratio_cap
                    extend_tip = pen_length_3d * ratio_tip

                    cap_3d = inner_cap_3d - pen_direction * extend_cap
                    tip_3d = inner_tip_3d + pen_direction * extend_tip
                else:
                    cap_3d = inner_cap_3d
                    tip_3d = inner_tip_3d
            else:
                cap_3d = inner_cap_3d
                tip_3d = inner_tip_3d
        else:
            # fallback: 깊이 없으면 zeros
            cap_3d = np.zeros(3)
            tip_3d = np.zeros(3)

        # === 칼만 필터 적용 ===
        has_valid_depth = np.linalg.norm(cap_3d) > 0.01 and np.linalg.norm(tip_3d) > 0.01

        if self.config.use_kalman_filter and has_valid_depth:
            # 유효한 측정값이면 필터 업데이트
            cap_3d = self.kf_cap.update(cap_3d)
            tip_3d = self.kf_tip.update(tip_3d)
            position = self.kf_center.update(position)
            self._lost_frames = 0
        elif self.config.use_kalman_filter and self.kf_cap.initialized:
            # 깊이 없지만 이전 값 있으면 예측값 사용 (최대 10프레임)
            self._lost_frames += 1
            if self._lost_frames < 10:
                cap_3d = self.kf_cap.get_position()
                tip_3d = self.kf_tip.get_position()
                position = self.kf_center.get_position()
            else:
                # 너무 오래 감지 안되면 필터 리셋
                self.kf_cap.reset()
                self.kf_tip.reset()
                self.kf_center.reset()

        # 3D 방향 벡터 (cap → tip) - 정규화
        pen_vector = tip_3d - cap_3d
        direction_3d = None

        # pitch, yaw 계산 (카메라 좌표계 기준)
        pitch_deg = 0.0
        yaw_deg = 0.0
        pen_length = np.linalg.norm(pen_vector)
        if pen_length > 0.01:  # 벡터 길이가 1cm 이상일 때만
            # 정규화된 방향 벡터
            direction_3d = pen_vector / pen_length

            # EMA 필터 적용
            if self._direction_ema is None:
                self._direction_ema = direction_3d.copy()
            else:
                self._direction_ema = (self._direction_alpha * direction_3d +
                                       (1 - self._direction_alpha) * self._direction_ema)
                # 필터 후 다시 정규화
                ema_norm = np.linalg.norm(self._direction_ema)
                if ema_norm > 0:
                    self._direction_ema = self._direction_ema / ema_norm

            direction_3d = self._direction_ema.copy()

            # Z축(깊이) 기준 pitch (앞뒤 기울기)
            pitch_deg = np.degrees(np.arctan2(pen_vector[1], pen_vector[2]))
            # Z축 기준 yaw (좌우 기울기)
            yaw_deg = np.degrees(np.arctan2(pen_vector[0], pen_vector[2]))

        # Grasp point = cap 위치
        grasp_pixel = cap_pixel
        grasp_point = cap_3d

        self._last_result = PenDetectionResult(
            state=DetectionState.DETECTED,
            position=position,
            pixel=(cx, cy),
            confidence=conf,
            bbox=(x1, y1, w, h),
            grasp_point=grasp_point,
            grasp_pixel=grasp_pixel,
            cap_pixel=cap_pixel,
            tip_pixel=tip_pixel,
            cap_3d=cap_3d,
            tip_3d=tip_3d,
            direction_3d=direction_3d,
            pitch_deg=pitch_deg,
            yaw_deg=yaw_deg,
            orientation_2d_deg=orientation_2d_deg,
            is_vertical=is_vertical,
            mask=mask_binary
        )

        return self._last_result

    def _find_pen_endpoints_from_mask(self, mask: np.ndarray) -> Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int], float]:
        """
        세그멘테이션 마스크에서 펜의 양 끝점, 중심, 방향 찾기 (PCA 기반)

        Args:
            mask: 이진 마스크 (255=펜, 0=배경)

        Returns:
            (cap_pixel, tip_pixel, center_pixel, orientation_deg): 펜 캡, 펜 끝, 중심, 2D 기울기(도)
        """
        # 컨투어 찾기
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            # 마스크에서 펜을 찾지 못한 경우 기본값 반환
            h, w = mask.shape
            return (w//2, 0), (w//2, h), (w//2, h//2), 90.0

        # 가장 큰 컨투어 선택 (펜)
        largest_contour = max(contours, key=cv2.contourArea)

        if len(largest_contour) < 5:
            h, w = mask.shape
            return (w//2, 0), (w//2, h), (w//2, h//2), 90.0

        # 컨투어 점들로 PCA 수행
        pts = largest_contour.reshape(-1, 2).astype(np.float32)

        # PCA: 주축 방향 찾기
        mean, eigenvectors = cv2.PCACompute(pts, mean=None)
        center_pixel = tuple(mean[0].astype(int))

        # 주축 방향 (첫 번째 고유벡터)
        principal_axis = eigenvectors[0]

        # 2D 기울기 각도 계산 (수평=0도, 수직=90도)
        orientation_rad = np.arctan2(principal_axis[1], principal_axis[0])
        orientation_deg = np.degrees(orientation_rad)

        # 주축 방향으로 가장 멀리 떨어진 두 점 찾기
        # 각 점을 주축에 투영하여 가장 앞/뒤 점 찾기
        projections = np.dot(pts - mean, principal_axis)
        min_idx = np.argmin(projections)
        max_idx = np.argmax(projections)

        endpoint1 = tuple(pts[min_idx].astype(int))
        endpoint2 = tuple(pts[max_idx].astype(int))

        # cap과 tip 결정 (일관성 유지 + EMA 필터)
        cap_pixel, tip_pixel = self._determine_cap_tip_with_ema(endpoint1, endpoint2)

        return cap_pixel, tip_pixel, center_pixel, orientation_deg

    def _determine_cap_tip_with_ema(self, endpoint1: Tuple[int, int], endpoint2: Tuple[int, int]) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """
        Cap과 Tip 결정 (일관성 유지 + EMA 필터로 진동 감소)

        Args:
            endpoint1, endpoint2: 펜의 양 끝점

        Returns:
            (cap_pixel, tip_pixel): EMA 필터링된 좌표
        """
        # 첫 프레임: Y좌표 기반으로 초기화
        if self._cap_pixel_ema is None:
            if endpoint1[1] < endpoint2[1]:
                raw_cap, raw_tip = endpoint1, endpoint2
            else:
                raw_cap, raw_tip = endpoint2, endpoint1

            self._cap_pixel_ema = np.array(raw_cap, dtype=np.float32)
            self._tip_pixel_ema = np.array(raw_tip, dtype=np.float32)
            return raw_cap, raw_tip

        # 이전 cap EMA와 각 끝점의 거리 계산
        dist1_to_cap = np.sqrt((endpoint1[0] - self._cap_pixel_ema[0])**2 +
                               (endpoint1[1] - self._cap_pixel_ema[1])**2)
        dist2_to_cap = np.sqrt((endpoint2[0] - self._cap_pixel_ema[0])**2 +
                               (endpoint2[1] - self._cap_pixel_ema[1])**2)

        # 이전 cap에 더 가까운 쪽을 현재 cap으로
        if dist1_to_cap < dist2_to_cap:
            raw_cap, raw_tip = endpoint1, endpoint2
        else:
            raw_cap, raw_tip = endpoint2, endpoint1

        # EMA 필터 적용
        alpha = self._pixel_ema_alpha
        self._cap_pixel_ema = alpha * np.array(raw_cap, dtype=np.float32) + (1 - alpha) * self._cap_pixel_ema
        self._tip_pixel_ema = alpha * np.array(raw_tip, dtype=np.float32) + (1 - alpha) * self._tip_pixel_ema

        # 정수로 반환
        cap_pixel = (int(round(self._cap_pixel_ema[0])), int(round(self._cap_pixel_ema[1])))
        tip_pixel = (int(round(self._tip_pixel_ema[0])), int(round(self._tip_pixel_ema[1])))

        return cap_pixel, tip_pixel

    def _pixel_to_3d(self, x: int, y: int, depth_image: np.ndarray) -> np.ndarray:
        """픽셀 좌표 → 3D 좌표"""
        # 주변 깊이 평균
        margin = 3
        x1 = max(0, x - margin)
        x2 = min(depth_image.shape[1], x + margin)
        y1 = max(0, y - margin)
        y2 = min(depth_image.shape[0], y + margin)

        depth_region = depth_image[y1:y2, x1:x2]
        valid_depths = depth_region[depth_region > 0]

        if len(valid_depths) == 0:
            return np.zeros(3)

        depth_mm = np.median(valid_depths)
        depth_m = depth_mm / 1000.0

        # 깊이 범위 체크
        if not (self.config.min_depth_m <= depth_m <= self.config.max_depth_m):
            return np.zeros(3)

        # 3D 좌표 계산
        x_3d = (x - self.intrinsics.ppx) * depth_m / self.intrinsics.fx
        y_3d = (y - self.intrinsics.ppy) * depth_m / self.intrinsics.fy
        z_3d = depth_m

        return np.array([x_3d, y_3d, z_3d])

    def visualize(self, image: np.ndarray, result: PenDetectionResult,
                  show_mask: bool = True, mask_alpha: float = 0.4) -> np.ndarray:
        """결과 시각화

        Args:
            image: 원본 이미지
            result: 감지 결과
            show_mask: 마스크 오버레이 표시 여부
            mask_alpha: 마스크 투명도 (0~1)
        """
        display = image.copy()

        if result.state == DetectionState.DETECTED:
            # 세그멘테이션 마스크 오버레이
            if show_mask and result.mask is not None:
                # 마스크를 컬러로 변환 (초록색 오버레이)
                mask_color = np.zeros_like(display)
                mask_color[result.mask > 0] = [0, 255, 100]  # 초록색

                # 반투명 블렌딩
                mask_area = result.mask > 0
                display[mask_area] = cv2.addWeighted(
                    display, 1 - mask_alpha,
                    mask_color, mask_alpha, 0
                )[mask_area]

                # 마스크 외곽선
                contours, _ = cv2.findContours(result.mask, cv2.RETR_EXTERNAL,
                                               cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(display, contours, -1, (0, 255, 0), 2)

            # 바운딩 박스 (점선으로 변경)
            if result.bbox:
                x, y, w, h = result.bbox
                # 점선 효과
                for i in range(0, w, 10):
                    cv2.line(display, (x+i, y), (x+min(i+5, w), y), (0, 200, 0), 1)
                    cv2.line(display, (x+i, y+h), (x+min(i+5, w), y+h), (0, 200, 0), 1)
                for i in range(0, h, 10):
                    cv2.line(display, (x, y+i), (x, y+min(i+5, h)), (0, 200, 0), 1)
                    cv2.line(display, (x+w, y+i), (x+w, y+min(i+5, h)), (0, 200, 0), 1)

            # Cap 끝점 (초록색 원)
            if result.cap_pixel:
                cv2.circle(display, result.cap_pixel, 6, (0, 255, 0), -1)
                cv2.putText(display, "CAP", (result.cap_pixel[0] + 8, result.cap_pixel[1]),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)

            # Tip 끝점 (빨간색 원)
            if result.tip_pixel:
                cv2.circle(display, result.tip_pixel, 6, (0, 0, 255), -1)
                cv2.putText(display, "TIP", (result.tip_pixel[0] + 8, result.tip_pixel[1]),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

            # Cap → Tip 연결선 (펜 축)
            if result.cap_pixel and result.tip_pixel:
                cv2.line(display, result.cap_pixel, result.tip_pixel, (255, 255, 0), 2)

            # Grasp point (파란색, 더 크게)
            if result.grasp_pixel:
                gx, gy = result.grasp_pixel
                cv2.circle(display, (gx, gy), 8, (255, 0, 0), -1)
                cv2.putText(display, "GRASP", (gx + 10, gy),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)

            # 정보 표시
            cv2.putText(display, f"DETECTED ({result.confidence:.2f})", (10, 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # 3D 위치 (깊이 데이터 있을 때만)
            has_depth = result.position is not None and np.linalg.norm(result.position) > 0.01
            if has_depth:
                cv2.putText(display,
                           f"Pos: [{result.position[0]:.3f}, {result.position[1]:.3f}, {result.position[2]:.3f}]",
                           (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            else:
                cv2.putText(display, "Pos: [No depth data]",
                           (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 165, 255), 1)

            # 3D 기울기 (pitch, yaw) - 깊이 있을 때만
            if has_depth:
                cv2.putText(display,
                           f"Pitch: {result.pitch_deg:.1f} deg | Yaw: {result.yaw_deg:.1f} deg",
                           (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)
            else:
                cv2.putText(display, "Pitch/Yaw: [No depth]",
                           (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 165, 255), 1)

            # Grasp point 3D 위치 (깊이 있을 때만)
            has_grasp_depth = result.grasp_point is not None and np.linalg.norm(result.grasp_point) > 0.01
            if has_grasp_depth:
                cv2.putText(display,
                           f"Grasp: [{result.grasp_point[0]:.3f}, {result.grasp_point[1]:.3f}, {result.grasp_point[2]:.3f}]",
                           (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 150, 50), 1)
            else:
                cv2.putText(display, "Grasp: [No depth - move pen closer]",
                           (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 165, 255), 1)

            # 2D 방향
            orientation_str = "Vertical" if result.is_vertical else "Horizontal"
            cv2.putText(display, f"2D: {result.orientation_2d_deg:.1f} deg ({orientation_str})",
                       (10, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        else:
            cv2.putText(display, "LOST - No pen detected", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        return display


def main():
    """테스트"""
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="YOLOv8 모델 경로")
    args = parser.parse_args()

    detector = YOLOPenDetector(args.model)

    if not detector.start():
        print("시작 실패!")
        return

    print("q: 종료")

    try:
        while True:
            result = detector.detect()
            color_image, _ = detector.get_frames()

            if color_image is not None and result is not None:
                display = detector.visualize(color_image, result)
                cv2.imshow("YOLO Pen Detector", display)

            if cv2.waitKey(30) & 0xFF == ord('q'):
                break

    finally:
        detector.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
