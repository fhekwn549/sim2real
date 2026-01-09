#!/usr/bin/env python3
"""
고급 펜 인식 모듈 - MediaPipe 손 마스킹 + Kalman 추적

손으로 펜을 가려도 인식 가능하도록 개선된 버전입니다.

Features:
    - MediaPipe 손 감지 및 마스킹
    - Kalman 필터 기반 위치 추적/예측
    - Grasp point 및 펜 방향(z축) 계산
    - 깊이 정보 기반 손/펜 분리

Usage:
    from pen_detector_advanced import AdvancedPenDetector

    detector = AdvancedPenDetector()
    detector.start()

    result = detector.detect()
    if result:
        print(f"Position: {result.position}")
        print(f"Grasp point: {result.grasp_point}")
        print(f"Direction: {result.direction}")

    detector.stop()
"""

import numpy as np
import cv2
import time
from typing import Optional, Tuple, List
from dataclasses import dataclass, field
from enum import Enum

# pyrealsense2 import
try:
    import pyrealsense2 as rs
    REALSENSE_AVAILABLE = True
except ImportError:
    REALSENSE_AVAILABLE = False
    print("[Warning] pyrealsense2 not found. Running in simulation mode.")

# MediaPipe import
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    print("[Warning] mediapipe not found. Hand masking disabled.")


class DetectionState(Enum):
    """감지 상태"""
    DETECTED = "detected"           # 직접 감지됨
    PREDICTED = "predicted"         # Kalman 예측 (가려짐)
    LOST = "lost"                   # 감지 실패


@dataclass
class PenDetectionResult:
    """펜 감지 결과"""
    # 기본 정보
    position: np.ndarray            # 3D 위치 (카메라 좌표계) [x, y, z]
    pixel: Tuple[int, int]          # 픽셀 좌표 (cx, cy)
    depth: float                    # 깊이 (미터)

    # 방향 정보
    grasp_point: np.ndarray         # 잡기 좋은 위치 (카메라 좌표계)
    direction: np.ndarray           # 펜 방향 벡터 (z축 방향)
    angle: float                    # 펜 기울기 각도 (도)

    # 상태 정보
    state: DetectionState           # 감지/예측/실패
    confidence: float               # 신뢰도 (0~1)
    hand_occluding: bool            # 손에 의한 가림 여부

    # 바운딩 박스
    bbox: Optional[Tuple[int, int, int, int]] = None  # (x, y, w, h)


@dataclass
class AdvancedDetectionConfig:
    """고급 펜 인식 설정"""
    # 카메라 설정
    width: int = 640
    height: int = 480
    fps: int = 30
    flip_image: bool = False  # 카메라가 뒤집혀 있을 때 True (상하좌우 반전)

    # 색상 필터 (HSV) - 펜 캡
    hue_low: int = 100
    hue_high: int = 130
    sat_low: int = 100
    sat_high: int = 255
    val_low: int = 50
    val_high: int = 255

    # 깊이 필터
    min_depth_m: float = 0.1
    max_depth_m: float = 2.5  # 2.5m까지 감지 가능

    # ROI 설정
    roi_ratio: float = 1.0  # 전체 이미지 사용

    # Kalman 필터 설정
    kalman_process_noise: float = 1e-2      # 프로세스 노이즈
    kalman_measurement_noise: float = 1e-1  # 측정 노이즈
    prediction_timeout: float = 1.0          # 예측 유지 시간 (초)

    # 손 마스킹 설정
    hand_mask_expansion: int = 20           # 손 마스크 확장 픽셀 (줄임)
    hand_depth_threshold: float = 0.03      # 손/펜 깊이 차이 임계값 (m)

    # 거리 기반 동적 조정
    base_contour_area: int = 50             # 기준 거리(0.5m)에서의 최소 면적 (줄임)
    reference_depth: float = 0.5            # 기준 거리 (m)

    # 펜 형태 설정
    min_contour_area: int = 30              # 최소 컨투어 면적 (대폭 줄임)
    pen_aspect_ratio_min: float = 1.3       # 펜 종횡비 최소값 (기울임 대응)
    grasp_offset_ratio: float = 0.3         # grasp point 오프셋 (펜 길이 대비)

    # 모폴로지 설정
    morph_kernel_size: int = 3              # 모폴로지 커널 크기 (작을수록 작은 객체 보존)


class KalmanTracker:
    """Kalman 필터 기반 3D 위치 추적기"""

    def __init__(self, process_noise: float = 1e-2, measurement_noise: float = 1e-1):
        """
        Args:
            process_noise: 프로세스 노이즈 (작을수록 부드러운 추적)
            measurement_noise: 측정 노이즈 (작을수록 측정값 신뢰)
        """
        # 상태: [x, y, z, vx, vy, vz] (위치 + 속도)
        self.kf = cv2.KalmanFilter(6, 3)

        # 전이 행렬 (등속 운동 모델)
        self.kf.transitionMatrix = np.array([
            [1, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 1],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]
        ], dtype=np.float32)

        # 측정 행렬 (위치만 측정)
        self.kf.measurementMatrix = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0]
        ], dtype=np.float32)

        # 노이즈 설정
        self.kf.processNoiseCov = np.eye(6, dtype=np.float32) * process_noise
        self.kf.measurementNoiseCov = np.eye(3, dtype=np.float32) * measurement_noise
        self.kf.errorCovPost = np.eye(6, dtype=np.float32)

        self.initialized = False
        self.last_update_time = 0

    def initialize(self, position: np.ndarray):
        """초기 위치 설정"""
        self.kf.statePost = np.array([
            position[0], position[1], position[2],
            0, 0, 0
        ], dtype=np.float32)
        self.initialized = True
        self.last_update_time = time.time()

    def update(self, position: np.ndarray) -> np.ndarray:
        """측정값으로 업데이트"""
        if not self.initialized:
            self.initialize(position)
            return position

        measurement = np.array(position, dtype=np.float32)
        self.kf.correct(measurement)
        self.last_update_time = time.time()

        return self.get_position()

    def predict(self) -> np.ndarray:
        """다음 위치 예측"""
        if not self.initialized:
            return np.zeros(3)

        predicted = self.kf.predict()
        return predicted[:3].flatten()

    def get_position(self) -> np.ndarray:
        """현재 추정 위치"""
        if not self.initialized:
            return np.zeros(3)
        return self.kf.statePost[:3].flatten()

    def get_velocity(self) -> np.ndarray:
        """현재 추정 속도"""
        if not self.initialized:
            return np.zeros(3)
        return self.kf.statePost[3:6].flatten()

    def time_since_update(self) -> float:
        """마지막 업데이트 이후 경과 시간"""
        return time.time() - self.last_update_time


class HandDetector:
    """MediaPipe 기반 손 감지기"""

    def __init__(self, max_hands: int = 2, detection_confidence: float = 0.5):
        if not MEDIAPIPE_AVAILABLE:
            self.hands = None
            return

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_hands,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils

    def detect(self, color_image: np.ndarray) -> List[np.ndarray]:
        """
        손 감지

        Returns:
            손 마스크 리스트 (각 손에 대한 convex hull 폴리곤)
        """
        if self.hands is None:
            return []

        # BGR -> RGB
        rgb_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
        self._last_results = self.hands.process(rgb_image)

        hand_polygons = []

        if self._last_results.multi_hand_landmarks:
            h, w = color_image.shape[:2]

            for hand_landmarks in self._last_results.multi_hand_landmarks:
                # 랜드마크 좌표 추출
                points = []
                for lm in hand_landmarks.landmark:
                    x = int(lm.x * w)
                    y = int(lm.y * h)
                    points.append([x, y])

                points = np.array(points, dtype=np.int32)

                # Convex hull로 손 영역
                hull = cv2.convexHull(points)
                hand_polygons.append(hull)

        return hand_polygons

    def get_grip_position(self, color_image: np.ndarray) -> Optional[Tuple[int, int]]:
        """
        펜을 잡는 위치 (엄지-검지 사이) 반환

        MediaPipe 랜드마크:
        - 4: 엄지 끝
        - 8: 검지 끝

        Returns:
            (x, y) 픽셀 좌표 또는 None
        """
        if self.hands is None:
            return None

        # 이미 detect()에서 처리된 결과 사용
        if not hasattr(self, '_last_results') or self._last_results is None:
            return None

        if not self._last_results.multi_hand_landmarks:
            return None

        h, w = color_image.shape[:2]

        # 첫 번째 손 사용
        hand_landmarks = self._last_results.multi_hand_landmarks[0]

        # 엄지 끝 (4)과 검지 끝 (8) 중점 = 펜 잡는 위치
        thumb_tip = hand_landmarks.landmark[4]
        index_tip = hand_landmarks.landmark[8]

        grip_x = int((thumb_tip.x + index_tip.x) / 2 * w)
        grip_y = int((thumb_tip.y + index_tip.y) / 2 * h)

        return (grip_x, grip_y)

    def create_hand_mask(self, color_image: np.ndarray, depth_image: np.ndarray = None,
                         expansion: int = 20, depth_threshold: float = 0.03) -> Tuple[np.ndarray, float]:
        """
        손 영역 마스크 생성 (깊이 정보 활용)

        손 영역 내에서도 손보다 뒤(더 깊은 곳)에 있는 픽셀은 마스킹에서 제외합니다.
        이를 통해 손으로 펜을 잡고 있어도 펜 부분은 감지 가능합니다.

        Args:
            color_image: BGR 이미지
            depth_image: 깊이 이미지 (mm 단위, optional)
            expansion: 마스크 확장 픽셀
            depth_threshold: 손/펜 깊이 차이 임계값 (m)

        Returns:
            (손 영역 마스크, 손 평균 깊이)
            마스크에서 255 = 손 영역 (제외할 영역)
        """
        h, w = color_image.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        hand_avg_depth = 0.0

        hand_polygons = self.detect(color_image)

        if not hand_polygons:
            return mask, hand_avg_depth

        # 기본 손 마스크 생성
        for polygon in hand_polygons:
            cv2.fillConvexPoly(mask, polygon, 255)

        # 깊이 정보가 있으면 손보다 뒤에 있는 픽셀 제외
        if depth_image is not None:
            # 손 영역의 평균 깊이 계산
            hand_depths = depth_image[mask > 0]
            valid_hand_depths = hand_depths[(hand_depths > 0) & (hand_depths < 10000)]

            if len(valid_hand_depths) > 0:
                hand_avg_depth = np.median(valid_hand_depths) / 1000.0  # mm -> m

                # 손보다 뒤(더 깊은 곳)에 있는 픽셀은 마스크에서 제외
                # 이 픽셀들은 손 뒤의 펜일 가능성이 높음
                depth_m = depth_image.astype(np.float32) / 1000.0
                behind_hand = depth_m > (hand_avg_depth + depth_threshold)

                # 손 영역이면서 손보다 뒤에 있는 픽셀은 마스크에서 제외
                mask[behind_hand] = 0

        # 마스크 확장 (손 가장자리도 제외, 단 확장량 줄임)
        if expansion > 0:
            kernel = np.ones((expansion, expansion), np.uint8)
            mask = cv2.dilate(mask, kernel, iterations=1)

        return mask, hand_avg_depth

    def close(self):
        if self.hands:
            self.hands.close()


class AdvancedPenDetector:
    """고급 펜 인식기 - MediaPipe + Kalman"""

    def __init__(self, config: AdvancedDetectionConfig = None, simulation: bool = False):
        self.config = config or AdvancedDetectionConfig()
        self.simulation = simulation or not REALSENSE_AVAILABLE

        # RealSense
        self.pipeline = None
        self.align = None
        self.intrinsics = None
        self.depth_scale = 0.001  # 기본값 (1mm = 0.001m)
        self.running = False

        # 손 감지기
        self.hand_detector = HandDetector()

        # Kalman 추적기
        self.tracker = KalmanTracker(
            process_noise=self.config.kalman_process_noise,
            measurement_noise=self.config.kalman_measurement_noise
        )

        # 방향 추적용 Kalman (펜 방향 벡터)
        self.direction_tracker = KalmanTracker(
            process_noise=self.config.kalman_process_noise * 0.1,
            measurement_noise=self.config.kalman_measurement_noise * 0.1
        )

        # 마지막 결과
        self._last_result: Optional[PenDetectionResult] = None
        self._last_valid_detection_time = 0

        # 디버그용 이미지
        self._debug_images = {}

        print(f"[AdvancedPenDetector] {'Simulation' if self.simulation else 'RealSense'} mode")
        print(f"[AdvancedPenDetector] MediaPipe: {'enabled' if MEDIAPIPE_AVAILABLE else 'disabled'}")

    def start(self) -> bool:
        """카메라 시작"""
        if self.simulation:
            self.running = True
            print("[AdvancedPenDetector] Started (simulation)")
            return True

        try:
            self.pipeline = rs.pipeline()
            config = rs.config()

            config.enable_stream(
                rs.stream.color,
                self.config.width, self.config.height,
                rs.format.bgr8, self.config.fps
            )
            config.enable_stream(
                rs.stream.depth,
                self.config.width, self.config.height,
                rs.format.z16, self.config.fps
            )

            profile = self.pipeline.start(config)
            self.align = rs.align(rs.stream.color)

            # color 스트림의 intrinsics 사용 (align으로 depth를 color에 맞추므로)
            color_stream = profile.get_stream(rs.stream.color)
            self.intrinsics = color_stream.as_video_stream_profile().get_intrinsics()
            print(f"[AdvancedPenDetector] Color intrinsics: fx={self.intrinsics.fx:.1f}, fy={self.intrinsics.fy:.1f}, "
                  f"ppx={self.intrinsics.ppx:.1f}, ppy={self.intrinsics.ppy:.1f}")

            # depth_scale 가져오기 (미터 변환에 필요)
            depth_sensor = profile.get_device().first_depth_sensor()
            self.depth_scale = depth_sensor.get_depth_scale()
            print(f"[AdvancedPenDetector] Depth scale: {self.depth_scale}")

            self.running = True
            print("[AdvancedPenDetector] Started")

            time.sleep(0.5)
            return True

        except Exception as e:
            print(f"[AdvancedPenDetector] Start failed: {e}")
            return False

    def stop(self):
        """정지"""
        if self.pipeline and not self.simulation:
            self.pipeline.stop()
        self.hand_detector.close()
        self.running = False
        print("[AdvancedPenDetector] Stopped")

    def get_frames(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """프레임 가져오기"""
        if self.simulation:
            color = np.zeros((self.config.height, self.config.width, 3), dtype=np.uint8)
            depth = np.ones((self.config.height, self.config.width), dtype=np.uint16) * 500
            return color, depth

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

            # 카메라가 뒤집혀 있으면 상하좌우 반전
            if self.config.flip_image:
                color_image = cv2.flip(color_image, -1)  # -1 = 상하좌우 모두 반전
                depth_image = cv2.flip(depth_image, -1)

            return color_image, depth_image

        except Exception as e:
            print(f"[AdvancedPenDetector] Get frames failed: {e}")
            return None, None

    def _create_color_mask(self, hsv_image: np.ndarray) -> np.ndarray:
        """HSV 색상 마스크 생성"""
        lower = np.array([self.config.hue_low, self.config.sat_low, self.config.val_low])
        upper = np.array([self.config.hue_high, self.config.sat_high, self.config.val_high])
        return cv2.inRange(hsv_image, lower, upper)

    def _create_depth_mask(self, depth_image: np.ndarray) -> np.ndarray:
        """깊이 마스크 생성"""
        min_depth_mm = self.config.min_depth_m * 1000
        max_depth_mm = self.config.max_depth_m * 1000
        return ((depth_image > min_depth_mm) & (depth_image < max_depth_mm)).astype(np.uint8) * 255

    def _get_dynamic_min_area(self, estimated_depth: float) -> int:
        """
        거리에 따른 동적 최소 컨투어 면적 계산

        멀리 있을수록 펜이 작게 보이므로 min_area를 줄임
        면적은 거리의 제곱에 반비례
        """
        if estimated_depth <= 0:
            return self.config.base_contour_area

        # 기준 거리 대비 비율 계산
        depth_ratio = self.config.reference_depth / max(estimated_depth, 0.1)

        # 면적은 거리의 제곱에 반비례
        dynamic_area = int(self.config.base_contour_area * (depth_ratio ** 2))

        # 최소값을 더 낮춤 (작은 펜 캡도 감지)
        return max(dynamic_area, 10)

    def _find_pen_contour(self, mask: np.ndarray, estimated_depth: float = 0.5) -> Optional[np.ndarray]:
        """
        펜 컨투어 찾기

        Args:
            mask: 감지 마스크
            estimated_depth: 예상 거리 (m) - 동적 면적 조정에 사용
        """
        # 모폴로지 연산 (작은 커널로 작은 객체 보존)
        k_size = self.config.morph_kernel_size
        kernel = np.ones((k_size, k_size), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return None

        # 거리에 따른 동적 최소 면적
        min_area = self._get_dynamic_min_area(estimated_depth)

        # 펜 형태 필터링 (종횡비 확인)
        valid_contours = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area:
                continue

            # 최소 외접 사각형으로 종횡비 계산
            rect = cv2.minAreaRect(cnt)
            w, h = rect[1]
            if w == 0 or h == 0:
                continue

            aspect_ratio = max(w, h) / min(w, h)

            # 펜은 길쭉한 형태 (단, 멀리 있으면 비율 조건 완화)
            min_aspect = self.config.pen_aspect_ratio_min
            if estimated_depth > 1.0:
                min_aspect = max(1.5, min_aspect - 0.5)  # 멀면 조건 완화

            if aspect_ratio >= min_aspect:
                valid_contours.append((cnt, area, aspect_ratio))

        if not valid_contours:
            # 형태 조건 완화 - 가장 큰 컨투어 반환
            largest = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest) >= min_area:
                return largest
            return None

        # 가장 큰 유효 컨투어 반환
        valid_contours.sort(key=lambda x: x[1], reverse=True)
        return valid_contours[0][0]

    def _find_pen_endpoint_far_from_hand(self, contour: np.ndarray,
                                          grip_pos: Optional[Tuple[int, int]]) -> Tuple[int, int]:
        """
        손에서 가장 먼 펜 끝점 찾기

        Args:
            contour: 펜 컨투어
            grip_pos: 손 잡는 위치 (x, y) 또는 None

        Returns:
            (x, y) 펜의 보이는 끝점 (그리퍼가 잡을 위치)
        """
        points = contour.reshape(-1, 2)

        if grip_pos is None:
            # 손 위치 모르면 컨투어 중심에서 가장 먼 점
            center = np.mean(points, axis=0)
            distances = np.linalg.norm(points - center, axis=1)
        else:
            # 손 위치에서 가장 먼 점 = 펜의 반대쪽 끝
            grip = np.array(grip_pos)
            distances = np.linalg.norm(points - grip, axis=1)

        farthest_idx = np.argmax(distances)
        endpoint = points[farthest_idx]

        return (int(endpoint[0]), int(endpoint[1]))

    def _calculate_pen_direction_3d(self, contour: np.ndarray, depth_image: np.ndarray,
                                     grip_pos: Optional[Tuple[int, int]],
                                     color_image: np.ndarray) -> Tuple[np.ndarray, float, Tuple[int, int]]:
        """
        손 위치 기반 3D 펜 방향 계산

        손 위치 → 펜 보이는 끝점 = 펜 방향 벡터
        깊이 정보로 실제 3D 방향 계산

        Returns:
            direction: 3D 방향 벡터 (단위 벡터, 손→끝점 방향)
            angle: 2D 기울기 각도 (도)
            endpoint: 펜 끝점 픽셀 좌표 (그리퍼가 잡을 위치)
        """
        # 펜 끝점 찾기 (손에서 가장 먼 점)
        endpoint = self._find_pen_endpoint_far_from_hand(contour, grip_pos)

        if grip_pos is None:
            # 손 감지 안 됨 - 2D PCA 방향만 계산
            points = contour.reshape(-1, 2).astype(np.float32)
            mean = np.mean(points, axis=0)

            if len(points) < 5:
                return np.array([0, 0, 1]), 0, endpoint

            cov = np.cov((points - mean).T)
            eigenvalues, eigenvectors = np.linalg.eig(cov)
            idx = np.argmax(eigenvalues)
            direction_2d = eigenvectors[:, idx].real

            # 끝점 방향으로 정규화
            to_endpoint = np.array(endpoint) - mean
            if np.dot(direction_2d, to_endpoint) < 0:
                direction_2d = -direction_2d

            angle = np.arctan2(direction_2d[1], direction_2d[0]) * 180 / np.pi
            direction_3d = np.array([direction_2d[0], direction_2d[1], 0])
            direction_3d = direction_3d / (np.linalg.norm(direction_3d) + 1e-6)

            return direction_3d, angle, endpoint

        # 손 위치 있음 - 3D 방향 계산
        grip_x, grip_y = grip_pos
        end_x, end_y = endpoint

        # 2D 방향
        dx = end_x - grip_x
        dy = end_y - grip_y
        angle = np.arctan2(dy, dx) * 180 / np.pi

        # 3D 방향 계산 (깊이 차이 활용)
        h_img, w_img = depth_image.shape

        # 손 위치 깊이
        grip_x_c = np.clip(grip_x, 5, w_img - 6)
        grip_y_c = np.clip(grip_y, 5, h_img - 6)
        grip_depth_roi = depth_image[grip_y_c-5:grip_y_c+5, grip_x_c-5:grip_x_c+5]
        grip_depths = grip_depth_roi[grip_depth_roi > 0]
        grip_depth = np.median(grip_depths) / 1000.0 if len(grip_depths) > 0 else 0

        # 끝점 깊이
        end_x_c = np.clip(end_x, 5, w_img - 6)
        end_y_c = np.clip(end_y, 5, h_img - 6)
        end_depth_roi = depth_image[end_y_c-5:end_y_c+5, end_x_c-5:end_x_c+5]
        end_depths = end_depth_roi[end_depth_roi > 0]
        end_depth = np.median(end_depths) / 1000.0 if len(end_depths) > 0 else 0

        # 3D 좌표 변환
        grip_3d = self._pixel_to_camera(grip_x, grip_y, grip_depth) if grip_depth > 0 else None
        end_3d = self._pixel_to_camera(end_x, end_y, end_depth) if end_depth > 0 else None

        if grip_3d is not None and end_3d is not None:
            # 실제 3D 방향 벡터
            direction_3d = end_3d - grip_3d
            length = np.linalg.norm(direction_3d)
            if length > 0.001:  # 1mm 이상
                direction_3d = direction_3d / length
            else:
                direction_3d = np.array([0, 0, 1])
        else:
            # 깊이 정보 부족 - 2D 방향만
            direction_2d = np.array([dx, dy], dtype=np.float32)
            direction_2d = direction_2d / (np.linalg.norm(direction_2d) + 1e-6)
            direction_3d = np.array([direction_2d[0], direction_2d[1], 0])

        return direction_3d, angle, endpoint

    def _calculate_grasp_point(self, contour: np.ndarray, depth_image: np.ndarray,
                                endpoint: Tuple[int, int]) -> np.ndarray:
        """
        Grasp point 계산 - 펜 끝점 기반

        Args:
            contour: 펜 컨투어
            depth_image: 깊이 이미지
            endpoint: 펜 끝점 (손에서 가장 먼 점)

        Returns:
            grasp_point: 3D 좌표 (카메라 좌표계)
        """
        end_x, end_y = endpoint
        h_img, w_img = depth_image.shape

        # 범위 체크
        end_x = np.clip(end_x, 5, w_img - 6)
        end_y = np.clip(end_y, 5, h_img - 6)

        # 깊이 값
        depth_roi = depth_image[end_y-5:end_y+5, end_x-5:end_x+5]
        valid_depths = depth_roi[depth_roi > 0]

        if len(valid_depths) > 0:
            depth_m = np.median(valid_depths) / 1000.0
        else:
            depth_m = 0.5  # 기본값

        # 3D grasp point
        grasp_3d = self._pixel_to_camera(end_x, end_y, depth_m)

        return grasp_3d

    def _pixel_to_camera(self, cx: int, cy: int, depth_m: float) -> np.ndarray:
        """픽셀 -> 카메라 좌표계"""
        if self.simulation or self.intrinsics is None:
            fx, fy = 600, 600
            ppx, ppy = self.config.width / 2, self.config.height / 2

            x = (cx - ppx) * depth_m / fx
            y = (cy - ppy) * depth_m / fy
            z = depth_m

            return np.array([x, y, z])

        point = rs.rs2_deproject_pixel_to_point(self.intrinsics, [cx, cy], depth_m)
        return np.array(point)

    def detect(self) -> Optional[PenDetectionResult]:
        """
        펜 감지 메인 함수

        Returns:
            PenDetectionResult 또는 None
        """
        color_image, depth_image = self.get_frames()
        if color_image is None:
            return self._handle_no_frame()

        h, w = color_image.shape[:2]

        # 1. 손 마스크 생성 (깊이 정보 활용)
        hand_mask, hand_depth = self.hand_detector.create_hand_mask(
            color_image,
            depth_image=depth_image,
            expansion=self.config.hand_mask_expansion,
            depth_threshold=self.config.hand_depth_threshold
        )
        hand_occluding = np.sum(hand_mask > 0) > 1000

        self._debug_images['hand_mask'] = hand_mask
        self._hand_depth = hand_depth  # 디버깅용

        # 1.5 손 잡는 위치 가져오기 (펜 방향 계산에 사용)
        grip_pos = self.hand_detector.get_grip_position(color_image)
        self._grip_pos = grip_pos  # 디버깅용

        # 2. ROI 설정
        roi_size = int(min(h, w) * self.config.roi_ratio)
        roi_x = (w - roi_size) // 2
        roi_y = (h - roi_size) // 2

        # 3. HSV 변환 및 색상 마스크
        hsv = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)
        color_mask = self._create_color_mask(hsv)

        # 4. 깊이 마스크
        depth_mask = self._create_depth_mask(depth_image)

        # 5. 손 영역 제외한 마스크 결합
        combined_mask = color_mask & depth_mask & (~hand_mask)

        self._debug_images['combined_mask'] = combined_mask

        # 6. 예상 거리 추정 (이전 결과 또는 깊이 이미지 중앙값)
        estimated_depth = 0.5  # 기본값
        if self._last_result and self._last_result.depth > 0:
            estimated_depth = self._last_result.depth
        else:
            # 마스크 영역의 평균 깊이 사용
            valid_depths = depth_image[combined_mask > 0]
            valid_depths = valid_depths[(valid_depths > 100) & (valid_depths < 5000)]
            if len(valid_depths) > 0:
                estimated_depth = np.median(valid_depths) / 1000.0

        # 7. 펜 컨투어 찾기 (거리에 따른 동적 필터링)
        pen_contour = self._find_pen_contour(combined_mask, estimated_depth)

        # 8. 감지 성공 시
        if pen_contour is not None:
            return self._process_detection(pen_contour, depth_image, color_image, hand_occluding, grip_pos)

        # 9. 감지 실패 - Kalman 예측 사용
        return self._handle_no_detection(hand_occluding)

    def _process_detection(self, contour: np.ndarray, depth_image: np.ndarray,
                           color_image: np.ndarray, hand_occluding: bool,
                           grip_pos: Optional[Tuple[int, int]] = None) -> PenDetectionResult:
        """감지 성공 처리"""
        # 중심점
        M = cv2.moments(contour)
        if M["m00"] == 0:
            return self._handle_no_detection(hand_occluding)

        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        # 깊이
        depth_roi = depth_image[max(0, cy-5):cy+5, max(0, cx-5):cx+5]
        valid_depths = depth_roi[depth_roi > 0]

        if len(valid_depths) == 0:
            return self._handle_no_detection(hand_occluding)

        depth_m = np.median(valid_depths) / 1000.0

        # 3D 위치
        position = self._pixel_to_camera(cx, cy, depth_m)

        # Kalman 업데이트
        filtered_position = self.tracker.update(position)

        # 3D 방향 계산 (손 위치 기반)
        direction, angle, endpoint = self._calculate_pen_direction_3d(
            contour, depth_image, grip_pos, color_image
        )
        self.direction_tracker.update(direction)

        # 펜 끝점 저장 (시각화용)
        self._pen_endpoint = endpoint

        # Grasp point 계산 (펜 끝점 기반)
        grasp_point = self._calculate_grasp_point(contour, depth_image, endpoint)

        # 바운딩 박스
        x, y, w, h = cv2.boundingRect(contour)

        # 결과 생성
        result = PenDetectionResult(
            position=filtered_position,
            pixel=(cx, cy),
            depth=depth_m,
            grasp_point=grasp_point,
            direction=direction,
            angle=angle,
            state=DetectionState.DETECTED,
            confidence=1.0,
            hand_occluding=hand_occluding,
            bbox=(x, y, w, h)
        )

        self._last_result = result
        self._last_valid_detection_time = time.time()

        return result

    def _handle_no_detection(self, hand_occluding: bool) -> Optional[PenDetectionResult]:
        """감지 실패 처리 - Kalman 예측"""
        time_since_detection = time.time() - self._last_valid_detection_time

        # 예측 타임아웃 체크
        if time_since_detection > self.config.prediction_timeout:
            return PenDetectionResult(
                position=np.zeros(3),
                pixel=(0, 0),
                depth=0,
                grasp_point=np.zeros(3),
                direction=np.array([0, 0, 1]),
                angle=0,
                state=DetectionState.LOST,
                confidence=0,
                hand_occluding=hand_occluding
            )

        # Kalman 예측
        predicted_position = self.tracker.predict()
        predicted_direction = self.direction_tracker.predict()

        # 신뢰도 감소 (시간에 따라)
        confidence = max(0, 1.0 - time_since_detection / self.config.prediction_timeout)

        # 마지막 결과 기반 예측 결과 생성
        if self._last_result:
            return PenDetectionResult(
                position=predicted_position,
                pixel=self._last_result.pixel,
                depth=predicted_position[2],
                grasp_point=self._last_result.grasp_point,  # 마지막 grasp point 유지
                direction=predicted_direction,
                angle=self._last_result.angle,
                state=DetectionState.PREDICTED,
                confidence=confidence,
                hand_occluding=hand_occluding,
                bbox=self._last_result.bbox
            )

        return None

    def _handle_no_frame(self) -> Optional[PenDetectionResult]:
        """프레임 없음 처리"""
        if self._last_result and self._last_result.state != DetectionState.LOST:
            return self._handle_no_detection(False)
        return None

    def visualize(self, wait_key: int = 1, scale: float = 2.0) -> bool:
        """
        시각화

        Args:
            wait_key: cv2.waitKey 대기 시간 (ms)
            scale: 화면 확대 비율 (기본 2배)

        Returns:
            계속 여부 (q로 종료)
        """
        color_image, depth_image = self.get_frames()
        if color_image is None:
            return True

        result = self.detect()
        display = color_image.copy()

        # 손 마스크 오버레이 (반투명 빨간색)
        if 'hand_mask' in self._debug_images:
            hand_overlay = display.copy()
            hand_overlay[self._debug_images['hand_mask'] > 0] = [0, 0, 200]
            display = cv2.addWeighted(display, 0.7, hand_overlay, 0.3, 0)

        if result and result.state != DetectionState.LOST:
            cx, cy = result.pixel

            # 상태에 따른 색상
            if result.state == DetectionState.DETECTED:
                color = (0, 255, 0)  # 녹색 - 직접 감지
            else:
                color = (0, 165, 255)  # 주황색 - 예측

            # 중심점
            cv2.circle(display, (cx, cy), 8, color, 2)

            # 바운딩 박스
            if result.bbox:
                x, y, w, h = result.bbox
                cv2.rectangle(display, (x, y), (x+w, y+h), color, 2)

            # 손 위치 표시 (노란색 원)
            if hasattr(self, '_grip_pos') and self._grip_pos is not None:
                grip_x, grip_y = self._grip_pos
                cv2.circle(display, (grip_x, grip_y), 12, (0, 255, 255), 3)
                cv2.putText(display, "HAND", (grip_x + 15, grip_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

            # 펜 끝점 표시 (파란색 원) = Grasp Point
            if hasattr(self, '_pen_endpoint') and self._pen_endpoint is not None:
                end_x, end_y = self._pen_endpoint
                cv2.circle(display, (end_x, end_y), 12, (255, 100, 0), 3)
                cv2.putText(display, "GRASP", (end_x + 15, end_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 100, 0), 2)

                # 손 → 끝점 방향 화살표 (분홍색)
                if hasattr(self, '_grip_pos') and self._grip_pos is not None:
                    grip_x, grip_y = self._grip_pos
                    cv2.arrowedLine(display, (grip_x, grip_y), (end_x, end_y),
                                   (255, 0, 255), 3, tipLength=0.15)

            # 3D 방향 정보 표시
            dir_3d = result.direction
            info_lines = [
                f"State: {result.state.value}",
                f"Depth: {result.depth:.3f}m",
                f"Dir3D: [{dir_3d[0]:.2f}, {dir_3d[1]:.2f}, {dir_3d[2]:.2f}]",
                f"Grasp: [{result.grasp_point[0]:.3f}, {result.grasp_point[1]:.3f}, {result.grasp_point[2]:.3f}]",
                f"Hand: {'Yes' if result.hand_occluding else 'No'}"
            ]

            for i, line in enumerate(info_lines):
                cv2.putText(display, line, (10, 30 + i * 25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        else:
            cv2.putText(display, "LOST - No pen detected", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # 디버그: 마스크에서 찾은 픽셀 수 표시
            if 'combined_mask' in self._debug_images:
                mask_pixels = np.sum(self._debug_images['combined_mask'] > 0)
                cv2.putText(display, f"Mask pixels: {mask_pixels}", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                cv2.putText(display, f"MinArea: {self.config.min_contour_area}px", (10, 90),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # 화면 확대
        if scale != 1.0:
            new_w = int(display.shape[1] * scale)
            new_h = int(display.shape[0] * scale)
            display = cv2.resize(display, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        cv2.imshow("Advanced Pen Detector", display)

        # 마스크 디버그 창 (확대)
        if 'combined_mask' in self._debug_images:
            mask_display = self._debug_images['combined_mask']
            if scale != 1.0:
                mask_display = cv2.resize(mask_display, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
            cv2.imshow("Detection Mask", mask_display)

        key = cv2.waitKey(wait_key) & 0xFF
        return key != ord('q')

    def set_color_filter(self, hue_low: int, hue_high: int,
                        sat_low: int = 100, sat_high: int = 255,
                        val_low: int = 50, val_high: int = 255):
        """색상 필터 설정"""
        self.config.hue_low = hue_low
        self.config.hue_high = hue_high
        self.config.sat_low = sat_low
        self.config.sat_high = sat_high
        self.config.val_low = val_low
        self.config.val_high = val_high
        print(f"[AdvancedPenDetector] Color filter updated: H[{hue_low}-{hue_high}]")

    def get_position_camera(self) -> Optional[np.ndarray]:
        """펜 위치 (카메라 좌표계) - 호환성"""
        result = self.detect()
        if result and result.state != DetectionState.LOST:
            return result.position
        return None

    def get_grasp_point_camera(self) -> Optional[np.ndarray]:
        """Grasp point (카메라 좌표계)"""
        result = self.detect()
        if result and result.state != DetectionState.LOST:
            return result.grasp_point
        return None

    def get_pen_direction(self) -> Optional[np.ndarray]:
        """펜 방향 벡터"""
        result = self.detect()
        if result and result.state != DetectionState.LOST:
            return result.direction
        return None


# =============================================================================
# 테스트
# =============================================================================
def test_advanced_detector(scale: float = 2.0):
    """고급 펜 인식기 테스트"""
    print("=" * 60)
    print("AdvancedPenDetector 테스트")
    print("=" * 60)
    print(f"- 화면 확대: {scale}배 (--scale 옵션으로 조정)")
    print("- 손으로 펜을 가려보세요")
    print("- 녹색: 직접 감지, 주황색: Kalman 예측")
    print("- 'q' 키로 종료")
    print("=" * 60)

    detector = AdvancedPenDetector()

    if not detector.start():
        print("카메라 시작 실패")
        return

    try:
        while True:
            result = detector.detect()

            if result:
                status = f"[{result.state.value}]"
                if result.state != DetectionState.LOST:
                    pos = result.position
                    print(f"\r{status} Pos: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}] "
                          f"Conf: {result.confidence:.2f} Hand: {result.hand_occluding}", end="")
                else:
                    print(f"\r{status} 펜을 찾을 수 없습니다", end="")

            if not detector.visualize(wait_key=30, scale=scale):
                break

    except KeyboardInterrupt:
        pass

    detector.stop()
    cv2.destroyAllWindows()
    print("\n테스트 완료!")


def test_color_calibration():
    """색상 필터 캘리브레이션 도구"""
    print("=" * 60)
    print("색상 필터 캘리브레이션")
    print("=" * 60)
    print("트랙바로 HSV 값을 조절하세요")
    print("'s' 키: 현재 값 저장, 'q' 키: 종료")
    print("=" * 60)

    detector = AdvancedPenDetector()

    if not detector.start():
        print("카메라 시작 실패")
        return

    # 트랙바 창 생성
    cv2.namedWindow("Calibration")
    cv2.createTrackbar("H_Low", "Calibration", detector.config.hue_low, 180, lambda x: None)
    cv2.createTrackbar("H_High", "Calibration", detector.config.hue_high, 180, lambda x: None)
    cv2.createTrackbar("S_Low", "Calibration", detector.config.sat_low, 255, lambda x: None)
    cv2.createTrackbar("S_High", "Calibration", detector.config.sat_high, 255, lambda x: None)
    cv2.createTrackbar("V_Low", "Calibration", detector.config.val_low, 255, lambda x: None)
    cv2.createTrackbar("V_High", "Calibration", detector.config.val_high, 255, lambda x: None)

    try:
        while True:
            # 트랙바 값 읽기
            h_low = cv2.getTrackbarPos("H_Low", "Calibration")
            h_high = cv2.getTrackbarPos("H_High", "Calibration")
            s_low = cv2.getTrackbarPos("S_Low", "Calibration")
            s_high = cv2.getTrackbarPos("S_High", "Calibration")
            v_low = cv2.getTrackbarPos("V_Low", "Calibration")
            v_high = cv2.getTrackbarPos("V_High", "Calibration")

            detector.set_color_filter(h_low, h_high, s_low, s_high, v_low, v_high)

            if not detector.visualize(wait_key=30):
                break

            key = cv2.waitKey(1) & 0xFF
            if key == ord('s'):
                print(f"\n저장된 값: H[{h_low}-{h_high}], S[{s_low}-{s_high}], V[{v_low}-{v_high}]")

    except KeyboardInterrupt:
        pass

    detector.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    import sys
    import argparse

    parser = argparse.ArgumentParser(description="펜 감지기 테스트")
    parser.add_argument("mode", nargs="?", default="test",
                       choices=["test", "calibrate"],
                       help="실행 모드: test(기본) 또는 calibrate")
    parser.add_argument("--scale", type=float, default=2.0,
                       help="화면 확대 비율 (기본: 2.0)")

    args = parser.parse_args()

    if args.mode == "calibrate":
        test_color_calibration()
    else:
        test_advanced_detector(scale=args.scale)
