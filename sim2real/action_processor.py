"""
Action Processor for Sim2Real

학습된 policy의 action을 후처리하여 더 부드럽고 안정적인 로봇 제어를 수행합니다.

지원 기능:
1. Scale by Distance: 목표에 가까울수록 action 크기 축소 (진동 방지)
2. Smoothing: 이전 action과 현재 action을 혼합 (급격한 변화 방지)
3. Dead Zone: 목표 근처에서 action을 0으로 (선택적)

사용 예시:
    from action_processor import ActionProcessor

    processor = ActionProcessor(scale_by_dist=True)

    while running:
        dist = get_distance_to_target()
        raw_action = policy(observation)
        action = processor.process(raw_action, dist)
        robot.send_command(action)
"""

import numpy as np
from typing import Optional, Union

# Type hint for array-like
ArrayLike = Union[np.ndarray, list]


class ActionProcessor:
    """Action 후처리 클래스 (Sim2Real용)

    시뮬레이션에서 검증된 후처리 로직을 실제 로봇에 동일하게 적용합니다.
    """

    def __init__(
        self,
        smooth_alpha: float = 1.0,
        dead_zone_cm: float = 0.0,
        scale_by_dist: bool = True,
        scale_min: float = 0.1,
        scale_range_cm: float = 10.0,
    ):
        """
        Args:
            smooth_alpha: smoothing factor (1.0=no smooth, 0.5=half smooth)
                          new = alpha * current + (1-alpha) * previous
            dead_zone_cm: dead zone 거리 (cm). 이 거리 이내면 action=0
            scale_by_dist: True면 거리에 비례해서 action 축소
            scale_min: scale_by_dist 시 최소 scale (0~1)
            scale_range_cm: 이 거리(cm)에서 scale=1.0, 이보다 가까우면 축소
        """
        self.smooth_alpha = smooth_alpha
        self.dead_zone = dead_zone_cm / 100.0  # cm → m
        self.scale_by_dist = scale_by_dist
        self.scale_min = scale_min
        self.scale_range = scale_range_cm / 100.0  # cm → m
        self.prev_action: Optional[np.ndarray] = None

    def process(self, action: ArrayLike, dist: float) -> np.ndarray:
        """Action 후처리

        Args:
            action: raw action from policy [action_dim] (numpy array or list)
            dist: 목표까지 거리 (meters)

        Returns:
            processed action [action_dim] (numpy array)
        """
        # numpy array로 변환
        if not isinstance(action, np.ndarray):
            action = np.array(action, dtype=np.float32)

        processed = action.copy()

        # 1. Dead Zone: 거리가 threshold 이내면 action=0
        if self.dead_zone > 0 and dist < self.dead_zone:
            processed = np.zeros_like(processed)

        # 2. Scale by Distance: 거리에 비례해서 action 축소
        if self.scale_by_dist:
            # scale = clamp(dist / scale_range, min, 1.0)
            scale = np.clip(dist / self.scale_range, self.scale_min, 1.0)
            processed = processed * scale

        # 3. Smoothing
        if self.smooth_alpha < 1.0:
            if self.prev_action is None:
                self.prev_action = processed.copy()
            else:
                processed = self.smooth_alpha * processed + (1 - self.smooth_alpha) * self.prev_action
                self.prev_action = processed.copy()

        return processed

    def reset(self):
        """Reset processor state (에피소드 시작 시 호출)"""
        self.prev_action = None

    def get_status_str(self) -> str:
        """현재 설정 상태 문자열 반환"""
        status = []
        if self.smooth_alpha < 1.0:
            status.append(f"Smoothing(α={self.smooth_alpha})")
        if self.dead_zone > 0:
            status.append(f"DeadZone({self.dead_zone*100:.1f}cm)")
        if self.scale_by_dist:
            status.append(f"ScaleByDist(min={self.scale_min}, range={self.scale_range*100:.0f}cm)")
        return ", ".join(status) if status else "None"

    def __repr__(self) -> str:
        return f"ActionProcessor({self.get_status_str()})"


# =============================================================================
# 사용 예시
# =============================================================================
if __name__ == "__main__":
    # 테스트
    processor = ActionProcessor(scale_by_dist=True, scale_min=0.1, scale_range_cm=10.0)
    print(f"Processor: {processor}")

    # 다양한 거리에서 테스트
    action = np.array([0.1, 0.2, 0.3])

    print("\n거리별 action scale 테스트:")
    for dist_cm in [20, 15, 10, 7, 5, 3, 1]:
        dist_m = dist_cm / 100.0
        processed = processor.process(action, dist_m)
        scale = np.linalg.norm(processed) / np.linalg.norm(action)
        print(f"  dist={dist_cm:2d}cm → scale={scale:.2f}, action={processed}")
