"""
Direct 환경 보상 구조 테스트 스크립트

실제 환경을 실행하지 않고 보상 계산 로직만 테스트합니다.
"""
import torch

# =============================================================================
# 보상 스케일 (e0509_direct_env.py에서 복사)
# =============================================================================
rew_scale_approach_dist = -2.0
rew_scale_approach_progress = 5.0
rew_scale_align_dist = -1.0
rew_scale_align_dot = 2.0
rew_scale_grasp_dist = -3.0
rew_scale_grasp_dot = 1.0
rew_scale_success = 100.0
rew_scale_phase_transition = 10.0
rew_scale_action = -0.001

# 단계 전환 조건
APPROACH_TO_ALIGN_DIST = 0.10
ALIGN_TO_GRASP_DOT = -0.8
SUCCESS_DIST = 0.02
SUCCESS_DOT = -0.9

# 단계 정의
PHASE_APPROACH = 0
PHASE_ALIGN = 1
PHASE_GRASP = 2


def calculate_reward(distance: float, dot: float, phase: int, prev_distance: float = None) -> dict:
    """
    보상 계산 시뮬레이션

    Args:
        distance: 그리퍼-펜 거리 (m)
        dot: gripper_z · pen_z (정렬도, -1이 완벽)
        phase: 현재 단계 (0=APPROACH, 1=ALIGN, 2=GRASP)
        prev_distance: 이전 거리 (progress 계산용)

    Returns:
        dict: 보상 상세 내역
    """
    result = {
        "phase": ["APPROACH", "ALIGN", "GRASP"][phase],
        "distance": distance,
        "dot": dot,
        "rewards": {},
        "total": 0.0,
        "phase_transition": None,
        "success": False,
    }

    total = 0.0

    # === 단계별 보상 ===
    if phase == PHASE_APPROACH:
        # 거리 페널티
        dist_penalty = rew_scale_approach_dist * distance
        result["rewards"]["dist_penalty"] = dist_penalty
        total += dist_penalty

        # 거리 감소 보상
        if prev_distance is not None:
            progress = prev_distance - distance
            if progress > 0:
                progress_reward = rew_scale_approach_progress * progress
                result["rewards"]["progress"] = progress_reward
                total += progress_reward

        # 단계 전환 체크
        if distance < APPROACH_TO_ALIGN_DIST:
            result["phase_transition"] = "APPROACH → ALIGN"
            transition_reward = rew_scale_phase_transition
            result["rewards"]["transition"] = transition_reward
            total += transition_reward

    elif phase == PHASE_ALIGN:
        # 거리 유지 페널티
        dist_penalty = rew_scale_align_dist * distance
        result["rewards"]["dist_penalty"] = dist_penalty
        total += dist_penalty

        # 정렬 보상
        align_reward_raw = (-dot - 0.5) / 0.5  # dot=-0.5 → 0, dot=-1 → 1
        align_reward_clamped = max(0, align_reward_raw)
        align_reward = rew_scale_align_dot * align_reward_clamped
        result["rewards"]["align (raw)"] = align_reward_raw
        result["rewards"]["align (clamped)"] = align_reward
        total += align_reward

        # 단계 전환 체크
        if dot < ALIGN_TO_GRASP_DOT:
            result["phase_transition"] = "ALIGN → GRASP"
            transition_reward = rew_scale_phase_transition
            result["rewards"]["transition"] = transition_reward
            total += transition_reward

    elif phase == PHASE_GRASP:
        # 거리 페널티 (강화)
        dist_penalty = rew_scale_grasp_dist * distance
        result["rewards"]["dist_penalty"] = dist_penalty
        total += dist_penalty

        # 정렬 유지 보상
        dot_reward = rew_scale_grasp_dot * (-dot)
        result["rewards"]["dot_maintain"] = dot_reward
        total += dot_reward

        # 성공 체크
        if distance < SUCCESS_DIST and dot < SUCCESS_DOT:
            result["success"] = True
            success_reward = rew_scale_success
            result["rewards"]["success"] = success_reward
            total += success_reward

    result["total"] = total
    return result


def print_reward(result: dict):
    """보상 결과 출력"""
    print(f"\n{'='*60}")
    print(f"단계: {result['phase']} | 거리: {result['distance']:.3f}m | dot: {result['dot']:.2f}")
    print(f"{'-'*60}")
    for name, value in result["rewards"].items():
        print(f"  {name:20s}: {value:+.4f}")
    print(f"{'-'*60}")
    print(f"  {'총 보상':20s}: {result['total']:+.4f}")
    if result["phase_transition"]:
        print(f"  >>> 단계 전환: {result['phase_transition']}")
    if result["success"]:
        print(f"  >>> 성공!")
    print(f"{'='*60}")


def main():
    print("\n" + "="*60)
    print("Direct 환경 보상 구조 테스트")
    print("="*60)

    # =============================================================================
    # 테스트 시나리오
    # =============================================================================

    print("\n\n### 시나리오 1: APPROACH 단계 ###")

    # 시작 (멀리)
    print_reward(calculate_reward(distance=0.30, dot=0.0, phase=PHASE_APPROACH))

    # 접근 중 (거리 감소)
    print_reward(calculate_reward(distance=0.20, dot=0.0, phase=PHASE_APPROACH, prev_distance=0.25))

    # 전환 직전
    print_reward(calculate_reward(distance=0.11, dot=0.0, phase=PHASE_APPROACH, prev_distance=0.15))

    # 전환 순간 (거리 < 10cm)
    print_reward(calculate_reward(distance=0.09, dot=0.0, phase=PHASE_APPROACH, prev_distance=0.11))


    print("\n\n### 시나리오 2: ALIGN 단계 ###")

    # 정렬 시작 (dot = 0, 아직 정렬 안됨)
    print_reward(calculate_reward(distance=0.08, dot=0.0, phase=PHASE_ALIGN))

    # 정렬 중 (dot = -0.5)
    print_reward(calculate_reward(distance=0.08, dot=-0.5, phase=PHASE_ALIGN))

    # 정렬 좋음 (dot = -0.7)
    print_reward(calculate_reward(distance=0.06, dot=-0.7, phase=PHASE_ALIGN))

    # 전환 순간 (dot < -0.8)
    print_reward(calculate_reward(distance=0.05, dot=-0.85, phase=PHASE_ALIGN))


    print("\n\n### 시나리오 3: GRASP 단계 ###")

    # GRASP 시작
    print_reward(calculate_reward(distance=0.05, dot=-0.85, phase=PHASE_GRASP))

    # 더 접근
    print_reward(calculate_reward(distance=0.03, dot=-0.9, phase=PHASE_GRASP))

    # 성공 직전
    print_reward(calculate_reward(distance=0.025, dot=-0.92, phase=PHASE_GRASP))

    # 성공!
    print_reward(calculate_reward(distance=0.015, dot=-0.95, phase=PHASE_GRASP))


    print("\n\n### 문제 상황 테스트 ###")

    # APPROACH에서 거리가 줄지 않을 때
    print("\n--- APPROACH: 제자리 (progress=0) ---")
    print_reward(calculate_reward(distance=0.20, dot=0.0, phase=PHASE_APPROACH, prev_distance=0.20))

    # ALIGN에서 dot이 양수일 때
    print("\n--- ALIGN: dot이 양수 (반대 방향) ---")
    print_reward(calculate_reward(distance=0.08, dot=0.5, phase=PHASE_ALIGN))

    # ALIGN에서 거리가 멀어질 때
    print("\n--- ALIGN: 거리 멀어짐 ---")
    print_reward(calculate_reward(distance=0.15, dot=-0.7, phase=PHASE_ALIGN))


if __name__ == "__main__":
    main()
