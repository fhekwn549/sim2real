"""
Pen Grasp RL 학습 스크립트 (Curriculum Learning 지원)

이 스크립트는 펜 잡기 강화학습 에이전트를 학습시킵니다.
RSL-RL 라이브러리의 PPO 알고리즘을 사용합니다.

=== Curriculum Learning (v3) ===
Stage 1: 거리 < 10cm, dot < -0.7  (85% 성공 시 전환)
Stage 2: 거리 < 5cm,  dot < -0.85 (90% 성공 시 전환)
Stage 3: 거리 < 2cm,  dot < -0.95 (최종 목표: 95%)

사용법:
    # 기본 실행 (headless 모드, 4096개 환경)
    python train.py --headless --num_envs 4096 --max_iterations 5000

    # Curriculum Learning (v3 환경)
    python train.py --headless --num_envs 4096 --max_iterations 100000 --env_version v3

    # 이전 학습 이어서 하기 (resume)
    python train.py --headless --num_envs 4096 --resume --checkpoint /path/to/model_3500.pt

주의:
    - 학습은 별도 터미널에서 실행해야 합니다 (Claude 터미널은 타임아웃 있음)
    - GPU 메모리에 따라 num_envs 조절 필요
"""
import argparse
import glob
import os
import shutil
import sys

# 프로젝트 경로 추가 (envs 모듈 import를 위해)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from isaaclab.app import AppLauncher

# ============================================================
# 명령행 인자 파싱
# ============================================================
parser = argparse.ArgumentParser(description="펜 잡기 정책 학습")
parser.add_argument("--num_envs", type=int, default=4096,
                    help="병렬 환경 개수 (기본: 4096)")
parser.add_argument("--max_iterations", type=int, default=100000,
                    help="최대 학습 반복 횟수 (기본: 100000)")
parser.add_argument("--resume", action="store_true",
                    help="이전 학습 이어서 하기")
parser.add_argument("--checkpoint", type=str, default=None,
                    help="이어서 학습할 체크포인트 파일 경로 (예: model_3500.pt)")
parser.add_argument("--env_version", type=str, default="v3", choices=["v1", "v2", "v3"],
                    help="환경 버전 (v1: 기존, v2: reach 기반, v3: Curriculum Learning)")
parser.add_argument("--check_interval", type=int, default=1000,
                    help="성공률 체크 간격 (iteration 단위, 기본: 1000)")

# AppLauncher 인자 추가 (--headless 등)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

# ============================================================
# Isaac Sim 실행
# ============================================================
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# Isaac Sim 실행 후 import (순서 중요!)
import torch

# 환경 버전에 따라 import
if args.env_version == "v3":
    from envs.pen_grasp_env_v3 import (
        PenGraspEnv, PenGraspEnvCfg,
        set_curriculum_stage, get_curriculum_stage,
        save_curriculum_state, load_curriculum_state,
        CURRICULUM_STAGES
    )
    print("환경: v3 (Curriculum Learning)")
elif args.env_version == "v2":
    from envs.pen_grasp_env_v2 import PenGraspEnv, PenGraspEnvCfg
    print("환경: v2 (reach 기반 단순화)")
else:
    from envs.pen_grasp_env import PenGraspEnv, PenGraspEnvCfg
    print("환경: v1 (기존)")

# RSL-RL 라이브러리
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
    RslRlVecEnvWrapper
)
from rsl_rl.runners import OnPolicyRunner


def find_and_save_best_model(log_dir: str):
    """
    TensorBoard 로그에서 best iteration을 찾아 model_best.pt로 저장
    """
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError:
        print("경고: tensorboard가 설치되지 않아 best model을 찾을 수 없습니다.")
        return

    event_files = glob.glob(os.path.join(log_dir, "**", "events.out.tfevents.*"), recursive=True)
    if not event_files:
        print(f"경고: {log_dir}에서 TensorBoard 이벤트 파일을 찾을 수 없습니다.")
        return

    event_file = max(event_files, key=os.path.getmtime)
    event_dir = os.path.dirname(event_file)

    print(f"\nBest model 탐색 중...")
    print(f"  로그 경로: {event_dir}")

    ea = EventAccumulator(event_dir)
    ea.Reload()

    scalar_tags = ea.Tags().get('scalars', [])
    reward_tag = None
    for tag in scalar_tags:
        if 'reward' in tag.lower() and 'mean' in tag.lower():
            reward_tag = tag
            break

    if reward_tag is None:
        for tag in scalar_tags:
            if 'reward' in tag.lower():
                reward_tag = tag
                break

    if reward_tag is None:
        print(f"  경고: reward 관련 태그를 찾을 수 없습니다.")
        return

    events = ea.Scalars(reward_tag)
    if not events:
        print("  경고: reward 데이터가 없습니다.")
        return

    best_event = max(events, key=lambda e: e.value)
    best_step = best_event.step
    best_reward = best_event.value

    print(f"  Best iteration: {best_step} (reward: {best_reward:.4f})")

    save_interval = 100
    closest_saved_step = round(best_step / save_interval) * save_interval

    model_path = os.path.join(log_dir, f"model_{closest_saved_step}.pt")
    if not os.path.exists(model_path):
        model_files = glob.glob(os.path.join(log_dir, "model_*.pt"))
        if not model_files:
            print(f"  경고: {log_dir}에서 모델 파일을 찾을 수 없습니다.")
            return

        def get_step(path):
            name = os.path.basename(path)
            try:
                return int(name.replace("model_", "").replace(".pt", ""))
            except ValueError:
                return -1

        model_files_with_steps = [(f, get_step(f)) for f in model_files if get_step(f) >= 0]
        if not model_files_with_steps:
            print("  경고: 유효한 모델 파일이 없습니다.")
            return

        model_path, closest_saved_step = min(model_files_with_steps, key=lambda x: abs(x[1] - best_step))

    best_model_path = os.path.join(log_dir, "model_best.pt")
    shutil.copy2(model_path, best_model_path)

    print(f"  Best model 저장: {best_model_path}")
    print(f"  (iteration {closest_saved_step}의 모델 복사)")


class CurriculumTrainer:
    """Curriculum Learning을 지원하는 학습 관리자"""

    def __init__(self, env, runner, log_dir: str, check_interval: int = 1000):
        self.env = env
        self.runner = runner
        self.log_dir = log_dir
        self.check_interval = check_interval

        # Curriculum 상태
        self.curriculum_state_path = os.path.join(log_dir, "curriculum_state.json")
        self.current_stage = 1
        self.success_rates = {1: 0.0, 2: 0.0, 3: 0.0}
        self.total_iterations = 0

        # 성공 추적 버퍼 (최근 N 에피소드)
        self.success_history = []
        self.success_window = 100  # 최근 100개 에피소드로 성공률 계산

    def load_state(self):
        """이전 curriculum 상태 로드"""
        if os.path.exists(self.curriculum_state_path):
            state = load_curriculum_state(self.curriculum_state_path)
            self.current_stage = state.get("current_stage", 1)
            self.success_rates = state.get("success_rates", {1: 0.0, 2: 0.0, 3: 0.0})
            self.total_iterations = state.get("last_iteration", 0)
            set_curriculum_stage(self.current_stage)
            print(f"[Curriculum] 상태 로드됨: Stage {self.current_stage}, Iteration {self.total_iterations}")
        else:
            set_curriculum_stage(1)
            print("[Curriculum] 새로운 학습 시작: Stage 1")

    def save_state(self):
        """현재 curriculum 상태 저장"""
        save_curriculum_state(
            self.curriculum_state_path,
            self.current_stage,
            self.success_rates,
            self.total_iterations
        )

    def check_and_advance_stage(self, current_success_rate: float):
        """성공률 체크 후 stage 전환 여부 결정"""
        if self.current_stage >= 3:
            # 최종 stage - 전환 없음
            return False

        threshold = CURRICULUM_STAGES[self.current_stage]["success_rate_to_advance"]

        print(f"[Curriculum] Stage {self.current_stage} 성공률: {current_success_rate*100:.1f}% "
              f"(목표: {threshold*100:.0f}%)")

        if current_success_rate >= threshold:
            self.current_stage += 1
            set_curriculum_stage(self.current_stage)
            print(f"[Curriculum] Stage {self.current_stage}로 전환!")
            self.success_history = []  # 히스토리 리셋
            self.save_state()
            return True

        return False

    def compute_success_rate(self) -> float:
        """
        환경에서 현재 성공률 계산

        에피소드 종료 시 성공 여부를 추적해서 평균 계산
        """
        # 환경의 성공률 직접 가져오기
        if hasattr(self.env, 'unwrapped'):
            base_env = self.env.unwrapped
        else:
            base_env = self.env

        if hasattr(base_env, 'get_success_rate'):
            return base_env.get_success_rate()

        return 0.0

    def train(self, max_iterations: int, resume_iteration: int = 0):
        """Curriculum Learning으로 학습 실행"""
        self.total_iterations = resume_iteration
        remaining = max_iterations - resume_iteration

        print(f"\n{'='*60}")
        print(f"Curriculum Learning 시작")
        print(f"  현재 Stage: {self.current_stage}")
        print(f"  남은 Iterations: {remaining}")
        print(f"  체크 간격: {self.check_interval}")
        print(f"{'='*60}\n")

        while remaining > 0:
            # 이번에 학습할 iteration 수
            iterations_this_round = min(self.check_interval, remaining)

            # 학습 실행
            self.runner.learn(
                num_learning_iterations=iterations_this_round,
                init_at_random_ep_len=(self.total_iterations == 0)
            )

            self.total_iterations += iterations_this_round
            remaining -= iterations_this_round

            # 성공률 계산 및 출력
            success_rate = self.compute_success_rate()
            self.success_rates[self.current_stage] = success_rate

            print(f"\n{'='*60}")
            print(f"[Iteration {self.total_iterations}] Stage {self.current_stage}")
            print(f"  성공률: {success_rate*100:.1f}%")
            print(f"  Stage 조건: 거리 < {CURRICULUM_STAGES[self.current_stage]['distance_threshold']*100:.0f}cm, "
                  f"dot < {CURRICULUM_STAGES[self.current_stage]['dot_threshold']:.2f}")
            print(f"{'='*60}\n")

            # Stage 전환 체크
            self.check_and_advance_stage(success_rate)

            # 상태 저장
            self.save_state()

            # 최종 stage에서 목표 달성 시 조기 종료
            if self.current_stage == 3 and success_rate >= 0.95:
                print(f"\n[Curriculum] 최종 목표 달성! 성공률 {success_rate*100:.1f}%")
                break

        print(f"\n[Curriculum] 학습 완료. 총 {self.total_iterations} iterations")


def main():
    """메인 학습 함수"""

    # ============================================================
    # 환경 설정
    # ============================================================
    env_cfg = PenGraspEnvCfg()
    env_cfg.scene.num_envs = args.num_envs

    # 환경 생성
    env = PenGraspEnv(cfg=env_cfg)

    # ============================================================
    # PPO 하이퍼파라미터 설정
    # ============================================================
    agent_cfg = RslRlOnPolicyRunnerCfg(
        seed=42,
        device="cuda:0",
        num_steps_per_env=24,
        max_iterations=args.max_iterations,
        save_interval=100,
        experiment_name="pen_grasp",
        run_name="curriculum_v3" if args.env_version == "v3" else "ppo_run",
        logger="tensorboard",
        obs_groups={"policy": ["policy"]},

        policy=RslRlPpoActorCriticCfg(
            init_noise_std=0.3,
            actor_hidden_dims=[256, 256, 128],
            critic_hidden_dims=[256, 256, 128],
            activation="elu",
            actor_obs_normalization=False,
            critic_obs_normalization=False,
        ),

        algorithm=RslRlPpoAlgorithmCfg(
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.2,
            entropy_coef=0.01,
            num_learning_epochs=5,
            num_mini_batches=4,
            learning_rate=3e-4,
            schedule="adaptive",
            gamma=0.99,
            lam=0.95,
            desired_kl=0.01,
            max_grad_norm=1.0,
        ),
    )

    # ============================================================
    # 학습 실행
    # ============================================================
    # RSL-RL 래퍼로 환경 감싸기
    wrapped_env = RslRlVecEnvWrapper(env)

    # 로그 디렉토리
    log_dir = "./logs/pen_grasp"

    # OnPolicyRunner 생성
    runner = OnPolicyRunner(
        wrapped_env,
        agent_cfg.to_dict(),
        log_dir=log_dir,
        device=agent_cfg.device
    )

    # 체크포인트에서 이어서 학습하기
    resume_iteration = 0
    if args.resume and args.checkpoint:
        if os.path.exists(args.checkpoint):
            print(f"체크포인트 로드 중: {args.checkpoint}")
            runner.load(args.checkpoint)
            checkpoint_name = os.path.basename(args.checkpoint)
            try:
                resume_iteration = int(checkpoint_name.replace("model_", "").replace(".pt", ""))
                print(f"  이전 학습 iteration: {resume_iteration}")
            except ValueError:
                print("  경고: iteration 번호를 추출할 수 없습니다. 0부터 시작합니다.")
        else:
            print(f"경고: 체크포인트 파일을 찾을 수 없습니다: {args.checkpoint}")
            print("새로운 학습을 시작합니다.")

    # 학습 시작
    print("=" * 60)
    print(f"펜 잡기 강화학습 {'재개' if args.resume else '시작'}")
    print("=" * 60)
    print(f"  환경 버전: {args.env_version}")
    print(f"  병렬 환경 수: {args.num_envs}")
    print(f"  최대 반복 횟수: {args.max_iterations}")
    print(f"  초기 노이즈 std: {agent_cfg.policy.init_noise_std}")
    print(f"  학습률: {agent_cfg.algorithm.learning_rate}")
    if args.resume:
        print(f"  체크포인트: {args.checkpoint}")
    print("=" * 60)

    # Curriculum Learning (v3) 또는 일반 학습
    if args.env_version == "v3":
        trainer = CurriculumTrainer(
            env=env,
            runner=runner,
            log_dir=log_dir,
            check_interval=args.check_interval
        )

        # 이전 curriculum 상태 로드
        if args.resume:
            trainer.load_state()

        # 학습 실행
        trainer.train(max_iterations=args.max_iterations, resume_iteration=resume_iteration)
    else:
        # 일반 학습
        runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=not args.resume)

    # ============================================================
    # 학습 완료 및 정리
    # ============================================================
    runner.save("final_model")
    print("\n학습 완료! 모델이 저장되었습니다.")

    # Best model 찾아서 저장
    find_and_save_best_model(log_dir)

    # 환경 정리
    wrapped_env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
