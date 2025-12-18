# Pen Grasp RL - Docker 사용 가이드

## 개요

이 프로젝트는 Docker를 사용해 Isaac Sim/Lab 환경을 실행합니다.
**노트북에 Isaac Sim을 직접 설치할 필요 없이**, Docker만 있으면 학습 환경을 바로 구축할 수 있습니다.

### Docker의 장점
- 호스트 PC의 Isaac Sim/Lab 버전과 무관하게 동작
- 동일한 환경을 모든 팀원이 공유 가능
- 노트북 GPU 성능만 활용 (환경은 컨테이너 내부)
- 코드 수정 시 바로 반영 (볼륨 마운트)

---

## 1. 사전 요구사항

| 항목 | 요구사항 |
|------|----------|
| OS | Ubuntu 22.04 |
| GPU | NVIDIA (RTX 3070 이상 권장) |
| NVIDIA Driver | 535 이상 |
| Docker | 26.0.0 이상 |
| Docker Compose | 2.25.0 이상 |

---

## 2. 최초 설정 (1회만)

### 2.1 NVIDIA 드라이버 설치
```bash
sudo apt update
sudo ubuntu-drivers autoinstall
sudo reboot
```

드라이버 확인:
```bash
nvidia-smi
```

### 2.2 Docker 설치
```bash
# Docker 설치
sudo apt install docker.io docker-compose-v2

# 현재 사용자를 docker 그룹에 추가 (sudo 없이 사용)
sudo usermod -aG docker $USER
newgrp docker
```

### 2.3 NVIDIA Container Toolkit 설치
```bash
# 저장소 추가
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

# 설치
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

# Docker 설정
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

### 2.4 NGC 로그인

Isaac Sim Docker 이미지는 NVIDIA NGC에서 제공됩니다.

1. [NGC 웹사이트](https://ngc.nvidia.com/) 가입 (무료)
2. 로그인 후 우측 상단 프로필 → **Setup**
3. **Generate API Key** 클릭 (기본 권한으로 생성)
4. 생성된 키 복사 (한 번만 표시되므로 꼭 저장!)

```bash
docker login nvcr.io
# Username: $oauthtoken  (그대로 입력)
# Password: <발급받은 API Key>
```

---

## 3. 프로젝트 설정

### 3.1 프로젝트 폴더 받기
```bash
# 1. Isaac Lab 공식 레포 클론
cd ~
git clone https://github.com/isaac-sim/IsaacLab.git

# 2. 팀 프로젝트 클론 (pen_grasp_rl + 로봇 USD 파일 포함)
git clone https://github.com/KERNEL3-2/CoWriteBotRL.git

# 3. pen_grasp_rl을 IsaacLab 안으로 복사
cp -r ~/CoWriteBotRL/pen_grasp_rl ~/IsaacLab/

# 4. 로봇 USD 파일을 홈 디렉토리에 복사 (Docker 마운트용)
cp -r ~/CoWriteBotRL/e0509_gripper_isaac ~/
```

> **참고**:
> - 로봇 USD 파일(`first_control.usd`)은 `pen_grasp_rl/models/`에 포함되어 있습니다.
> - `e0509_gripper_isaac/`는 로봇 모델이 참조하는 추가 USD 파일입니다.

### 3.2 Docker 설정 파일 수정
```bash
cd ~/IsaacLab/docker
```

`docker-compose.yaml` 파일을 열고 `x-default-isaac-lab-volumes` 섹션에 아래 내용 추가:

```yaml
    # Pen Grasp RL project
  - type: bind
    source: ../pen_grasp_rl
    target: ${DOCKER_ISAACLAB_PATH}/pen_grasp_rl
    # Logs - 호스트에서 바로 접근 가능하도록
  - type: bind
    source: ../logs
    target: ${DOCKER_ISAACLAB_PATH}/logs
    # 로봇 USD 참조 파일 (first_control.usd가 참조함)
  - type: bind
    source: ~/e0509_gripper_isaac
    target: /workspace/e0509_gripper_isaac
```

> **참고**:
> - logs를 bind mount하면 학습 결과를 호스트에서 바로 확인/복사할 수 있습니다.
> - e0509_gripper_isaac은 로봇 USD 파일이 참조하는 경로이므로 반드시 마운트해야 합니다.

### 3.3 Docker 이미지 빌드 및 실행
```bash
cd ~/IsaacLab

# 필수 파일/폴더 생성
touch docker/.isaac-lab-docker-history
mkdir -p logs  # 볼륨 마운트용 logs 폴더 생성

# 이미지 빌드 + 컨테이너 시작 (최초 1회, 약 30분 소요)
./docker/container.py start
```

> **참고**: `docker compose` 명령어를 직접 사용하면 환경변수 오류가 발생할 수 있습니다. 반드시 `container.py` 스크립트를 사용하세요.

---

## 4. 학습 실행

### 4.1 컨테이너 진입
```bash
cd ~/IsaacLab

# 컨테이너 진입
./docker/container.py enter base
```

### 4.2 의존성 설치 (컨테이너 내부, 최초 1회)
```bash
cd /workspace/isaaclab
./pen_grasp_rl/docker_setup.sh
```

### 4.3 학습 실행 (컨테이너 내부)
```bash
# Headless 학습 (GUI 없이)
# RTX 5080 (16GB VRAM) 기준 num_envs=8192 권장
python pen_grasp_rl/scripts/train.py --headless --num_envs 8192 --max_iterations 10000

# 학습 결과는 /workspace/isaaclab/logs/pen_grasp 에 저장됨
```

### 4.4 컨테이너 종료
```bash
# 컨테이너 내부에서 나가기
exit

# 컨테이너 중지
cd ~/IsaacLab
./docker/container.py stop
```

---

## 5. 일상적인 사용

### 매일 작업 시작할 때
```bash
cd ~/IsaacLab
./docker/container.py start        # 컨테이너 시작
./docker/container.py enter base   # 컨테이너 진입

# 컨테이너 내부
cd /workspace/isaaclab
python pen_grasp_rl/scripts/train.py --headless --num_envs 8192
```

### 작업 끝날 때
```bash
exit  # 컨테이너에서 나가기
./docker/container.py stop  # 컨테이너 중지
```

---

## 6. 코드 업데이트 (git pull)

GitHub에서 최신 코드를 받으려면 **호스트(Docker 밖)**에서 실행:

```bash
# 1. CoWriteBotRL 폴더에서 git pull
cd ~/CoWriteBotRL
git pull

# 2. 업데이트된 파일 복사 (Docker가 root 소유권으로 만들어서 sudo 필요)
sudo cp -r ~/CoWriteBotRL/pen_grasp_rl ~/IsaacLab/
sudo cp -r ~/CoWriteBotRL/e0509_gripper_isaac ~/

# 3. 확인
ls -la ~/IsaacLab/pen_grasp_rl/models/first_control.usd
```

> **참고**: Docker 컨테이너는 볼륨 마운트로 `~/IsaacLab/pen_grasp_rl`을 참조하므로, 호스트에서 복사하면 컨테이너에 자동 반영됩니다.

---

## 7. 코드 직접 수정

코드는 **호스트(노트북)에서 수정**하면 됩니다.
볼륨 마운트 되어 있어서 컨테이너 안에 바로 반영됩니다.

| 호스트 경로 | 컨테이너 경로 |
|-------------|---------------|
| `~/IsaacLab/pen_grasp_rl/` | `/workspace/isaaclab/pen_grasp_rl/` |
| `~/IsaacLab/source/` | `/workspace/isaaclab/source/` |

```bash
# 예: VS Code로 수정
code ~/IsaacLab/pen_grasp_rl/

# 수정 후 컨테이너에서 바로 실행 가능
```

---

## 8. TensorBoard로 학습 모니터링

### 방법 1: 컨테이너 안에서 실행 (권장)
```bash
# 새 터미널에서 컨테이너 진입
cd ~/IsaacLab
./docker/container.py enter base

# 컨테이너 내부에서
tensorboard --logdir=/workspace/isaaclab/logs/pen_grasp --bind_all
# 브라우저: http://localhost:6006
```

### 방법 2: 호스트에서 실행
```bash
# 호스트 터미널에서
pip install tensorboard
tensorboard --logdir=~/IsaacLab/logs/pen_grasp
# 브라우저: http://localhost:6006
```

> **주의**: 호스트에서 실행하려면 logs 폴더가 볼륨 마운트되어 있어야 합니다. 로그가 안 보이면 방법 1을 사용하세요.

---

## 9. 로그 파일 내보내기

### 9.1 Docker 컨테이너에서 호스트로 복사

볼륨 마운트가 안 되어 있는 경우:
```bash
# 호스트에서 실행
docker cp isaac-lab-base:/workspace/isaaclab/logs/pen_grasp ~/pen_grasp_logs
```

### 9.2 다른 컴퓨터로 전송 (SCP)

학습용 PC에서 다른 PC로 로그를 전송하려면 SCP를 사용합니다.

#### 학습용 PC (보내는 쪽) 설정
```bash
# SSH 서버 설치 및 시작
sudo apt install openssh-server
sudo systemctl start ssh

# IP 주소 확인
hostname -I
```

#### 받는 PC에서 실행
```bash
# 로그 파일 가져오기
scp -r <사용자명>@<학습PC_IP>:~/pen_grasp_logs ./

# 예시
scp -r user@192.168.50.40:~/pen_grasp_logs ./
```

#### 반대로 보내기 (학습 PC → 다른 PC)

받는 PC에서 SSH 서버가 실행 중이라면:
```bash
# 학습 PC에서 실행
scp -r ~/pen_grasp_logs <사용자명>@<받는PC_IP>:~/
```

> **참고**:
> - 처음 연결 시 "authenticity of host" 메시지가 나오면 `yes` 입력
> - 비밀번호는 대상 PC의 로그인 비밀번호 입력

---

## 10. 문제 해결

### GPU 인식 안 될 때
```bash
# Docker에서 GPU 확인
docker run --rm --gpus all nvidia/cuda:12.0-base nvidia-smi
```

### 권한 문제
```bash
# docker 그룹에 추가 안 된 경우
sudo usermod -aG docker $USER
newgrp docker
```

### 이미지 빌드 실패
```bash
# 캐시 삭제 후 재빌드
docker system prune -a
docker compose --profile base build --no-cache
```

### 컨테이너 상태 확인
```bash
docker ps -a
docker compose --profile base logs
```

---

## 11. NGC 계정 관리

### 다른 PC에서 사용
- 동일한 API Key를 여러 PC에서 사용 가능
- 각 PC에서 `docker login nvcr.io` 실행

### PC 사용 종료 시
```bash
# 해당 PC에서 로그아웃
docker logout nvcr.io
```

### API Key 분실/유출 시
1. [NGC 웹사이트](https://ngc.nvidia.com/) 접속
2. Setup → API Key → **Revoke** (폐기)
3. 새 키 발급
4. 모든 PC에서 다시 로그인

---

## 12. 요약: 새 PC 설정 체크리스트

- [ ] NVIDIA 드라이버 설치
- [ ] Docker 설치
- [ ] NVIDIA Container Toolkit 설치
- [ ] NGC 로그인 (`docker login nvcr.io`)
- [ ] Isaac Lab 클론 (`git clone https://github.com/isaac-sim/IsaacLab.git`)
- [ ] 팀 프로젝트 클론 (`git clone https://github.com/KERNEL3-2/CoWriteBotRL.git`)
- [ ] pen_grasp_rl 폴더를 IsaacLab 안으로 복사
- [ ] e0509_gripper_isaac 폴더를 홈 디렉토리로 복사
- [ ] docker-compose.yaml에 볼륨 마운트 추가 (pen_grasp_rl, logs, e0509_gripper_isaac)
- [ ] `./docker/container.py start` (빌드 + 실행)
- [ ] `./pen_grasp_rl/docker_setup.sh` (컨테이너 내부)

설정 완료 후에는 4번(학습 실행) 섹션부터 따라하면 됩니다.
