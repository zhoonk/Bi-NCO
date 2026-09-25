# ATSP 학습 실행 가이드 (A-T1 ~ A-T4)

리눅스 GPU 서버에서 ATSP100 학습 4개를 돌리는 순서다. 위에서부터 차례로 따라 하면 된다.

| 실험 | 변형 | 에폭 | wandb run 이름 |
|---|---|---|---|
| A-T1 | M0 Bi-NCO | 5000 | `A-T1_M0_s0` |
| A-T2 | M1 forward-only | 5000 | `A-T2_M1_s0` |
| A-T3 | M3 direction token | 5000 | `A-T3_M3_s0` |
| A-T4 | M4 forward + 전치 증강 | 5000 | `A-T4_M4_s0` |

필요한 것: NVIDIA GPU 1장, 디스크 여유 **5GB 이상**(체크포인트 약 4GB), GitHub와 wandb에 연결할 인터넷.

---

## 1. 코드와 데이터 받기

코드와 데이터(ATSP100 테스트 인스턴스와 LKH 결과)가 모두 GitHub 저장소에 있다. 서버에서 실행:
```bash
git clone -b exp/v4-ablation https://github.com/zhoonk/Bi-NCO.git ~/Bi-NCO-v4
```

**데이터 확인** (서버):
```bash
cd ~/Bi-NCO-v4
md5sum data/ATSP/problems100.pt data/ATSP/lkh_result100.csv
# 954586f6f30c4043fc7e32c0ab7756be  data/ATSP/problems100.pt
# 75fbb06b4cf68cda584ce179ed43c67d  data/ATSP/lkh_result100.csv
```
두 값이 위와 다르면 파일이 깨진 것이니 저장소를 다시 받는다.

## 2. Python 환경

이미 쓰던 환경(Python 3.8 이상, PyTorch 2.1 이상, CUDA 사용 가능)이 있으면 활성화하고 wandb만 설치한다.
```bash
conda activate <환경 이름>
pip install wandb pandas matplotlib pytz
```

환경이 없으면 저장소에 있는 파일로 새로 만든다(Python 3.8, PyTorch 2.4.1 + CUDA 11.8):
```bash
conda env create -f ATSP/BOPN/Model/environment.yaml    # 환경 이름: aiopt
conda activate aiopt
pip install wandb
```
> Python 3.8이면 pip가 3.8을 지원하는 wandb(0.24.x)를 자동으로 고른다. 버전을 따로 지정할 필요는 없다.

**GPU 확인:**
```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```
`True`와 GPU 이름이 나와야 한다. `False`가 나오면 드라이버와 PyTorch의 CUDA 버전이 맞지 않는 것이다(`nvidia-smi`의 CUDA Version이 11.8 이상이어야 함).
> 매우 최신 GPU(예: RTX 50 시리즈, B200)는 CUDA 11.8용 PyTorch 2.4.1이 동작하지 않을 수 있다. 이때는 그 GPU를 지원하는 PyTorch를 설치한다(https://pytorch.org 의 설치 명령). 코드는 PyTorch 2.1 이상이면 된다.

## 3. wandb 로그인

기록은 계정 소유자(`jizhoonk-vms`)의 wandb에 남는다. 실행하는 사람은 **계정 소유자에게 API 키를 개인 메시지로 전달받아** 로그인한다.

**계정 소유자가 할 일 (키 전달)**
1. https://wandb.ai/authorize 에서 API 키를 복사한다.
2. 실행할 사람에게 **개인 메시지로만** 보낸다. 이 문서, git 저장소, 스크립트, 공개 채널에는 절대 적지 않는다.

**실행하는 사람이 할 일 (서버에서)**
```bash
wandb login
```
1. 터미널에 API 키를 입력하라는 안내가 나온다.
2. 전달받은 키를 붙여 넣고 Enter를 누른다. 입력한 키는 화면에 보이지 않는 게 정상이다.
3. `Appending key for api.wandb.ai to your netrc file`이 나오면 로그인이 끝난 것이다. 한 번만 하면 된다.
4. 받은 키가 남은 메시지나 메모는 지운다. 키는 서버의 `~/.netrc`에만 남는다.

> ⚠️ 이 키가 있으면 계정 전체를 쓸 수 있다. **학습이 모두 끝나면** 아래 8단계대로 키를 폐기한다.

> 서버가 외부 인터넷에 연결되지 않으면 로그인 없이 오프라인으로 기록할 수 있다. 4단계 실행 명령 앞에 `WANDB_MODE=offline`을 붙이고, 나중에 인터넷이 되는 곳에서 `wandb sync ATSP/BOPN/Model/result/*/wandb/offline-run-*`으로 올린다.

## 4. 본 학습 실행

SSH가 끊겨도 계속 돌도록 **tmux 안에서** 실행한다.
```bash
tmux new -s atsp
cd ~/Bi-NCO-v4
conda activate <환경 이름>
GPU=0 bash run_v4.sh atsp 2>&1 | tee atsp_train.log
```
- tmux에서 빠져나오기(학습은 계속 돔): `Ctrl+b` 누른 뒤 `d`
- 다시 들어가기: `tmux attach -t atsp`
- GPU 번호가 0이 아니면 `GPU=<번호>`로 바꾼다.

A-T1 → A-T2 → A-T3 → A-T4가 차례로 돈다. 명령 하나로 학습 4개가 모두 끝난다.

## 5. 진행 확인

- **wandb:** https://wandb.ai/jizhoonk-vms/bi-nco-atsp. `val_gap`, `train_loss`, `grad_var`, `epoch_time_s` 곡선을 본다. 원본 CSV는 각 run의 Files 탭에 10 에폭마다 올라간다.
- **서버에서:**
  ```bash
  tail -f ~/Bi-NCO-v4/atsp_train.log                       # 진행 로그 (남은 시간 표시)
  nvidia-smi                                               # GPU 사용 확인
  ls ~/Bi-NCO-v4/ATSP/BOPN/Model/result/                   # run별 결과 폴더
  ```

## 6. 중간에 멈췄을 때

- 스크립트는 하나가 실패하면 **거기서 멈추고 다음 실험도 돌리지 않는다.**
- 멈춘 실험부터 하나씩 다시 돌린다. 예를 들어 A-T3에서 멈췄으면:
  ```bash
  GPU=0 bash run_v4.sh atsp A-T3 2>&1 | tee -a atsp_train.log
  GPU=0 bash run_v4.sh atsp A-T4 2>&1 | tee -a atsp_train.log
  ```
- 현재는 이어서 학습하는 기능이 없어서, 실패한 run은 처음부터 다시 돈다. 실패한 run의 결과 폴더는 남으니 원인(`run_log` 끝부분)을 확인한 뒤 지운다.
- wandb에서는 실패한 run이 `crashed`로 표시된다. 같은 그룹(A-T3 등)에 새 run이 생기니, 실패한 run은 지우거나 무시한다.

## 7. 끝난 뒤

run마다 다음 파일이 남는다: `ATSP/BOPN/Model/result/<날짜>_train__atsp100_A-T?_M?_s0/`
- `checkpoint-5000.pt`: 최종 모델 (추론에 사용). 500 에폭마다 저장한 중간 체크포인트도 있다.
- `epoch_log.csv`: 곡선 원자료
- `run_log`: 학습 로그
- `src/`: 실행 당시 코드 사본

**체크포인트는 wandb에 올라가지 않는다.** 서버가 초기화될 수 있으니 Mac이나 다른 저장소로 백업한다(Mac 터미널에서 실행):
```bash
rsync -avh --progress <user>@<server>:~/Bi-NCO-v4/ATSP/BOPN/Model/result/ ~/Bi-NCO-v4-results/ATSP/
```
백업이 끝나면 Claude에게 알려 주면, 이 체크포인트로 추론 실험(A-I1~8)을 진행한다.

## 8. 끝난 뒤 API 키 정리 (필수)

학습 4개가 모두 끝나고 wandb에 기록이 다 올라간 것을 확인한 뒤에 한다.

**실행한 사람 (서버에서):**
```bash
wandb logout                      # 서버의 ~/.netrc에서 키를 지운다
grep -c api.wandb.ai ~/.netrc     # 0이 나오면 지워진 것이다
```

**계정 소유자 (브라우저에서):**
1. https://wandb.ai/settings 의 **API keys** 항목에서 전달했던 키를 삭제(revoke)한다.
2. 필요하면 새 키를 만든다. 계정 소유자의 다른 컴퓨터(예: 개인 Mac)가 같은 키를 쓰고 있었다면 그 컴퓨터에서 `wandb login --relogin`으로 새 키를 다시 등록한다.

키를 폐기해도 이미 올라간 run과 파일은 그대로 남는다.
