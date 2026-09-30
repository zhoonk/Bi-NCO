# PFSP 학습 실행 가이드 (P-T1 ~ P-T14)

리눅스 GPU 서버에서 PFSP 20×10 학습을 돌리는 순서다. 위에서부터 차례로 따라 하면 된다.

| 실험 | 변형 | seed | wandb run 이름 |
|---|---|---|---|
| P-T1 ~ P-T3 | M0 Bi-NCO | 0, 1, 2 | `P-T1_M0_s0` … |
| P-T4 ~ P-T6 | M1 forward-only | 0, 1, 2 | `P-T4_M1_s0` … |
| P-T7 | M2 backward-only | 0 | `P-T7_M2_s0` |
| P-T8 | M3 direction token | 0 | `P-T8_M3_s0` |
| P-T9 | M5 global best-of-2N | 0 | `P-T9_M5_s0` |
| P-T10 | M6 uniform weights | 0 | `P-T10_M6_s0` |
| P-T11 | M7 REINFORCE | 0 | `P-T11_M7_s0` |
| P-T13 | M9 uncoupled encoder streams | 0 | `P-T13_M9_s0` |
| P-T14 | SEDD | 0 | `P-T14_SEDD_s0` |
| (P-T12) | M8 advantage clipping | 0 | 계정 소유자가 따로 요청할 때만 (7단계) |

- 모든 run은 같은 에폭 수(`PFSP_EPOCHS`)로 학습한다. 값은 3단계의 시간 측정 뒤 계정 소유자가 정한다.
- 검증 곡선은 무작위 20×10 인스턴스 200개에서 Taillard 하한 대비 gap(LB gap)으로 기록한다. 데이터는 저장소의 `data/PFSP/`에 있다.
- 필요한 것: NVIDIA GPU, 디스크 여유 **15GB 이상**(체크포인트 1개 약 85MB, run당 `PFSP_EPOCHS/500`개), GitHub와 wandb에 연결할 인터넷.

---

## 1. 코드 받기 (새 폴더에)

같은 서버에서 ATSP 학습이 돌고 있다면 **그 폴더(`~/Bi-NCO-v4`)는 건드리지 않는다.** 그 폴더에서 `git pull`을 하면 돌고 있는 ATSP 스크립트가 망가질 수 있다. PFSP는 새 폴더에 받는다.
```bash
git clone -b exp/v4-ablation https://github.com/zhoonk/Bi-NCO.git ~/Bi-NCO-v4-pfsp
cd ~/Bi-NCO-v4-pfsp
ls data/PFSP/        # val_random20x10.pt 와 tai*.pt 가 있어야 한다
```

## 2. Python 환경과 wandb

- **ATSP를 돌린 서버와 계정이면** 같은 conda 환경을 쓰면 되고, wandb도 이미 로그인되어 있다. 이 단계를 건너뛴다.
- **새 서버면** `RUN_ATSP.md`의 2단계(환경)와 3단계(wandb 로그인)를 그대로 따른다. API 키는 계정 소유자에게 개인 메시지로 받는다.

## 3. 에폭당 시간 측정 (약 10분)

구조가 다른 세 변형(M0, M7, SEDD)을 2 에폭씩 돌려 에폭당 시간을 잰다.
```bash
cd ~/Bi-NCO-v4-pfsp
conda activate <환경 이름>
GPU=0 bash run_v4.sh timing-pfsp
```
마지막에 출력되는 세 `epoch_log.csv`의 `epoch_time_s` 값(초)을 **계정 소유자에게 알려 준다.** 계정 소유자가 에폭 수(`PFSP_EPOCHS`)를 정해 준다.

측정용 결과 폴더는 지운다: `rm -rf PFSP/BOPN/Model/result/*timing_pfsp*`

## 4. 본 학습 실행

SSH가 끊겨도 계속 돌도록 **tmux 안에서** 실행한다. `<E>`에는 계정 소유자가 정한 에폭 수를 넣는다.
```bash
tmux new -s pfsp
cd ~/Bi-NCO-v4-pfsp
conda activate <환경 이름>
PFSP_EPOCHS=<E> GPU=0 bash run_v4.sh pfsp 2>&1 | tee pfsp_train.log
```
- 위 표의 13개 run이 P-T1부터 차례로 돈다(M0·M1의 seed 3개가 먼저).
- tmux에서 빠져나오기: `Ctrl+b` 누른 뒤 `d` / 다시 들어가기: `tmux attach -t pfsp`

**GPU를 여러 장 쓰거나 나눠 돌릴 때:** 실험 ID를 나눠서 창마다 따로 실행한다. 한 창에서 전체(`bash run_v4.sh pfsp`)를 돌리면서 다른 창에서 같은 ID를 또 돌리면 **같은 실험이 두 번 돈다.**
```bash
# 예: GPU 두 장
PFSP_EPOCHS=<E> GPU=0 bash run_v4.sh pfsp P-T1 P-T2 P-T3 P-T7 P-T8 P-T9 P-T13 2>&1 | tee pfsp_gpu0.log
PFSP_EPOCHS=<E> GPU=1 bash run_v4.sh pfsp P-T4 P-T5 P-T6 P-T10 P-T11 P-T14 2>&1 | tee pfsp_gpu1.log
```

## 5. 진행 확인

- **wandb:** https://wandb.ai/jizhoonk-vms/bi-nco-pfsp. `val_gap`(LB gap), `train_loss`, `grad_var`, `alpha_*`(가중치 분포), `epoch_time_s`를 본다. 각 run의 Files 탭에 `epoch_log.csv`와 `val_costs.csv`(검증 인스턴스별 비용)가 10 에폭마다 올라간다.
- **서버에서:**
  ```bash
  tail -f ~/Bi-NCO-v4-pfsp/pfsp_train.log
  nvidia-smi
  ls ~/Bi-NCO-v4-pfsp/PFSP/BOPN/Model/result/
  ```

## 6. 중간에 멈췄을 때

- 스크립트는 하나가 실패하면 거기서 멈추고 다음 실험도 돌리지 않는다.
- 실패한 run의 결과 폴더에서 `run_log` 끝부분으로 원인을 확인한 뒤 폴더를 지우고, **멈춘 실험부터 남은 것만** 다시 돌린다. 예를 들어 P-T8에서 멈췄으면:
  ```bash
  PFSP_EPOCHS=<E> GPU=0 bash run_v4.sh pfsp P-T8 P-T9 P-T10 P-T11 P-T13 P-T14 2>&1 | tee -a pfsp_train.log
  ```
- 이어서 학습하는 기능은 없어서 실패한 run은 처음부터 다시 돈다. wandb에서는 실패한 run이 `crashed`로 표시되니 지우거나 무시한다.

## 7. M8 (P-T12) — 계정 소유자가 요청할 때만

M8은 가중치 상한(`CLIP_VALUE`)이 필요하다. 계정 소유자가 M0 학습의 가중치 분포를 보고 값을 정해 줄 때만 실행한다.
```bash
PFSP_EPOCHS=<E> CLIP_VALUE=<값> GPU=0 bash run_v4.sh pfsp P-T12 2>&1 | tee -a pfsp_train.log
```

## 8. 끝난 뒤

run마다 `PFSP/BOPN/Model/result/<날짜>_train__pfsp20x10_P-T?_M?_s?/`에 다음이 남는다.
- `checkpoint-<E>.pt`: 최종 모델(추론에 사용). 500 에폭마다 저장한 중간 체크포인트도 있다.
- `epoch_log.csv`: 곡선 원자료(가중치 분포 포함)
- `val_costs.csv`: 검증 인스턴스별 최선 비용(검증 시점마다 한 줄)
- `run_log`, `src/`: 학습 로그와 실행 당시 코드 사본

**체크포인트는 wandb에 올라가지 않는다.** Mac이나 다른 저장소로 백업한다(Mac 터미널에서 실행):
```bash
rsync -avh --progress <user>@<server>:~/Bi-NCO-v4-pfsp/PFSP/BOPN/Model/result/ ~/Bi-NCO-v4-results/PFSP/
```
백업이 끝나면 계정 소유자(또는 Claude)에게 알려 준다. 추론 실험(P-I1~P-I7)은 이 체크포인트로 진행한다.

API 키 정리는 ATSP와 PFSP 학습이 **모두** 끝난 뒤 `RUN_ATSP.md` 8단계대로 한다.
