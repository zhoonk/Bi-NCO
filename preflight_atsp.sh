#!/usr/bin/env bash
# Preflight check before the ATSP training runs (RUN_ATSP.md, step 4). About 10 minutes on a GPU.
#   1) each variant (M0, M1, M3, M4): one short training epoch with validation, a checkpoint
#      save, and a reload of that checkpoint by test.py
#   2) one full-size epoch of M0 to measure the time per epoch
# Runs are logged to the wandb project bi-nco-atsp-test (WANDB=0: off). The results are deleted at the end.
#   GPU=0 bash preflight_atsp.sh
set -euo pipefail

GPU=${GPU:-0}
WANDB=${WANDB:-1}
QUICK_EPISODES=${QUICK_EPISODES:-400}      # instances in the short epoch
QUICK_BATCH=${QUICK_BATCH:-200}
TIMING_EPISODES=${TIMING_EPISODES:-10000}  # instances in the timing epoch (the real per-epoch workload)
ROOT=$(cd "$(dirname "$0")" && pwd)
DATA=$ROOT/data/ATSP
cd "$ROOT/ATSP/BOPN/Model"
LOG=$ROOT/preflight_atsp.log   # full output of every command; its tail is printed on failure
: > "$LOG"
run() { "$@" >> "$LOG" 2>&1 || { echo "FAIL: $*" >&2; echo "--- last lines of $LOG ---" >&2; tail -n 30 "$LOG" >&2; exit 1; }; }
last_field() { python -c "import csv,sys; print(list(csv.reader(open(sys.argv[1])))[-1][int(sys.argv[2])])" "$1" "$2"; }

for f in "$DATA/problems100.pt" "$DATA/lkh_result100.csv"; do
  [ -f "$f" ] || { echo "FAIL: data file not found: $f" >&2; exit 1; }
done
python -c "import torch, pandas, matplotlib, pytz" || { echo "FAIL: missing Python packages (see RUN_ATSP.md step 2)" >&2; exit 1; }
if [ "$GPU" -ge 0 ]; then
  python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'; print('GPU:', torch.cuda.get_device_name($GPU))"
fi
if [ "$WANDB" = 1 ]; then
  python -c "import wandb" || { echo "FAIL: wandb is not installed (pip install wandb)" >&2; exit 1; }
fi

VAL="--val_problems $DATA/problems100.pt --val_ref $DATA/lkh_result100.csv --val_ref_key cost_orig --val_max 20"
wandb_args() { [ "$WANDB" = 1 ] && echo "--wandb --wandb_project bi-nco-atsp-test --wandb_group preflight --wandb_name preflight_$1" || true; }

for v in M0 M1 M3 M4; do
  echo "=== $v: train one short epoch, save, reload ==="
  run python train.py --variant $v --epochs 1 --seed 0 --cuda $GPU --episodes $QUICK_EPISODES --batch $QUICK_BATCH \
      --save_interval 1 $VAL $(wandb_args $v) --desc preflight_$v
  rdir=$(ls -td result/*preflight_$v/ | head -1)
  [ -f "$rdir/checkpoint-1.pt" ] || { echo "FAIL: $v checkpoint not saved" >&2; exit 1; }
  case $v in M1|M4) split="--n_fwd 200 --n_bwd 0";; *) split="--n_fwd 100 --n_bwd 100";; esac
  run python test.py --variant $v --ckpt_dir "$rdir" --ckpt_epoch 1 --problems $DATA/problems100.pt \
      --ref $DATA/lkh_result100.csv --ref_key cost_orig --node_cnt 100 $split --episodes 20 --batch 20 \
      --cuda $GPU --desc preflight_test_$v
  echo "OK  $v: val_gap $(python -c "print(round(float('$(last_field "$rdir/epoch_log.csv" 8)'), 1))")% (untrained model: about 200% is normal)"
done

echo "=== M0: one full-size epoch ($TIMING_EPISODES instances) for timing ==="
run python train.py --variant M0 --epochs 1 --seed 0 --cuda $GPU --episodes $TIMING_EPISODES --no_save --desc preflight_timing
t=$(last_field "$(ls -td result/*preflight_timing/ | head -1)/epoch_log.csv" 6)
python -c "t=$t; print('epoch time: %.1f s -> one 5000-epoch run: %.1f days, all four runs: %.1f days' % (t, t*5000/86400, 4*t*5000/86400))"

rm -rf result/*preflight*
echo "=== preflight passed: start the training with  GPU=$GPU bash run_v4.sh atsp ==="
