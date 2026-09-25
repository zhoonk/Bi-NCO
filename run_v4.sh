#!/usr/bin/env bash
# v4 experiment plan (IEEE Access resubmission). Runs are sequential on one GPU.
# Usage:  bash run_v4.sh timing      # stage 0: measure time per epoch (about 10 minutes)
#         bash run_v4.sh atsp        # ATSP100 runs A-T1..A-T4 (bash run_v4.sh atsp A-T2: one of them)
#         bash run_v4.sh pfsp        # PFSP 20x10 runs
#         bash run_v4.sh profile     # parameter counts, FLOPs, inference time and memory
# All ATSP runs use the full 5000 epochs of the paper. REDUCED_EPOCHS applies to PFSP only;
# set it after stage 0 so that the whole plan fits the available GPU time.
set -euo pipefail

GPU=${GPU:-0}
ATSP_EPOCHS=5000                           # paper setting, for every ATSP variant
REDUCED_EPOCHS=${REDUCED_EPOCHS:-1500}   # PFSP only. TODO: decide from the stage-0 timing
SAVE_INTERVAL=${SAVE_INTERVAL:-500}
WANDB=${WANDB:-1}                          # 1: log to wandb (projects bi-nco-atsp, bi-nco-pfsp); 0: off

# data for validation curves and tests (see data/README.md); the test sets are used, with no model selection
ROOT=$(cd "$(dirname "$0")" && pwd)
ATSP_VAL=${ATSP_VAL:-$ROOT/data/ATSP/problems100.pt}
ATSP_VAL_REF=${ATSP_VAL_REF:-$ROOT/data/ATSP/lkh_result100.csv}
ATSP_VAL_KEY=${ATSP_VAL_KEY:-}            # set if problems100.pt holds a dict
ATSP_VAL_REF_KEY=${ATSP_VAL_REF_KEY:-cost_orig}   # LKH tour length column of lkh_result100.csv
PFSP_VAL=${PFSP_VAL:-$ROOT/data/PFSP/tai20x10_with_ub.pt}

ATSP_DIR=$ROOT/ATSP/BOPN/Model
PFSP_DIR=$ROOT/PFSP/BOPN/Model

# validation curves can only be recorded during training, so a missing file stops the stage
require_files() {
  for f in "$@"; do
    [ -f "$f" ] || { echo "validation file not found: $f (set ATSP_VAL, ATSP_VAL_REF or PFSP_VAL)" >&2; exit 1; }
  done
}
atsp_val_args() { echo "--val_problems $ATSP_VAL ${ATSP_VAL_KEY:+--val_problems_key $ATSP_VAL_KEY} --val_ref $ATSP_VAL_REF --val_ref_key $ATSP_VAL_REF_KEY --val_every 10"; }
# wandb options for one experiment of wiki/experiments/experiment-list.md: wandb_args <exp id> <variant> <seed>
wandb_args() { [ "$WANDB" = 1 ] && echo "--wandb --wandb_name $1_$2_s$3 --wandb_group $1 --wandb_tags $1 $2 seed$3" || true; }
pfsp_val_args() { echo "--val_problems $PFSP_VAL --val_problems_key data --val_ref $PFSP_VAL --val_ref_key ub --val_every 10"; }

case "${1:-}" in
  timing)
    # two epochs each with the full per-epoch workload; read epoch_time_s in result/*/epoch_log.csv
    (cd $ATSP_DIR && python train.py --variant M0 --epochs 2 --no_save --cuda $GPU --desc timing_atsp100_M0)
    (cd $ATSP_DIR && python train.py --variant M3 --epochs 2 --no_save --cuda $GPU --desc timing_atsp100_M3)
    (cd $PFSP_DIR && python train.py --variant M0 --epochs 2 --no_save --cuda $GPU --desc timing_pfsp20x10_M0)
    (cd $PFSP_DIR && python train.py --variant M7 --epochs 2 --no_save --cuda $GPU --desc timing_pfsp20x10_M7)
    tail -n 2 $ATSP_DIR/result/*timing_*/epoch_log.csv $PFSP_DIR/result/*timing_*/epoch_log.csv
    ;;
  atsp)
    require_files "$ATSP_VAL" "$ATSP_VAL_REF"
    case "${2:-}" in ""|A-T1|A-T2|A-T3|A-T4) ;; *) echo "unknown experiment id: $2 (A-T1..A-T4)" >&2; exit 1;; esac
    # every variant with the full budget of the paper; M0 gives Table 3 (greedy and best-of-k)
    # experiment ids follow wiki/experiments/experiment-list.md (A-T1..A-T4)
    for pair in A-T1:M0 A-T2:M1 A-T3:M3 A-T4:M4; do
      id=${pair%%:*}; v=${pair##*:}
      [ -n "${2:-}" ] && [ "$2" != "$id" ] && continue   # bash run_v4.sh atsp A-T2 runs one experiment
      (cd $ATSP_DIR && python train.py --variant $v --epochs $ATSP_EPOCHS --seed 0 --cuda $GPU \
          --save_interval $SAVE_INTERVAL $(atsp_val_args) $(wandb_args $id $v 0) --desc train__atsp100_${id}_${v}_s0)
    done
    ;;
  pfsp)
    require_files "$PFSP_VAL"
    # three seeds for M0 and M1 (reviewer asks for at least three), one seed for the rest
    for seed in 0 1 2; do
      for v in M0 M1; do
        (cd $PFSP_DIR && python train.py --variant $v --epochs $REDUCED_EPOCHS --seed $seed --cuda $GPU $(pfsp_val_args))
      done
    done
    for v in M2 M3 M5 M6 M7 M8; do
      (cd $PFSP_DIR && python train.py --variant $v --epochs $REDUCED_EPOCHS --seed 0 --cuda $GPU $(pfsp_val_args))
    done
    ;;
  profile)
    (cd $ATSP_DIR && python profile_model.py --variants M0 M1 M3 LEGACY --node_cnt 100 --batch 1 --cuda $GPU)
    (cd $PFSP_DIR && python profile_model.py --variants M0 M1 M3 --job 20 --machine 10 --batch 1 --cuda $GPU)
    ;;
  *)
    sed -n '2,8p' "$0"; exit 1;;
esac
