#!/usr/bin/env bash
# v4 experiment plan (IEEE Access resubmission). Runs are sequential on one GPU.
# Usage:  bash run_v4.sh timing      # stage 0: measure time per epoch (about 10 minutes)
#         bash run_v4.sh atsp        # ATSP100 runs A-T1..A-T4 (bash run_v4.sh atsp A-T2: one of them)
#         bash run_v4.sh pfsp        # PFSP 20x10 runs P-T1..P-T14 (bash run_v4.sh pfsp P-T8 P-T9: only those)
#         bash run_v4.sh timing-pfsp # PFSP only: time per epoch of the structurally different variants
#         bash run_v4.sh profile     # parameter counts, FLOPs, inference time and memory
# All ATSP runs use the full 5000 epochs of the paper. PFSP_EPOCHS must be set for the PFSP
# runs (decided from timing-pfsp); every PFSP run uses the same value.
set -euo pipefail

GPU=${GPU:-0}
ATSP_EPOCHS=5000                           # paper setting, for every ATSP variant
PFSP_EPOCHS=${PFSP_EPOCHS:-}               # required for the pfsp stage, e.g. PFSP_EPOCHS=5000
CLIP_VALUE=${CLIP_VALUE:-}                 # required for P-T12 (M8), from the alpha distribution of M0
SAVE_INTERVAL=${SAVE_INTERVAL:-500}
WANDB=${WANDB:-1}                          # 1: log to wandb (projects bi-nco-atsp, bi-nco-pfsp); 0: off

# data for validation curves and tests (see data/README.md); the test sets are used, with no model selection
ROOT=$(cd "$(dirname "$0")" && pwd)
ATSP_VAL=${ATSP_VAL:-$ROOT/data/ATSP/problems100.pt}
ATSP_VAL_REF=${ATSP_VAL_REF:-$ROOT/data/ATSP/lkh_result100.csv}
ATSP_VAL_KEY=${ATSP_VAL_KEY:-}            # set if problems100.pt holds a dict
ATSP_VAL_REF_KEY=${ATSP_VAL_REF_KEY:-cost_orig}   # LKH tour length column of lkh_result100.csv
PFSP_VAL=${PFSP_VAL:-$ROOT/data/PFSP/val_random20x10.pt}   # 200 random 20x10 instances, Taillard lower bounds
PFSP_VAL_REF_KEY=${PFSP_VAL_REF_KEY:-lb}   # LB gap, as in Fig. 3 of the paper

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
pfsp_val_args() { echo "--val_problems $PFSP_VAL --val_problems_key data --val_ref $PFSP_VAL --val_ref_key $PFSP_VAL_REF_KEY --val_every 10"; }
pfsp_wandb_args() { [ "$WANDB" = 1 ] && echo "--wandb --wandb_project bi-nco-pfsp --wandb_name $1_$2_s$3 --wandb_group $1 --wandb_tags $1 $2 seed$3" || true; }
# experiment id:variant:seed, following wiki/experiments/experiment-list.md
PFSP_PLAN="P-T1:M0:0 P-T2:M0:1 P-T3:M0:2 P-T4:M1:0 P-T5:M1:1 P-T6:M1:2 P-T7:M2:0 P-T8:M3:0 P-T9:M5:0 P-T10:M6:0 P-T11:M7:0 P-T13:M9:0 P-T14:SEDD:0"

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
  timing-pfsp)
    # two epochs each with the full per-epoch workload; read epoch_time_s in result/*/epoch_log.csv
    for v in M0 M7 SEDD; do
      (cd $PFSP_DIR && python train.py --variant $v --epochs 2 --no_save --cuda $GPU --desc timing_pfsp20x10_$v)
    done
    tail -n 2 $PFSP_DIR/result/*timing_pfsp*/epoch_log.csv
    ;;
  pfsp)
    require_files "$PFSP_VAL"
    [ -n "$PFSP_EPOCHS" ] || { echo "set PFSP_EPOCHS (the same value for every PFSP run)" >&2; exit 1; }
    plan=$PFSP_PLAN
    if [ $# -ge 2 ]; then
      # bash run_v4.sh pfsp P-T7 P-T8 ...: only the named experiments, in the given order
      plan=""
      for want in "${@:2}"; do
        found=""
        for item in $PFSP_PLAN P-T12:M8:0; do [ "${item%%:*}" = "$want" ] && found=$item; done
        [ -n "$found" ] || { echo "unknown experiment id: $want (P-T1..P-T14)" >&2; exit 1; }
        plan="$plan $found"
      done
    fi
    # M0 and M1 with three seeds first (Fig. 3), then one seed for each other variant.
    # P-T12 (M8) runs only when named: bash run_v4.sh pfsp P-T12 with CLIP_VALUE set.
    for item in $plan; do
      id=${item%%:*}; rest=${item#*:}; v=${rest%%:*}; seed=${rest##*:}
      extra=""
      if [ "$v" = M8 ]; then
        [ -n "$CLIP_VALUE" ] || { echo "set CLIP_VALUE for P-T12 (M8)" >&2; exit 1; }
        extra="--clip_value $CLIP_VALUE"
      fi
      (cd $PFSP_DIR && python train.py --variant $v --epochs $PFSP_EPOCHS --seed $seed --cuda $GPU \
          --save_interval $SAVE_INTERVAL $extra $(pfsp_val_args) $(pfsp_wandb_args $id $v $seed) --desc train__pfsp20x10_${id}_${v}_s${seed})
    done
    ;;
  profile)
    (cd $ATSP_DIR && python profile_model.py --variants M0 M1 M3 LEGACY --node_cnt 100 --batch 1 --cuda $GPU)
    (cd $PFSP_DIR && python profile_model.py --variants M0 M1 M3 M9 SEDD --job 20 --machine 10 --batch 1 --cuda $GPU)
    ;;
  *)
    sed -n '2,8p' "$0"; exit 1;;
esac
