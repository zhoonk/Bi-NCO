# Bi-NCO v4 experiments (IEEE Access resubmission)

Branch `exp/v4-ablation`, based on `wip/uopn` (commit c5c072f). Only the `BOPN`
folders are used; `UOPN` and `BOPN copy` are left unchanged.

## What changed

| Area | Change |
|---|---|
| **ATSP decoder** | The decoder now follows Eqs. (5)–(6) of the paper: forward query from the preceding-role stream (`f`), keys/values from the succeeding-role stream (`t`); backward the opposite (`decoder_wiring='exchange'`). The previous code used the same stream for query and keys; it is kept as `LEGACY`. |
| **ATSP first-node query** | Each rollout now uses the embedding of its own start city. The previous code used the start city of the paired rollout of the other direction, which differs when start cities are random. `LEGACY` reproduces the previous code exactly when both halves share start cities (verified). |
| **ATSP environment** | Forward and backward rollouts are split at `n_fwd`. The previous split at `node_cnt` was correct only when the number of rollouts per direction equalled the number of cities. |
| **Direction split** | `direction_mode` = `bi` (N+N), `fwd` (2N+0), `bwd` (0+2N); the total number of rollouts per instance is 2N for every variant. |
| **PFSP** | The original code already followed Eqs. (5)–(6); `M0` reproduces it bit for bit (verified). |
| Measurement | seed, per-epoch CSV (`epoch_log.csv`: loss, score, gradient variance, gradient norm, epoch time, peak GPU memory, validation gap), checkpoint stores all options, per-instance test results (`per_instance.csv`), profiling script. |

## Variants (`exp_config.py`)

| Name | Options | Reviewer comments |
|---|---|---|
| M0 | Bi-NCO: `bi`, `exchange`, pseudo-label per direction, adaptive weights, self-improvement | reference |
| M1 | forward only | IA-R1-C4, C5; IA-R2-C3 |
| M2 | backward only (M1+M2 ensemble at test time) | IA-R1-C4; IA-R2-C2 |
| M3 | direction token, same wiring in both directions (`shared`) | IA-R2-C1, C3 |
| M4 | ATSP only: forward model trained on randomly transposed cost matrices | IA-R1-C4 |
| M5 | global best-of-2N pseudo-label | IA-R1-C2; IA-R2-C8 |
| M6 | uniform pseudo-label weights | IA-R2-C3, C8 |
| M7 | REINFORCE with the POMO shared baseline instead of self-improvement | IA-R1-C5; IA-R2-C3 |
| M8 | Standardized pseudo-label weight clipped at `clip_value` (default 2.0, `--clip_value`) | IA-R2-C8 |
| LEGACY | ATSP only: decoder wiring of the previous code | reproduction |

## Running

Step-by-step guide for the ATSP training runs on a new Linux server (environment, wandb login, tmux, recovery): `RUN_ATSP.md`.

```bash
bash run_v4.sh timing     # stage 0: time per epoch (read result/*timing*/epoch_log.csv)
bash run_v4.sh atsp       # M0, M1, M3, M4, all with the full 5000 epochs
REDUCED_EPOCHS=... bash run_v4.sh pfsp   # reduced budget, decided from stage 0
bash run_v4.sh profile    # parameters, FLOPs, inference time, memory (IA-R1-C13)
```

### Weights & Biases

`run_v4.sh` logs every training run to wandb (`WANDB=0` turns it off); run `wandb login` once on the server.
Projects: `bi-nco-atsp` and `bi-nco-pfsp`. For ATSP each experiment of the plan is one run:
name `A-T1_M0_s0`, group `A-T1`, tags `A-T1 M0 seed0` (A-T1 = M0, A-T2 = M1, A-T3 = M3, A-T4 = M4).
The run logs the columns of `epoch_log.csv` with step = epoch, and a copy of the file itself is
uploaded to the run (Files tab) every 10 epochs and at the end, so the exact raw data can be downloaded.
Without internet access: `WANDB_MODE=offline bash run_v4.sh atsp`, then `wandb sync <result>/wandb/offline-run-*`.

`bash run_v4.sh atsp A-T2` runs a single experiment (e.g. to restart it).

Single runs: `python train.py --variant M1 --epochs 1500 --seed 0 --cuda 0` in `ATSP/BOPN/Model` or `PFSP/BOPN/Model`.
Defaults follow Section V-A2: 5000 epochs × 10,000 instances, batch 200, lr 1e-4 × 0.97 every 100 epochs. Rollouts per direction keep the original settings: N = 64 for PFSP (128 per instance) and N = 100 for ATSP (200 per instance).

Testing (see the header of `test.py` for all modes):

```bash
# greedy, one candidate
python test.py --variant M0 --ckpt_dir result/<run> --ckpt_epoch 5000 --node_cnt 100 \
  --problems <test.pt> --ref <lkh.csv> --n_fwd 1 --n_bwd 0 --start_mode fixed
# best-of-200 (ATSP: 100 + 100)
python test.py ... --n_fwd 100 --n_bwd 100
# M1+M2 ensemble (100 + 100 candidates)
python test.py --variant M1 --ckpt_dir <M1> --ckpt_epoch E --n_fwd 100 --n_bwd 0 \
  --ckpt2_dir <M2> --ckpt2_epoch E --variant2 M2 --n_fwd2 0 --n_bwd2 100 ...
```

## To confirm before the full runs

1. ~~**N for ATSP.**~~ **Decided (2026-09-25): keep the original settings — N = 100 per direction for ATSP, N = 64 for PFSP.** The paper says the ATSP setup is identical to the PFSP one; that sentence must be corrected.
2. **ATSP validation data** for the convergence curves (IA-R1-C7): the paths in `run_v4.sh` are guesses; point `ATSP_VAL`, `ATSP_VAL_REF`, `PFSP_VAL` to the real files.
3. ~~**Test-time augmentation.**~~ **Decided (2026-09-25): not used.** `test.py` fixes `augmentation_enable = False` (the old `aug_factor` repeated each instance and only multiplied the number of candidates).
4. **SEDD (M0')** is implemented in `BOPN copy` and not integrated here; its parameter count for IA-R1-C13 can be profiled from that folder.
5. `PFSPModel` keeps an unused layer `Wz` for checkpoint compatibility; it is included in the parameter count.

## Checks done (CPU, torch 2.4.1)

- All variants train for two epochs on tiny instances (ATSP: 10 cities; PFSP: 6 × 3).
- LEGACY equals the previous ATSP code (identical tours and rewards with paired start cities); PFSP M0 equals the previous PFSP code.
- Backward costs equal the costs of the reversed recorded sequences, also when n_fwd ≠ node_cnt.
- The cost of a forward tour on D^T equals the cost of the reversed tour on D (premise of M4).
- With `exchange`, the forward probabilities depend on the `t`-stream keys; with `LEGACY` they do not.
- Test modes (greedy, best-of-k, sampling, transposed pool, ensemble), validation logging, and profiling run.
