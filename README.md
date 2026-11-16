# Bi-NCO: Bidirectional Neural Combinatorial Optimization for Sequence Ordering Problems

Code, data, and evaluation scripts for the paper *Bi-NCO: Bidirectional Neural Combinatorial Optimization for Sequence Ordering Problems* (IEEE Access, under review). A single policy, trained once with label-free self-improvement, constructs solutions of the permutation flow shop scheduling problem (PFSP) and the asymmetric traveling salesman problem (ATSP) in both the forward and the backward direction.

The trained Bi-NCO models (seed 0) are in `checkpoints/`. The models of seeds 1 and 2, the per-instance results, and the ATSP200/500 test sets are attached to the [release](../../releases) of this version.

## Repository layout

| Path | Content |
|---|---|
| `PFSP/BOPN/Model/` | PFSP model, environment, training (`train.py`), evaluation (`test.py`), profiling (`profile_model.py`), validation-set generator (`make_val_data.py`), evaluation of the official MP-ASIL checkpoints (`eval_official_mpasil.py`) |
| `ATSP/BOPN/Model/` | ATSP model, environment, training (`train.py`), evaluation (`test.py`), profiling (`profile_model.py`) |
| `PFSP/utils`, `ATSP/utils` | logging utilities |
| `checkpoints/` | trained Bi-NCO models, seed 0: `pfsp20x10_seed0`, `pfsp20x5_seed0` (2000 epochs), `atsp100_seed0` (5000 epochs) |
| `data/` | Taillard instances with upper and lower bounds, PFSP validation sets, ATSP100 test set with LKH results (see `data/README.md`) |

## Environment

The experiments were run with Python 3.12, PyTorch 2.4.1 (CUDA 12.4), NumPy 1.26, pandas, Matplotlib, and pytz on an NVIDIA GeForce RTX 4090 GPU.

```bash
pip install -r requirements.txt
```

Weights & Biases logging is optional (`--wandb`).

## Variants

Variants are selected with `--variant` (`exp_config.py`). Names in the paper:

| `--variant` | Paper |
|---|---|
| `M0` | Bi-NCO (dual-encoder model, DESD) |
| `M1` | Uni-MDP (forward-only training) |
| `M2` | backward-only training |
| `M3` | direction-token policy without role exchange |
| `M5` | joint pseudo-label selection over both directions |
| `M6` | uniform pseudo-label weights |
| `M7` | policy-gradient training (REINFORCE with the shared POMO baseline) |
| `M8` | clipped pseudo-label weights (`--clip_value 2.24`) |
| `M9` | uncoupled encoder streams |
| `SEDD` | single-encoder dual-decoder variant |
| `SEDD1` | SEDD-Uni (SEDD trained in the forward direction only) |

## PFSP

Run from `PFSP/BOPN/Model`. Training uses 2000 epochs of 10,000 instances, batch 200 (`--accum 2` splits each batch into two micro-batches with the same gradient to fit a 24 GB GPU), and seeds 0, 1, and 2.

```bash
# training, 20x10 (Table 1, Table 4, Fig. 3); use --machine 5 and val_random20x5.pt for 20x5
python train.py --variant M0 --job 20 --machine 10 --epochs 2000 --seed 0 --accum 2 \
  --val_problems ../../../data/PFSP/val_random20x10.pt --val_problems_key data \
  --val_ref ../../../data/PFSP/val_random20x10.pt --val_ref_key lb --val_every 10

# evaluation on Taillard J x M (Table 1): best of k = 128 (64 forward + 64 backward)
python test.py --variant M0 --ckpt_dir result/<run> --ckpt_epoch 2000 \
  --problems ../../../data/PFSP/taiJxM.pt --problems_key data \
  --ref ../../../data/PFSP/taiJxM.pt --ref_key ub --job J --machine M \
  --n_fwd 64 --n_bwd 64 --batch 10 --seed 1234
# greedy (G): --n_fwd 1 --n_bwd 0
# forward-only model with 128 candidates (Table 4): --variant M1 --n_fwd 128 --n_bwd 0
# forward-only model, 64 on the original and 64 on the machine-reversed instance (Table 4):
#   --variant M1 --n_fwd 64 --n_bwd 0 --reverse_machine_pool
# ensemble of independently trained forward and backward policies (Table 4):
#   --variant M1 --ckpt_dir <M1 run> --n_fwd 64 --n_bwd 0 --ckpt2_dir <M2 run> --ckpt2_epoch 2000 --variant2 M2 --n_fwd2 0 --n_bwd2 64
```

`test.py` writes `per_instance.csv` (cost and reference per instance) in its result folder; the paired Wilcoxon tests in the paper are computed from these files.

Official MP-ASIL checkpoints (Tables 1 and 2, Appendix D): download the PFSP code and checkpoints released with the MP-ASIL paper, then

```bash
python eval_official_mpasil.py <official PFSP code dir> ../../../data/PFSP <output.tsv>
```

Model size and cost (parameters, FLOPs, inference time, memory):

```bash
python profile_model.py --variants M0 M1 SEDD SEDD1 --job 20 --machine 10 --batch 10
```

## ATSP

Run from `ATSP/BOPN/Model`. Training uses 5000 epochs on ATSP100 with seed 0.

```bash
# training (Table 3, Fig. in Appendix C)
python train.py --variant M0 --epochs 5000 --seed 0 \
  --val_problems ../../../data/ATSP/problems100.pt \
  --val_ref ../../../data/ATSP/lkh_result100.csv --val_ref_key cost_orig --val_every 10

# evaluation on 200 instances with M cities (Table 3): one rollout from every start city in each direction (k = 2M)
python test.py --variant M0 --ckpt_dir result/<run> --ckpt_epoch 5000 --node_cnt M --episodes 200 \
  --problems ../../../data/ATSP/problemsM.pt --ref ../../../data/ATSP/lkh_resultM.csv --ref_key <cost_orig for 100, Length for 200 and 500> \
  --n_fwd M --n_bwd M --start_mode all --batch 10 --seed 1234
# greedy (G): --n_fwd 1 --n_bwd 0 --start_mode fixed
# forward-only model on the cost matrix and its transpose (Section V-C1):
#   --variant M1 --n_fwd 100 --n_bwd 0 --start_mode all --transpose_pool
```

## Checkpoints and results

`checkpoints/` holds the model weights (no optimizer state) of the seed-0 Bi-NCO models; they reproduce the results of the paper, e.g. from `PFSP/BOPN/Model`:

```bash
python test.py --variant M0 --ckpt_dir ../../../checkpoints/pfsp20x10_seed0 --ckpt_epoch 2000 \
  --problems ../../../data/PFSP/tai20x10.pt --problems_key data --ref ../../../data/PFSP/tai20x10.pt --ref_key ub \
  --job 20 --machine 10 --n_fwd 64 --n_bwd 64 --batch 10 --seed 1234     # mean UB gap 0.79%
```

and from `ATSP/BOPN/Model` with `--ckpt_dir ../../../checkpoints/atsp100_seed0 --ckpt_epoch 5000` (LKH gap 1.54% on ATSP100 with `--n_fwd 100 --n_bwd 100 --start_mode all`).

The release of this version contains the seed-1 and seed-2 models of the PFSP settings (the tables report the mean over seeds 0, 1, and 2), the per-instance results (`per_instance.csv`) of the evaluations in the paper, and the ATSP200/500 test sets with their LKH results.

## Acknowledgment

The encoder blocks follow the public MatNet code (https://github.com/yd-kwon/MatNet, MIT License).

## Citation

```bibtex
@article{kim2026binco,
  title   = {Bi-NCO: Bidirectional Neural Combinatorial Optimization for Sequence Ordering Problems},
  author  = {Kim, Ji-Hoon and Kim, Jae-Gon},
  journal = {IEEE Access},
  note    = {under review},
  year    = {2026}
}
```
