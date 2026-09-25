# Data for the v4 experiments

`run_v4.sh` reads the validation (and test) files from here by default.
The validation curves use the test sets; no checkpoint is selected with them
(the last epoch is always used).

## ATSP (`data/ATSP/`)

| File | Content |
|---|---|
| `problems100.pt` | ATSP100 instances, tensor of shape (1000, 100, 100): the first 1000 instances of the original `ATSP100.pth` (10000 instances), i.e. the instances covered by the LKH results. The first 200 are the test set of the paper and are used for validation (`--val_max 200`). |
| `lkh_result100.csv` | LKH results for the first 1000 instances, same order: column `cost_orig` (tour length) and `t0`..`t99` (tour). Other column names: set `ATSP_VAL_REF_KEY`. |
| `problems200.pt`, `lkh_result200.csv` | ATSP200 test set (inference only) |
| `problems500.pt`, `lkh_result500.csv` | ATSP500 test set (inference only) |

## PFSP (`data/PFSP/`)

| File | Content |
|---|---|
| `tai20x10_with_ub.pt` | Taillard 20x10 instances, dict with keys `data` (processing times) and `ub` (upper bounds) |

Other paths can be used without moving files, e.g.
`ATSP_VAL=/path/problems100.pt ATSP_VAL_REF=/path/lkh_result100.csv bash run_v4.sh atsp`.

Other instance files (`*.pt`, `*.pth`) are not tracked by git (`data/.gitignore`); `ATSP/problems100.pt` (38 MB) is the only exception.
