##########################################################################################
# Validation and test data for the PFSP (v4 experiments).
#
#   python make_val_data.py --benchmark_dir <dir with taiJxM.npy and taiJxM_ub.npy>
#
# Writes to ../../../data/PFSP/:
#   val_random20x10.pt  200 random 20x10 instances from the training distribution
#                       (processing times uniform on {1,...,99}), drawn with a fixed seed
#                       that is independent of the training seeds, and their Taillard
#                       lower bounds: {'data': (200, 20, 10) int, 'lb': (200,) float}
#   taiJxM.pt           Taillard instances with the best-known upper bounds and the
#                       Taillard lower bounds: {'data', 'ub', 'lb'}
#
# The lower bound is that of Taillard (1993): the largest of the machine-based bounds
# (shortest head before the machine + total processing time on it + shortest tail
# after it) and the job-based bound (largest total processing time of a job).

import argparse
import os

import numpy as np
import torch

VAL_SEED = 20261001
VAL_SIZE = 200


def taillard_lb(p):
    """Taillard lower bound of the makespan. p: (batch, job, machine) processing times."""
    p = torch.as_tensor(p, dtype=torch.float64)
    cum = p.cumsum(dim=2)                                   # time of a job up to and including machine i
    head = (cum - p).min(dim=1).values                      # shortest time before machine i
    tail = (p.sum(dim=2, keepdim=True) - cum).min(dim=1).values   # shortest time after machine i
    load = p.sum(dim=1)                                      # total processing time on machine i
    machine_bound = (head + load + tail).max(dim=1).values
    job_bound = p.sum(dim=2).max(dim=1).values
    return torch.maximum(machine_bound, job_bound)


def main():
    a = argparse.ArgumentParser()
    a.add_argument('--benchmark_dir', required=True)
    a.add_argument('--out_dir', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../../data/PFSP'))
    a.add_argument('--job', type=int, default=20)
    a.add_argument('--machine', type=int, default=10)
    args = a.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    g = torch.Generator().manual_seed(VAL_SEED)
    data = torch.randint(1, 100, size=(VAL_SIZE, args.job, args.machine), generator=g)
    path = os.path.join(args.out_dir, 'val_random{}x{}.pt'.format(args.job, args.machine))
    torch.save({'data': data, 'lb': taillard_lb(data).float()}, path)
    print('wrote', path)

    for name in sorted(os.listdir(args.benchmark_dir)):
        if not (name.startswith('tai') and name.endswith('.npy')) or name.endswith('_ub.npy'):
            continue
        stem = name[:-4]
        data = torch.as_tensor(np.load(os.path.join(args.benchmark_dir, name))).long()
        ub = torch.as_tensor(np.load(os.path.join(args.benchmark_dir, stem + '_ub.npy'))).reshape(-1).float()
        lb = taillard_lb(data).float()
        if (lb > ub).any():
            raise ValueError('{}: lower bound above the upper bound'.format(stem))
        path = os.path.join(args.out_dir, stem + '.pt')
        torch.save({'data': data, 'ub': ub, 'lb': lb}, path)
        print('wrote', path)


if __name__ == '__main__':
    main()
