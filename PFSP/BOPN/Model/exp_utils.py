"""Measurement utilities for the v4 experiments (shared by the ATSP and PFSP code)."""

import csv
import os
import random

import numpy as np
import torch


def set_seed(seed):
    """Seed python, numpy, and torch (CPU and CUDA). Call before building the model."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


class GradStats:
    """Accumulates mini-batch gradients within an epoch.

    grad_var is the trace of the covariance of the mini-batch gradient vector,
    E||g - E[g]||^2, estimated over the mini-batches of one epoch. It is computed
    from a running sum and a running sum of squared norms, so the overhead is one
    extra vector of the size of the model parameters.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.g_sum = None
        self.sq_sum = 0.0
        self.n = 0

    def add(self, model):
        g = torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).detach().reshape(-1)
                       for p in model.parameters()])
        if self.g_sum is None:
            self.g_sum = torch.zeros_like(g)
        self.g_sum += g
        self.sq_sum += float((g * g).sum())
        self.n += 1

    def summary(self):
        if self.n == 0:
            return float('nan'), float('nan')
        mean = self.g_sum / self.n
        grad_var = self.sq_sum / self.n - float((mean * mean).sum())
        grad_norm = (self.sq_sum / self.n) ** 0.5
        return grad_var, grad_norm


class CsvLog:
    """Appends one row per call to a CSV file (header written once)."""

    def __init__(self, path, fields):
        self.path = path
        self.fields = fields
        if not os.path.exists(path):
            with open(path, 'w', newline='') as f:
                csv.writer(f).writerow(fields)

    def write(self, **row):
        with open(self.path, 'a', newline='') as f:
            csv.writer(f).writerow([row.get(k, '') for k in self.fields])


class AlphaStats:
    """Distribution of the standardized pseudo-label weight alpha within an epoch.

    alpha = (r* - mean) / sqrt(var + eps) is computed for every rollout group (one
    instance and one direction) whatever the loss, so that runs can be compared.
    The value is before any clipping. 'zero' counts groups whose rollouts all have
    the same cost (sigma = 0, where alpha = 0 and the group gives no update).
    """

    QUANTILES = (0.5, 0.9, 0.95, 0.99)

    def __init__(self, clip_value=None):
        self.clip_value = clip_value
        self.reset()

    def reset(self):
        self.values = []
        self.zero = 0

    def add(self, alpha, reward):
        self.values.append(alpha.detach().reshape(-1).float())
        self.zero += int((reward.max(dim=1).values == reward.min(dim=1).values).sum())

    def summary(self):
        if not self.values:
            return {}
        a = torch.cat(self.values)
        q = torch.quantile(a, torch.tensor(self.QUANTILES, device=a.device))
        out = {'alpha_mean': float(a.mean()), 'alpha_max': float(a.max()),
               'alpha_zero_frac': self.zero / a.numel()}
        for p, v in zip(self.QUANTILES, q.tolist()):
            out['alpha_p{}'.format(int(round(p * 100)))] = v
        if self.clip_value is not None:
            out['alpha_clip_frac'] = float((a > self.clip_value).float().mean())
        return out


class WandbLog:
    """Optional Weights & Biases logging of the per-epoch measurements.

    The run is created only when cfg['enable'] is true, so wandb is needed only then.
    A copy of epoch_log.csv is uploaded to the run's Files tab every `upload_every` epochs
    and at the end. (A 'live' save links the file and misses changes made to the link
    target, so the copy is uploaded explicitly.)
    """

    def __init__(self, cfg, config, run_dir, csv_path):
        self.run = None
        if not cfg or not cfg.get('enable'):
            return
        import wandb
        self.wandb = wandb
        self.csv_path = csv_path
        self.upload_every = cfg.get('upload_every', 10)
        self.run = wandb.init(project=cfg.get('project'), name=cfg.get('name'), group=cfg.get('group'),
                              tags=cfg.get('tags') or None, config=config, dir=run_dir)

    def upload_csv(self):
        # epoch_log.csv and, if present, val_costs.csv from the same result folder
        if self.run is None:
            return
        import shutil
        folder = os.path.dirname(self.csv_path)
        for name in (os.path.basename(self.csv_path), 'val_costs.csv'):
            src = os.path.join(folder, name)
            if not os.path.exists(src):
                continue
            dst = os.path.join(self.run.dir, name)
            shutil.copyfile(src, dst)
            self.wandb.save(dst, base_path=self.run.dir, policy='now')

    def log(self, epoch, **metrics):
        if self.run is None:
            return
        # skip empty entries (e.g. val_gap at epochs without validation)
        row = {k: v for k, v in metrics.items() if v != '' and v is not None}
        self.run.log(row, step=epoch)
        if epoch == 1 or epoch % self.upload_every == 0:
            self.upload_csv()

    def finish(self):
        if self.run is not None:
            self.upload_csv()
            self.run.finish()
            self.run = None


def load_tensor(path, key=None, device='cpu'):
    """Load a tensor from a .pt/.pth file; if the file holds a dict, return entry `key`."""
    obj = torch.load(path, map_location=device, weights_only=True)
    if isinstance(obj, dict):
        if key is None:
            raise ValueError('{} holds a dict {}; specify a key'.format(path, list(obj.keys())))
        obj = obj[key]
    return obj


def load_reference(path, key=None, device='cpu'):
    """Load per-instance reference costs from a .csv (column `key`) or .pt/.pth file."""
    if path.endswith('.csv'):
        import pandas as pd
        values = pd.read_csv(path)[key or 'Length'].to_numpy()
        return torch.as_tensor(values, dtype=torch.float32, device=device)
    return load_tensor(path, key=key, device=device).float().to(device)


def gap_percent(cost, ref, clamp_negative=True):
    """Per-instance gap (%) of `cost` to the reference `ref`, as in Eq. (gap) of the paper."""
    num = cost - ref
    if clamp_negative:
        num = num.clamp(min=0)
    return 100.0 * num / ref
