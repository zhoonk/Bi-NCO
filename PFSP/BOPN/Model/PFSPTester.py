
import os
import time

import torch
from logging import getLogger

from PFSPEnv import PFSPEnv as Env
from PFSPModel import PFSPModel as Model

from utils.utils import *
from exp_utils import load_tensor, load_reference, gap_percent, CsvLog


class PFSPTester:
    """Evaluates a checkpoint on a fixed test set.

    The candidate budget is set by n_fwd and n_bwd (rollouts per direction), the
    decoding rule by eval_type ('argmax' = greedy per rollout, 'softmax' = sampling).
    Rollouts differ through their noise vectors z. Greedy decoding with a single
    candidate is n_fwd=1, n_bwd=0, eval_type='argmax'.
    aug_factor repeats each instance, which multiplies the number of candidates.
    """

    def __init__(self,
                 env_params,
                 model_params,
                 tester_params):

        # save arguments
        self.env_params = env_params
        self.model_params = model_params
        self.tester_params = tester_params

        # result folder, logger
        self.logger = getLogger(name='trainer')
        self.result_folder = get_result_folder()

        # cuda
        USE_CUDA = self.tester_params['use_cuda']
        if USE_CUDA:
            cuda_device_num = self.tester_params['cuda_device_num']
            torch.cuda.set_device(cuda_device_num)
            device = torch.device('cuda', cuda_device_num)
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            device = torch.device('cpu')
            torch.set_default_tensor_type('torch.FloatTensor')
        self.device = device

        # ENV and MODEL
        self.env = Env(**self.env_params)
        self.model = Model(**self.model_params)

        # Restore
        model_load = tester_params['model_load']
        if model_load.get('path'):
            checkpoint_fullname = '{path}/checkpoint-{epoch}.pt'.format(**model_load)
            checkpoint = torch.load(checkpoint_fullname, map_location=device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.logger.info('WARNING: no checkpoint given; evaluating an untrained model')

        # optional second model (M1+M2 ensemble: each model contributes its own rollouts)
        self.model2, self.env2 = None, None
        second = tester_params.get('model2')
        if second:
            self.env2 = Env(**second['env_params'])
            self.model2 = Model(**second['model_params'])
            ckpt2 = torch.load('{path}/checkpoint-{epoch}.pt'.format(**second['model_load']), map_location=device)
            self.model2.load_state_dict(ckpt2['model_state_dict'])

        # test data
        self.problems = load_tensor(tester_params['problems_path'], key=tester_params.get('problems_key'),
                                    device=device).float()
        self.ref = None
        if tester_params.get('ref_path'):
            self.ref = load_reference(tester_params['ref_path'], key=tester_params.get('ref_key'), device=device)
        n = self.problems.size(0) if not tester_params.get('test_episodes') else tester_params['test_episodes']
        self.problems = self.problems[:n]
        if self.ref is not None:
            self.ref = self.ref[:n]

    def run(self):
        aug_factor = self.tester_params.get('aug_factor', 1) if self.tester_params.get('augmentation_enable', False) else 1
        batch_size = self.tester_params['test_batch_size']
        costs = []
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        for lo in range(0, self.problems.size(0), batch_size):
            hi = min(lo + batch_size, self.problems.size(0))
            best = self._best_cost(self.problems[lo:hi], aug_factor)
            if self.model2 is not None:
                best = torch.minimum(best, self._best_cost(self.problems[lo:hi], aug_factor, self.model2, self.env2))
            costs.append(best)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        peak_mem = torch.cuda.max_memory_allocated() / 2**20 if torch.cuda.is_available() else float('nan')
        costs = torch.cat(costs).float()

        candidates = (self.env.n_fwd + self.env.n_bwd + (self.env2.n_fwd + self.env2.n_bwd if self.env2 else 0)) * aug_factor
        self.logger.info(' *** Test Done *** ')
        self.logger.info(' instances: {}, candidates per instance: {} (n_fwd={}, n_bwd={}, aug={}), eval_type: {}'.format(
                             costs.size(0), candidates, self.env.n_fwd, self.env.n_bwd, aug_factor,
                             self.model_params['eval_type']))
        self.logger.info(' mean cost: {:.6f}'.format(float(costs.mean())))
        self.logger.info(' total time: {:.2f}s, peak GPU memory: {:.1f} MB'.format(elapsed, peak_mem))

        gaps = None
        if self.ref is not None:
            gaps = gap_percent(costs, self.ref, clamp_negative=self.tester_params.get('clamp_negative', False))
            self.logger.info(' mean gap: {:.4f}%'.format(float(gaps.mean())))

        # per-instance results (for paired comparisons and the raw-result release)
        log = CsvLog(os.path.join(self.result_folder, 'per_instance.csv'), ['instance', 'cost', 'ref', 'gap'])
        for i in range(costs.size(0)):
            log.write(instance=i, cost=float(costs[i]),
                      ref='' if self.ref is None else float(self.ref[i]),
                      gap='' if gaps is None else float(gaps[i]))
        return costs, gaps

    def _best_cost(self, problems, aug_factor, model=None, env=None):
        model = model or self.model
        env = env or self.env
        model.eval()
        with torch.no_grad():
            env.load_problems_given(problems, aug_factor)
            reset_state, _, _ = env.reset()
            model.pre_forward(reset_state)
            state, reward, done = env.pre_step()
            while not done:
                selected, _ = model(state)
                state, reward, done, _ = env.step(selected)
        # reward.shape: (aug * batch, candidates)
        batch = problems.size(0)
        best_reward = reward.reshape(aug_factor, batch, -1).max(dim=2).values.max(dim=0).values
        return -best_reward
