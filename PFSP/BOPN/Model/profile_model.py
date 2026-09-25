##########################################################################################
# Model size and cost profile for the PFSP variants (reviewer comment IA-R1-C13).
#
# Reports parameter count, FLOPs of one full construction (encoder + all decoding
# steps), inference time, and peak GPU memory. No trained weights are needed: the
# cost of a forward pass does not depend on the parameter values.
#
#   python profile_model.py --variants M0 M1 M3 --job 20 --machine 10 --batch 1

import argparse
import os
import sys
import time

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "..")
sys.path.insert(0, "../..")

import torch
from torch.utils.flop_counter import FlopCounterMode

from exp_config import resolve_variant
from exp_utils import count_parameters, set_seed
from PFSPEnv import PFSPEnv
from PFSPModel import PFSPModel
from PFSProblemDef import get_random_problems


def build(variant, job, machine, trajectory):
    opts = resolve_variant(variant)
    model_params = {
        'job_size': job, 'machine_size': machine, 'trajectory_size': trajectory, 'dz_cat': 12, 'dz_cont': 4,
        'direction_mode': opts['direction_mode'], 'decoder_wiring': opts['decoder_wiring'],
        'embedding_dim': 256, 'sqrt_embedding_dim': 256**(1/2), 'encoder_layer_num': 6,
        'qkv_dim': 16, 'head_num': 16, 'ms_hidden_dim': 16, 'ms_layer1_init': (1/2)**(1/2),
        'ms_layer2_init': (1/16)**(1/2), 'sqrt_qkv_dim': 16**(1/2), 'logit_clipping': 10,
        'ff_hidden_dim': 512, 'eval_type': 'argmax',
    }
    env = PFSPEnv(job_size=job, machine_size=machine, trajectory_size=trajectory, direction_mode=opts['direction_mode'])
    return PFSPModel(**model_params), env


def rollout(model, env, problems):
    env.load_problems_given(problems)
    reset_state, _, _ = env.reset()
    model.pre_forward(reset_state)
    state, reward, done = env.pre_step()
    while not done:
        selected, _ = model(state)
        state, reward, done, _ = env.step(selected)
    return reward


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--variants', nargs='+', default=['M0', 'M1', 'M3'])
    p.add_argument('--job', type=int, default=20)
    p.add_argument('--machine', type=int, default=10)
    p.add_argument('--trajectory', type=int, default=64)
    p.add_argument('--batch', type=int, default=1)
    p.add_argument('--repeat', type=int, default=5)
    p.add_argument('--cuda', type=int, default=0)
    a = p.parse_args()

    use_cuda = a.cuda >= 0 and torch.cuda.is_available()
    if use_cuda:
        torch.cuda.set_device(a.cuda)
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    set_seed(0)
    problems = get_random_problems(a.batch, a.job, a.machine).float()

    print('variant,parameters,gflops_per_instance,time_per_batch_s,peak_mem_mb')
    for v in a.variants:
        model, env = build(v, a.job, a.machine, a.trajectory)
        model.eval()
        with torch.no_grad():
            with FlopCounterMode(display=False) as fc:
                rollout(model, env, problems)
            gflops = fc.get_total_flops() / 1e9 / a.batch
            if use_cuda:
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
            t0 = time.perf_counter()
            for _ in range(a.repeat):
                rollout(model, env, problems)
            if use_cuda:
                torch.cuda.synchronize()
            dt = (time.perf_counter() - t0) / a.repeat
            mem = torch.cuda.max_memory_allocated() / 2**20 if use_cuda else float('nan')
        print('{},{},{:.3f},{:.4f},{:.1f}'.format(v, count_parameters(model), gflops, dt, mem))


if __name__ == '__main__':
    main()
