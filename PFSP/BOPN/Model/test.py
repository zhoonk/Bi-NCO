##########################################################################################
# Test entry point for the PFSP (v4 experiments).
#
# The candidate budget and decoding rule are explicit, so that greedy and best-of-k
# results can be reported separately (reviewer comment IA-R1-C9):
#   greedy (one candidate): --n_fwd 1 --n_bwd 0 --eval_type argmax
#   best-of-128 (Bi-NCO):   --n_fwd 64 --n_bwd 64 --eval_type argmax
#   best-of-128 (fwd-only): --n_fwd 128 --n_bwd 0
#   M1+M2 ensemble:         --variant M1 --n_fwd 64 --n_bwd 0 --ckpt2_dir <M2> --ckpt2_epoch E --variant2 M2 --n_fwd2 0 --n_bwd2 64
# Taillard files: --problems tai20x10_with_ub.pt --problems_key data --ref <same> --ref_key ub

import argparse
import logging
import os
import sys

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "..")  # for problem_def
sys.path.insert(0, "../..")  # for utils

from utils.utils import create_logger, copy_all_src
from exp_config import resolve_variant


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--variant', default='M0', help='architecture options of the trained model')
    p.add_argument('--ckpt_dir', required=True)
    p.add_argument('--ckpt_epoch', type=int, required=True)
    p.add_argument('--problems', required=True, help='.pt file with test instances (processing times)')
    p.add_argument('--problems_key', default=None)
    p.add_argument('--ref', default=None, help='.pt with reference makespans (key ub = best-known)')
    p.add_argument('--ref_key', default=None)
    p.add_argument('--job', type=int, required=True)
    p.add_argument('--machine', type=int, required=True)
    p.add_argument('--n_fwd', type=int, default=64)
    p.add_argument('--n_bwd', type=int, default=64)
    p.add_argument('--eval_type', default='argmax', choices=['argmax', 'softmax'])
    p.add_argument('--episodes', type=int, default=None)
    p.add_argument('--ckpt2_dir', default=None, help='second checkpoint for an ensemble (e.g. M2 with M1)')
    p.add_argument('--ckpt2_epoch', type=int, default=None)
    p.add_argument('--variant2', default='M2')
    p.add_argument('--n_fwd2', type=int, default=0)
    p.add_argument('--n_bwd2', type=int, default=64)
    p.add_argument('--batch', type=int, default=50)
    p.add_argument('--seed', type=int, default=1234, help='seed for noise vectors z and sampling')
    p.add_argument('--cuda', type=int, default=0)
    p.add_argument('--desc', default=None)
    return p.parse_args()


args = parse_args()
opts = resolve_variant(args.variant)
USE_CUDA = args.cuda >= 0
CUDA_DEVICE_NUM = max(args.cuda, 0)

from PFSPTester import PFSPTester as Tester
from exp_utils import set_seed

env_params = {
    'job_size': args.job,
    'machine_size': args.machine,
    'trajectory_size': max(args.n_fwd, args.n_bwd, 1),
    'n_fwd': args.n_fwd,
    'n_bwd': args.n_bwd,
}

model_params = {
    'job_size': args.job,
    'machine_size': args.machine,
    'trajectory_size': max(args.n_fwd, args.n_bwd, 1),
    'n_fwd': args.n_fwd,
    'n_bwd': args.n_bwd,
    'decoder_wiring': opts['decoder_wiring'],
    'encoder_coupling': opts['encoder_coupling'],
    'architecture': opts['architecture'],
    'dz_cat': 12,
    'dz_cont': 4,
    'embedding_dim': 256,
    'sqrt_embedding_dim': 256**(1/2),
    'encoder_layer_num': 6,
    'qkv_dim': 16,
    'head_num': 16,
    'ms_hidden_dim': 16,
    'ms_layer1_init': (1/2)**(1/2),
    'ms_layer2_init': (1/16)**(1/2),
    'sqrt_qkv_dim': 16**(1/2),
    'logit_clipping': 10,
    'ff_hidden_dim': 512,
    'eval_type': args.eval_type,
}

tester_params = {
    'use_cuda': USE_CUDA,
    'cuda_device_num': CUDA_DEVICE_NUM,
    'model_load': {'path': args.ckpt_dir, 'epoch': args.ckpt_epoch},
    'problems_path': args.problems,
    'problems_key': args.problems_key,
    'ref_path': args.ref,
    'ref_key': args.ref_key,
    'test_episodes': args.episodes,
    'test_batch_size': args.batch,
    'augmentation_enable': False,   # test-time augmentation is not used (decision 2026-09-25)
    'aug_factor': 1,
    'clamp_negative': False,
}

if args.ckpt2_dir:
    opts2 = resolve_variant(args.variant2)
    env2 = dict(env_params, n_fwd=args.n_fwd2, n_bwd=args.n_bwd2, trajectory_size=max(args.n_fwd2, args.n_bwd2, 1))
    model2 = dict(model_params, n_fwd=args.n_fwd2, n_bwd=args.n_bwd2, trajectory_size=max(args.n_fwd2, args.n_bwd2, 1),
                  decoder_wiring=opts2['decoder_wiring'], encoder_coupling=opts2['encoder_coupling'],
                  architecture=opts2['architecture'])
    tester_params['model2'] = {'env_params': env2, 'model_params': model2,
                               'model_load': {'path': args.ckpt2_dir, 'epoch': args.ckpt2_epoch}}

desc = args.desc or 'test__pfsp{}x{}_{}_f{}b{}_{}{}'.format(
    args.job, args.machine, args.variant, args.n_fwd, args.n_bwd, args.eval_type,
    '')
logger_params = {'log_file': {'desc': desc, 'filename': 'log.txt'}}


def main():
    create_logger(**logger_params)
    logger = logging.getLogger('root')
    logger.info('args: {}'.format(vars(args)))
    [logger.info(g_key + "{}".format(globals()[g_key])) for g_key in globals().keys() if g_key.endswith('params')]
    set_seed(args.seed)

    tester = Tester(env_params=env_params, model_params=model_params, tester_params=tester_params)
    copy_all_src(tester.result_folder)
    tester.run()


if __name__ == "__main__":
    main()
