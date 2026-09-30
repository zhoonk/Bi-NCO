##########################################################################################
# Training entry point for the PFSP (v4 experiments).
#
# Examples
#   python train.py --variant M0 --seed 0                      # Bi-NCO, manuscript settings
#   python train.py --variant M1 --epochs 1500 --seed 0        # forward-only, reduced budget
#   python train.py --variant M0 --epochs 2 --no_save          # timing run (see epoch_log.csv)
#
# Variants are defined in exp_config.py. Defaults follow Section V-A2 of the paper:
# 5000 epochs x 10,000 instances, batch 200, lr 1e-4 decayed by 0.97 every 100 epochs.

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
    p.add_argument('--variant', default='M0', help='M0..M9 except M4, or SEDD (see exp_config.py)')
    p.add_argument('--seed', type=int, default=None)
    p.add_argument('--job', type=int, default=20)
    p.add_argument('--machine', type=int, default=10)
    p.add_argument('--trajectory', type=int, default=64, help='N, rollouts per direction (2N in total)')
    p.add_argument('--epochs', type=int, default=5000)
    p.add_argument('--episodes', type=int, default=10000, help='instances per epoch')
    p.add_argument('--batch', type=int, default=200)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--clip_value', type=float, default=None, help='upper bound of alpha for M8 (required for M8; choose from the alpha distribution of M0)')
    p.add_argument('--cuda', type=int, default=0, help='CUDA device index; -1 for CPU')
    p.add_argument('--save_interval', type=int, default=500)
    p.add_argument('--no_save', action='store_true', help='do not save checkpoints (timing runs)')
    p.add_argument('--val_problems', default=None, help='.pt file with validation instances (processing times)')
    p.add_argument('--val_problems_key', default=None)
    p.add_argument('--val_ref', default=None, help='.pt with reference makespans (e.g. key ub or lb)')
    p.add_argument('--val_ref_key', default=None)
    p.add_argument('--val_every', type=int, default=1)
    p.add_argument('--val_max', type=int, default=200)
    p.add_argument('--desc', default=None, help='suffix of the result folder name')
    p.add_argument('--wandb', action='store_true', help='log to Weights & Biases (needs wandb login)')
    p.add_argument('--wandb_project', default='bi-nco-pfsp')
    p.add_argument('--wandb_name', default=None, help='run name; default: the result folder description')
    p.add_argument('--wandb_group', default=None, help='runs of the same experiment, e.g. A-T1')
    p.add_argument('--wandb_tags', nargs='*', default=None)
    p.add_argument('--debug', action='store_true', help='tiny CPU run for code checks')
    return p.parse_args()


args = parse_args()
opts = resolve_variant(args.variant)

if args.debug:
    args.job, args.machine, args.trajectory, args.epochs, args.episodes, args.batch, args.cuda = 6, 3, 4, 2, 8, 4, -1
USE_CUDA = args.cuda >= 0
CUDA_DEVICE_NUM = max(args.cuda, 0)

# the result folder name is fixed when utils is imported; set it through the log description
desc = args.desc or 'train__pfsp{}x{}_{}{}'.format(args.job, args.machine, args.variant,
                                                '' if args.seed is None else '_s{}'.format(args.seed))

from PFSPTrainer import PFSPTrainer as Trainer

env_params = {
    'job_size': args.job,
    'machine_size': args.machine,
    'trajectory_size': args.trajectory,
    'direction_mode': opts['direction_mode'],
}

model_params = {
    'job_size': args.job,
    'machine_size': args.machine,
    'trajectory_size': args.trajectory,
    'direction_mode': opts['direction_mode'],
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
    'eval_type': 'argmax',
}

optimizer_params = {
    'optimizer': {
        'lr': args.lr,
        'weight_decay': 1e-6
    },
    'scheduler': {
        'milestones': list(range(100, args.epochs + 1, 100)),
        'gamma': 0.97
    }
}

trainer_params = {
    'use_cuda': USE_CUDA,
    'cuda_device_num': CUDA_DEVICE_NUM,
    'epochs': args.epochs,
    'train_episodes': args.episodes,
    'train_batch_size': args.batch,
    'seed': args.seed,
    'variant': args.variant,
    'pseudo_label': opts['pseudo_label'],
    'weighting': opts['weighting'],
    'clip_value': args.clip_value if args.clip_value is not None else opts['clip_value'],
    'loss_type': opts['loss_type'],
    'transpose_aug': opts['transpose_aug'],
    'validation': {
        'enable': args.val_problems is not None and args.val_ref is not None,
        'problems_path': args.val_problems,
        'problems_key': args.val_problems_key,
        'ref_path': args.val_ref,
        'ref_key': args.val_ref_key,
        'every': args.val_every,
        'max_instances': args.val_max,
        'batch_size': 50,
        'clamp_negative': False,  # UB or LB gap; no negative gaps occurred for PFSP
        'eval_type': 'argmax',    # greedy per latent variable, as in the test protocol
        'seed': 1234,             # same random draws at every validation and in every run
    },
    'logging': {
        'model_save_interval': (10**9 if args.no_save else args.save_interval),
        'img_save_interval': (10**9 if args.no_save else args.save_interval),
        'log_image_params_1': {
            'json_foldername': 'log_image_style',
            'filename': 'style_tsp_20.json'
        },
        'log_image_params_2': {
            'json_foldername': 'log_image_style',
            'filename': 'style_loss_1.json'
        },
    },
    'wandb': {
        'enable': args.wandb,
        'project': args.wandb_project,
        'name': args.wandb_name or desc,
        'group': args.wandb_group,
        'tags': args.wandb_tags,
    },
    'model_load': {
        'enable': False,
        'path': None,
        'epoch': None,
    }
}

logger_params = {
    'log_file': {
        'desc': desc,
        'filename': 'run_log'
    }
}


def main():
    create_logger(**logger_params)
    _print_config()

    trainer = Trainer(env_params=env_params,
                      model_params=model_params,
                      optimizer_params=optimizer_params,
                      trainer_params=trainer_params)

    copy_all_src(trainer.result_folder)

    trainer.run()


def _print_config():
    logger = logging.getLogger('root')
    logger.info('args: {}'.format(vars(args)))
    logger.info('USE_CUDA: {}, CUDA_DEVICE_NUM: {}'.format(USE_CUDA, CUDA_DEVICE_NUM))
    [logger.info(g_key + "{}".format(globals()[g_key])) for g_key in globals().keys() if g_key.endswith('params')]


if __name__ == "__main__":
    main()
