
import os
import time

import torch
from logging import getLogger

from PFSPEnv import PFSPEnv as Env
from PFSPModel import build_model as Model

from torch.optim import Adam as Optimizer
from torch.optim.lr_scheduler import MultiStepLR as Scheduler

from utils.utils import *
from exp_config import DEFAULTS
from exp_utils import set_seed, GradStats, AlphaStats, CsvLog, WandbLog, load_tensor, load_reference, gap_percent, count_parameters


EPOCH_FIELDS = ['epoch', 'lr', 'train_score', 'train_loss', 'grad_var', 'grad_norm',
                'epoch_time_s', 'peak_mem_mb', 'val_gap',
                'alpha_mean', 'alpha_p50', 'alpha_p90', 'alpha_p95', 'alpha_p99', 'alpha_max',
                'alpha_zero_frac', 'alpha_clip_frac']


class PFSPTrainer:
    def __init__(self,
                 env_params,
                 model_params,
                 optimizer_params,
                 trainer_params):

        # save arguments
        self.env_params = env_params
        self.model_params = model_params
        self.optimizer_params = optimizer_params
        self.trainer_params = trainer_params
        self.opts = {k: trainer_params.get(k, DEFAULTS[k])
                     for k in ('pseudo_label', 'weighting', 'clip_value', 'loss_type', 'transpose_aug')}
        if self.opts['transpose_aug']:
            raise ValueError('transpose_aug (M4) is defined for the ATSP only')

        # result folder, logger
        self.logger = getLogger(name='trainer')
        self.result_folder = get_result_folder()
        self.result_log = LogData()

        # seed (before the model is built, so that initialization is reproducible)
        self.seed = trainer_params.get('seed')
        if self.seed is not None:
            set_seed(self.seed)

        # cuda
        USE_CUDA = self.trainer_params['use_cuda']
        if USE_CUDA:
            cuda_device_num = self.trainer_params['cuda_device_num']
            torch.cuda.set_device(cuda_device_num)
            device = torch.device('cuda', cuda_device_num)
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            device = torch.device('cpu')
            torch.set_default_tensor_type('torch.FloatTensor')
        self.device = device

        # Main Components
        self.model = Model(**self.model_params)
        self.env = Env(**self.env_params)
        self.optimizer = Optimizer(self.model.parameters(), **self.optimizer_params['optimizer'])
        self.scheduler = Scheduler(self.optimizer, **self.optimizer_params['scheduler'])
        self.logger.info('options: {}, n_fwd={}, n_bwd={}, parameters={}'.format(
            self.opts, self.env.n_fwd, self.env.n_bwd, count_parameters(self.model)))

        # Restore
        self.start_epoch = 1
        model_load = trainer_params['model_load']
        if model_load['enable']:
            checkpoint_fullname = '{path}/checkpoint-{epoch}.pt'.format(**model_load)
            checkpoint = torch.load(checkpoint_fullname, map_location=device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.start_epoch = 1 + model_load['epoch']
            self.result_log.set_raw_data(checkpoint['result_log'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.last_epoch = model_load['epoch']-1
            self.logger.info('Saved Model Loaded !!')

        # measurement
        self.grad_stats = GradStats()
        if self.opts['weighting'] == 'clipped' and self.opts['clip_value'] is None:
            raise ValueError('M8 (clipped weights) needs --clip_value')
        self.alpha_stats = AlphaStats(self.opts['clip_value'] if self.opts['weighting'] == 'clipped' else None)
        self.epoch_log = CsvLog(os.path.join(self.result_folder, 'epoch_log.csv'), EPOCH_FIELDS)
        self.val = self._load_validation(trainer_params.get('validation'))
        self.wandb = WandbLog(trainer_params.get('wandb'),
                              config={'env_params': env_params, 'model_params': model_params,
                                      'optimizer_params': optimizer_params, 'trainer_params': trainer_params,
                                      'options': self.opts, 'result_folder': self.result_folder},
                              run_dir=self.result_folder, csv_path=self.epoch_log.path)

        # utility
        self.time_estimator = TimeEstimator()

    def _load_validation(self, cfg):
        # cfg: {'enable', 'problems_path', 'problems_key', 'ref_path', 'ref_key', 'every', 'batch_size',
        #       'clamp_negative', 'eval_type', 'seed'}
        if not cfg or not cfg.get('enable'):
            return None
        problems = load_tensor(cfg['problems_path'], key=cfg.get('problems_key'), device=self.device).float()
        ref = load_reference(cfg['ref_path'], key=cfg.get('ref_key'), device=self.device)
        n = min(problems.size(0), ref.size(0), cfg.get('max_instances', problems.size(0)))
        self.logger.info('validation: {} instances from {}'.format(n, cfg['problems_path']))
        return {'problems': problems[:n], 'ref': ref[:n], 'every': cfg.get('every', 1),
                'batch_size': cfg.get('batch_size', 100), 'clamp': cfg.get('clamp_negative', False),
                'eval_type': cfg.get('eval_type', 'argmax'), 'seed': cfg.get('seed', 1234)}

    def run(self):
        self.time_estimator.reset(self.start_epoch)
        for epoch in range(self.start_epoch, self.trainer_params['epochs']+1):
            self.logger.info('=================================================================')

            # LR Decay
            self.scheduler.step()

            # Train
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
            t0 = time.perf_counter()
            train_score, train_loss = self._train_one_epoch(epoch)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            epoch_time = time.perf_counter() - t0
            peak_mem = torch.cuda.max_memory_allocated() / 2**20 if torch.cuda.is_available() else float('nan')
            self.result_log.append('train_score', epoch, train_score)
            self.result_log.append('train_loss', epoch, train_loss)

            grad_var, grad_norm = self.grad_stats.summary()
            alpha = self.alpha_stats.summary()
            val_gap = ''
            if self.val is not None and (epoch % self.val['every'] == 0 or epoch == 1):
                val_gap, costs = self._validate()
                self._write_val_costs(epoch, costs)
                self.logger.info('Epoch {:3d}: validation gap {:.4f}%'.format(epoch, val_gap))
            self.epoch_log.write(epoch=epoch, lr=self.optimizer.param_groups[0]['lr'], train_score=train_score,
                                 train_loss=train_loss, grad_var=grad_var, grad_norm=grad_norm,
                                 epoch_time_s=epoch_time, peak_mem_mb=peak_mem, val_gap=val_gap, **alpha)
            self.wandb.log(epoch, lr=self.optimizer.param_groups[0]['lr'], train_score=train_score,
                           train_loss=train_loss, grad_var=grad_var, grad_norm=grad_norm,
                           epoch_time_s=epoch_time, peak_mem_mb=peak_mem, val_gap=val_gap, **alpha)

            ############################
            # Logs & Checkpoint
            ############################
            elapsed_time_str, remain_time_str = self.time_estimator.get_est_string(epoch, self.trainer_params['epochs'])
            self.logger.info("Epoch {:3d}/{:3d}: Time Est.: Elapsed[{}], Remain[{}], epoch {:.1f}s, grad_var {:.4e}".format(
                epoch, self.trainer_params['epochs'], elapsed_time_str, remain_time_str, epoch_time, grad_var))

            all_done = (epoch == self.trainer_params['epochs'])
            model_save_interval = self.trainer_params['logging']['model_save_interval']
            img_save_interval = self.trainer_params['logging']['img_save_interval']

            if epoch > 1:  # save latest images, every epoch
                self.logger.info("Saving log_image")
                image_prefix = '{}/latest'.format(self.result_folder)
                util_save_log_image_with_label(image_prefix, self.trainer_params['logging']['log_image_params_1'],
                                    self.result_log, labels=['train_score'])
                util_save_log_image_with_label(image_prefix, self.trainer_params['logging']['log_image_params_2'],
                                    self.result_log, labels=['train_loss'])

            if all_done or (epoch % model_save_interval) == 0:
                self.logger.info("Saving trained_model")
                checkpoint_dict = {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'result_log': self.result_log.get_raw_data(),
                    'model_params': self.model_params,
                    'env_params': self.env_params,
                    'options': self.opts,
                    'seed': self.seed,
                }
                torch.save(checkpoint_dict, '{}/checkpoint-{}.pt'.format(self.result_folder, epoch))

            if all_done or (epoch % img_save_interval) == 0:
                image_prefix = '{}/img/checkpoint-{}'.format(self.result_folder, epoch)
                util_save_log_image_with_label(image_prefix, self.trainer_params['logging']['log_image_params_1'],
                                    self.result_log, labels=['train_score'])
                util_save_log_image_with_label(image_prefix, self.trainer_params['logging']['log_image_params_2'],
                                    self.result_log, labels=['train_loss'])

            if all_done:
                self.logger.info(" *** Training Done *** ")
                self.logger.info("Now, printing log array...")
                util_print_log_array(self.logger, self.result_log)
                self.wandb.finish()

    def _train_one_epoch(self, epoch):

        score_AM = AverageMeter()
        loss_AM = AverageMeter()
        self.grad_stats.reset()
        self.alpha_stats.reset()

        train_num_episode = self.trainer_params['train_episodes']
        episode = 0
        loop_cnt = 0
        while episode < train_num_episode:

            remaining = train_num_episode - episode
            batch_size = min(self.trainer_params['train_batch_size'], remaining)

            avg_score, avg_loss = self._train_one_batch(batch_size)
            score_AM.update(avg_score, batch_size)
            loss_AM.update(avg_loss, batch_size)

            episode += batch_size

            # Log First 10 Batch, only at the first epoch
            if epoch == self.start_epoch:
                loop_cnt += 1
                if loop_cnt <= 10:
                    self.logger.info('Epoch {:3d}: Train {:3d}/{:3d}({:1.1f}%)  Score: {:.4f},  Loss: {:.4f}'
                                     .format(epoch, episode, train_num_episode, 100. * episode / train_num_episode,
                                             score_AM.avg, loss_AM.avg))

        # Log Once, for each epoch
        self.logger.info('Epoch {:3d}: Train ({:3.0f}%)  Score: {:.4f},  Loss: {:.4f}'
                         .format(epoch, 100. * episode / train_num_episode,
                                 score_AM.avg, loss_AM.avg))

        return score_AM.avg, loss_AM.avg

    def _train_one_batch(self, batch_size):

        # Prep
        ###############################################
        self.model.train()
        self.env.load_problems(batch_size)
        reset_state, _, _ = self.env.reset()
        self.model.pre_forward(reset_state)

        prob_list = torch.zeros(size=(batch_size, self.env.sample_size, 0))
        # shape: (batch, pomo, 0~problem)

        # POMO Rollout
        ###############################################
        state, reward, done = self.env.pre_step()
        while not done:
            selected, prob = self.model(state)
            # shape: (batch, pomo)
            state, reward, done, _ = self.env.step(selected)
            prob_list = torch.cat((prob_list, prob[:, :, None]), dim=2)

        # Loss
        ###############################################
        loss_mean = self._compute_loss(reward, prob_list)

        # Score
        ###############################################
        max_pomo_reward, _ = reward.max(dim=1)  # get best results from pomo
        score_mean = -max_pomo_reward.float().mean()  # negative sign to make positive value

        # Step & Return
        ###############################################
        self.model.zero_grad()
        loss_mean.backward()
        self.grad_stats.add(self.model)
        self.optimizer.step()

        return score_mean.item(), loss_mean.item()

    def _groups(self, sample_size):
        """Index ranges of the rollout groups from which pseudo-labels are selected."""
        nf = self.env.n_fwd
        if self.opts['pseudo_label'] == 'global':
            return [(0, sample_size)]                       # M5: best of all 2N rollouts
        return [g for g in ((0, nf), (nf, sample_size)) if g[1] > g[0]]  # separately per direction

    def _compute_loss(self, reward, prob_list):
        losses = []
        for lo, hi in self._groups(reward.size(1)):
            self.alpha_stats.add(self._alpha(reward[:, lo:hi]), reward[:, lo:hi])
            if self.opts['loss_type'] == 'pg':
                losses.append(self._pg_loss(reward[:, lo:hi], prob_list[:, lo:hi]))
            else:
                losses.append(self._loss_calc(reward[:, lo:hi], prob_list[:, lo:hi]))
        return torch.stack(losses).mean()   # equals 0.5 * (forward + backward) for two groups

    def _loss_calc(self, reward, prob_list):
        # adaptive self-improvement loss (Algorithm 1)
        _, argmax = reward.max(dim=1)
        batch_size = reward.size(0)

        if self.opts['weighting'] == 'uniform':
            loss_weight = torch.ones(size=(batch_size, 1))  # M6
        else:
            loss_weight = self._alpha(reward)  # [batch, 1]
            if self.opts['weighting'] == 'clipped':
                loss_weight = loss_weight.clamp(max=self.opts['clip_value'])  # M8

        probs = prob_list[torch.arange(batch_size), argmax, :]
        probs = probs[:, :-1]   # the last step has a single feasible action (probability one)
        log_probs = torch.log(probs + 1e-8)

        batch_loss = log_probs*loss_weight
        SIL_loss = -batch_loss.mean()

        return SIL_loss

    @staticmethod
    def _alpha(reward):
        # standardized weight of the best rollout: (r* - mean) / sqrt(var + eps), eps = 1e-8
        max_reward = reward.max(dim=1, keepdim=True).values  # [batch, 1]
        mean_reward = reward.mean(dim=1, keepdim=True)  # [batch, 1]
        pomo_variance = reward.var(dim=1, keepdim=True, unbiased=False)  # [batch, 1]
        return (max_reward - mean_reward) / torch.sqrt(pomo_variance + 1e-8)

    def _pg_loss(self, reward, prob_list):
        # M7: REINFORCE with the shared (mean) baseline of POMO, within each rollout group
        advantage = reward - reward.mean(dim=1, keepdim=True)
        log_prob = torch.log(prob_list + 1e-8).sum(dim=2)
        return -(advantage * log_prob).mean()

    def _validate(self):
        """Mean gap (%) of the best of the sample_size rollouts on the fixed validation set.

        The random state is forked and reseeded with a fixed value, so that every
        validation of every run uses the same random draws (start cities, latent
        variables, samples) and validation does not change the training random stream.
        """
        was_training = self.model.training
        self.model.eval()
        saved_eval_type = self.model.model_params['eval_type']
        self.model.model_params['eval_type'] = self.val['eval_type']
        gaps, costs = [], []
        devices = [torch.cuda.current_device()] if self.trainer_params['use_cuda'] else []
        with torch.no_grad(), torch.random.fork_rng(devices=devices):
            torch.manual_seed(self.val['seed'])
            problems, ref = self.val['problems'], self.val['ref']
            for lo in range(0, problems.size(0), self.val['batch_size']):
                hi = min(lo + self.val['batch_size'], problems.size(0))
                self.env.load_problems_given(problems[lo:hi])
                reset_state, _, _ = self.env.reset()
                self.model.pre_forward(reset_state)
                state, reward, done = self.env.pre_step()
                while not done:
                    selected, _ = self.model(state)
                    state, reward, done, _ = self.env.step(selected)
                best_cost = -reward.max(dim=1).values
                gaps.append(gap_percent(best_cost.float(), ref[lo:hi], clamp_negative=self.val['clamp']))
                costs.append(best_cost.float())
        self.model.model_params['eval_type'] = saved_eval_type
        if was_training:
            self.model.train()
        return float(torch.cat(gaps).mean()), torch.cat(costs).tolist()

    def _write_val_costs(self, epoch, costs):
        # best cost of every validation instance at this epoch, one row per validation, so that
        # the gap to any reference (e.g. an upper or a lower bound) can be computed afterwards
        path = os.path.join(self.result_folder, 'val_costs.csv')
        new = not os.path.exists(path)
        with open(path, 'a') as f:
            if new:
                f.write('epoch,' + ','.join('i{}'.format(k) for k in range(len(costs))) + '\n')
            f.write('{},'.format(epoch) + ','.join('{:.6g}'.format(c) for c in costs) + '\n')
