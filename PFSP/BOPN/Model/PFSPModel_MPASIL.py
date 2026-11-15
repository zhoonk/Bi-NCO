"""MP-ASIL model for the PFSP, re-implemented from the paper (no official code is released).

Choi et al., "Towards Generalizable Multi-Policy Optimization with Self-Evolution for
Job Scheduling", NeurIPS 2025. The implementation follows
  - Appendix C.3: HELD-based model on MatNet; input is the |J| x |M| processing-time
    matrix; context h_c = W_c [h_{t-1}; h^u_t; h^m]  (Eq. (9)), with h_{t-1} the last
    selected job (a learnable start token at t = 1), h^u_t the mean embedding of the
    unselected jobs, and h^m all machine embeddings concatenated and projected to d_h.
  - Appendix B.1: z is concatenated to the query, key, and value inputs of the decoder
    MHA (Eq. (5)); h'_c = MHA(q, K, V) (Eq. (6)); pi = softmax(C tanh(h'_c^T K / sqrt(d_h)))
    (Eq. (7)). z is drawn once per rollout from U(-1,1)^4 x one-hot(12) (Appendix B.3).
  - Table 23: 6 encoder layers, 1 decoder layer, 8 heads, d_h = 128.
Training uses the Bi-NCO trainer with direction_mode='fwd' (all 128 rollouts forward,
best-of-128 pseudo-label, normalized-advantage weight), which is the MP-ASIL loss (Eq. (4)).

Choices not specified in the paper:
  - machine embeddings start as the one-hot vector of the machine index (machine order
    matters in the PFSP; MatNet's random one-hot is meant for identical machines);
    job embeddings start at zero, as in MatNet;
  - processing times are min-max scaled per instance, as in Bi-NCO;
  - the encoder feed-forward width is MatNet's default (512).

The encoder blocks (EncodingBlock, MixedScore_MultiHeadAttention) are taken from MatNet
(https://github.com/yd-kwon/MatNet, FFSP/FFSP_MatNet, MIT License, Copyright (c) 2021 MatNet).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from exp_config import direction_split

# Appendix C.3 / Table 23 of the MP-ASIL paper; they replace the Bi-NCO sizes
MPASIL_PARAMS = {
    'embedding_dim': 128,
    'sqrt_embedding_dim': 128**(1/2),
    'encoder_layer_num': 6,
    'head_num': 8,
    'qkv_dim': 16,
    'sqrt_qkv_dim': 16**(1/2),
    'ff_hidden_dim': 512,
    'ms_hidden_dim': 16,
    'ms_layer1_init': (1/2)**(1/2),
    'ms_layer2_init': (1/16)**(1/2),
    'logit_clipping': 10,
}


class PFSPModel_MPASIL(nn.Module):

    def __init__(self, **model_params):
        super().__init__()
        model_params = dict(model_params, **MPASIL_PARAMS)
        self.model_params = model_params
        self.job_size = model_params['job_size']
        self.machine_size = model_params['machine_size']
        self.n_fwd, self.n_bwd = direction_split(model_params)
        if self.n_bwd > 0:
            raise ValueError('MP-ASIL constructs forward only (n_bwd must be 0)')
        self.dz_cont = model_params['dz_cont']
        self.dz_cat = model_params['dz_cat']

        self.encoder = MatNet_Encoder(**model_params)
        self.decoder = MPASIL_Decoder(**model_params)

        self.encoded_jobs = None
        # shape: (batch, job, embedding)
        self.latent = None
        # shape: (batch, sample, dz)

    def set_z(self, batch_size, sample_size):
        # U(-1,1)^4 and a 12-dimensional one-hot, drawn once per rollout (Appendix B.3)
        latent_c_var = torch.empty(batch_size, sample_size, self.dz_cont).uniform_(-1, 1)
        latent_d_var = torch.zeros((batch_size, sample_size, self.dz_cat), dtype=torch.float32)
        one_hot_idx = torch.randint(0, self.dz_cat, (batch_size, sample_size), dtype=torch.long)
        latent_d_var[torch.arange(batch_size).unsqueeze(1), torch.arange(sample_size).unsqueeze(0), one_hot_idx] = 1
        return torch.cat([latent_d_var, latent_c_var], dim=-1)

    def pre_forward(self, reset_state):
        problems = reset_state.problems
        # shape: (batch, job, machine)
        batch_size = problems.size(0)
        embedding_dim = self.model_params['embedding_dim']

        row_emb = torch.zeros(size=(batch_size, self.job_size, embedding_dim))
        col_emb = torch.zeros(size=(batch_size, self.machine_size, embedding_dim))
        col_emb[:, torch.arange(self.machine_size), torch.arange(self.machine_size)] = 1
        # machine i starts as the one-hot vector e_i

        cost_mat = normalize_processing_times(problems).float()
        self.encoded_jobs, encoded_machines = self.encoder(row_emb, col_emb, cost_mat)
        # shapes: (batch, job, embedding), (batch, machine, embedding)

        self.latent = self.set_z(batch_size, self.n_fwd)
        self.decoder.set_kv(self.encoded_jobs, encoded_machines, self.latent)

    def forward(self, state):
        batch_size = state.BATCH_IDX.size(0)
        sample_size = state.BATCH_IDX.size(1)

        if state.current_node is None:
            last = None
        else:
            last = _get_encoding(self.encoded_jobs, state.current_node)
            # shape: (batch, sample, embedding)
        probs = self.decoder(self.encoded_jobs, last, self.latent, ninf_mask=state.ninf_mask)
        # shape: (batch, sample, job)

        if self.training or self.model_params['eval_type'] == 'softmax':
            while True:
                selected = probs.reshape(batch_size * sample_size, -1).multinomial(1) \
                    .squeeze(dim=1).reshape(batch_size, sample_size)
                prob = probs[state.BATCH_IDX, state.SAMPLE_IDX, selected].reshape(batch_size, sample_size)
                if (prob != 0).all():
                    break
        else:
            selected = probs.argmax(dim=2)
            prob = None

        return selected, prob


########################################
# ENCODER (MatNet)
########################################

class MatNet_Encoder(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.layers = nn.ModuleList([EncoderLayer(**model_params) for _ in range(model_params['encoder_layer_num'])])

    def forward(self, row_emb, col_emb, cost_mat):
        # row_emb.shape: (batch, job, embedding); col_emb.shape: (batch, machine, embedding)
        # cost_mat.shape: (batch, job, machine)
        for layer in self.layers:
            row_emb, col_emb = layer(row_emb, col_emb, cost_mat)
        return row_emb, col_emb


class EncoderLayer(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.row_encoding_block = EncodingBlock(**model_params)
        self.col_encoding_block = EncodingBlock(**model_params)

    def forward(self, row_emb, col_emb, cost_mat):
        row_emb_out = self.row_encoding_block(row_emb, col_emb, cost_mat)
        col_emb_out = self.col_encoding_block(col_emb, row_emb, cost_mat.transpose(1, 2))
        return row_emb_out, col_emb_out


class EncodingBlock(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = model_params['embedding_dim']
        head_num = model_params['head_num']
        qkv_dim = model_params['qkv_dim']

        self.Wq = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wk = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.mixed_score_MHA = MixedScore_MultiHeadAttention(**model_params)
        self.multi_head_combine = nn.Linear(head_num * qkv_dim, embedding_dim)

        self.add_n_normalization_1 = AddAndInstanceNormalization(**model_params)
        self.feed_forward = FeedForward(**model_params)
        self.add_n_normalization_2 = AddAndInstanceNormalization(**model_params)

    def forward(self, row_emb, col_emb, cost_mat):
        head_num = self.model_params['head_num']
        q = reshape_by_heads(self.Wq(row_emb), head_num=head_num)
        k = reshape_by_heads(self.Wk(col_emb), head_num=head_num)
        v = reshape_by_heads(self.Wv(col_emb), head_num=head_num)
        out_concat = self.mixed_score_MHA(q, k, v, cost_mat)
        multi_head_out = self.multi_head_combine(out_concat)
        out1 = self.add_n_normalization_1(row_emb, multi_head_out)
        out2 = self.feed_forward(out1)
        return self.add_n_normalization_2(out1, out2)


class MixedScore_MultiHeadAttention(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        head_num = model_params['head_num']
        ms_hidden_dim = model_params['ms_hidden_dim']
        mix1_init = model_params['ms_layer1_init']
        mix2_init = model_params['ms_layer2_init']

        self.mix1_weight = nn.Parameter(torch.distributions.Uniform(low=-mix1_init, high=mix1_init).sample((head_num, 2, ms_hidden_dim)))
        self.mix1_bias = nn.Parameter(torch.distributions.Uniform(low=-mix1_init, high=mix1_init).sample((head_num, ms_hidden_dim)))
        self.mix2_weight = nn.Parameter(torch.distributions.Uniform(low=-mix2_init, high=mix2_init).sample((head_num, ms_hidden_dim, 1)))
        self.mix2_bias = nn.Parameter(torch.distributions.Uniform(low=-mix2_init, high=mix2_init).sample((head_num, 1)))

    def forward(self, q, k, v, cost_mat):
        # q.shape: (batch, head, row, qkv); k, v.shape: (batch, head, col, qkv); cost_mat.shape: (batch, row, col)
        batch_size, head_num, row_cnt, qkv_dim = q.shape
        col_cnt = k.size(2)

        dot_product_score = torch.matmul(q, k.transpose(2, 3)) / self.model_params['sqrt_qkv_dim']
        cost_mat_score = cost_mat[:, None, :, :].expand(batch_size, head_num, row_cnt, col_cnt)
        two_scores = torch.stack((dot_product_score, cost_mat_score), dim=4).transpose(1, 2)
        # shape: (batch, row, head, col, 2)

        ms1 = F.relu(torch.matmul(two_scores, self.mix1_weight) + self.mix1_bias[None, None, :, None, :])
        ms2 = torch.matmul(ms1, self.mix2_weight) + self.mix2_bias[None, None, :, None, :]
        mixed_scores = ms2.transpose(1, 2).squeeze(4)
        # shape: (batch, head, row, col)

        weights = nn.Softmax(dim=3)(mixed_scores)
        out = torch.matmul(weights, v)
        return out.transpose(1, 2).reshape(batch_size, row_cnt, head_num * qkv_dim)


########################################
# DECODER (Eqs. (5)-(7), (9))
########################################

class MPASIL_Decoder(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = model_params['embedding_dim']
        head_num = model_params['head_num']
        qkv_dim = model_params['qkv_dim']
        machine_size = model_params['machine_size']
        dz = model_params['dz_cont'] + model_params['dz_cat']

        self.start = nn.Parameter(torch.empty(1, 1, embedding_dim).uniform_(-1, 1))   # h_{t-1} at t = 1
        self.W_m = nn.Linear(machine_size * embedding_dim, embedding_dim)            # h^m
        self.W_c = nn.Linear(3 * embedding_dim, embedding_dim, bias=False)           # Eq. (9)
        self.Wq = nn.Linear(embedding_dim + dz, head_num * qkv_dim, bias=False)      # Eq. (5)
        self.Wk = nn.Linear(embedding_dim + dz, head_num * qkv_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim + dz, head_num * qkv_dim, bias=False)
        self.multi_head_combine = nn.Linear(head_num * qkv_dim, embedding_dim)

        self.k = None   # (batch*sample, head, job, qkv)
        self.v = None
        self.single_head_key = None   # (batch*sample, embedding, job): K_t of Eq. (7)
        self.h_m = None   # (batch, embedding)

    def set_kv(self, encoded_jobs, encoded_machines, latent):
        # encoded_jobs: (batch, job, emb); encoded_machines: (batch, machine, emb); latent: (batch, sample, dz)
        batch_size, job_size, _ = encoded_jobs.shape
        sample_size = latent.size(1)
        head_num = self.model_params['head_num']

        self.h_m = self.W_m(encoded_machines.reshape(batch_size, -1))
        jobs = encoded_jobs.repeat_interleave(sample_size, dim=0)
        z = latent.reshape(batch_size * sample_size, 1, -1).expand(-1, job_size, -1)
        kz = torch.cat([jobs, z], dim=-1)
        # shape: (batch*sample, job, emb+dz)
        key = self.Wk(kz)
        self.k = reshape_by_heads(key, head_num=head_num)
        self.v = reshape_by_heads(self.Wv(kz), head_num=head_num)
        self.single_head_key = key.transpose(1, 2)

    def forward(self, encoded_jobs, encoded_last, latent, ninf_mask):
        # encoded_last: (batch, sample, emb) or None at t = 1; ninf_mask: (batch, sample, job)
        batch_size, sample_size, job_size = ninf_mask.shape
        embedding_dim = self.model_params['embedding_dim']
        head_num = self.model_params['head_num']

        if encoded_last is None:
            encoded_last = self.start.expand(batch_size, sample_size, embedding_dim)
        unselected = (ninf_mask == 0).float()
        h_u = (unselected @ encoded_jobs) / unselected.sum(dim=-1, keepdim=True).clamp_min(1.0)
        h_m = self.h_m[:, None, :].expand(batch_size, sample_size, embedding_dim)
        h_c = self.W_c(torch.cat([encoded_last, h_u, h_m], dim=-1))
        # shape: (batch, sample, emb)

        q = self.Wq(torch.cat([h_c, latent], dim=-1)).reshape(batch_size * sample_size, 1, -1)
        q = reshape_by_heads(q, head_num=head_num)
        mask = ninf_mask.reshape(batch_size * sample_size, 1, job_size)
        h_c_new = self.multi_head_combine(multi_head_attention(q, self.k, self.v, rank3_ninf_mask=mask))
        # shape: (batch*sample, 1, emb)

        score = torch.matmul(h_c_new, self.single_head_key) / self.model_params['sqrt_embedding_dim']
        score = self.model_params['logit_clipping'] * torch.tanh(score) + mask
        return F.softmax(score.reshape(batch_size, sample_size, job_size), dim=2)


########################################
# SUB MODULES
########################################

def normalize_processing_times(data):
    # min-max scaling per instance over the whole (job, machine) matrix, as in Bi-NCO
    batch_size = data.size(0)
    min_vals = data.reshape(batch_size, -1).min(dim=1)[0].view(batch_size, 1, 1)
    max_vals = data.reshape(batch_size, -1).max(dim=1)[0].view(batch_size, 1, 1)
    range_vals = max_vals - min_vals
    range_vals[range_vals == 0] = 1.0
    return (data - min_vals) / range_vals


def _get_encoding(encoded_nodes, node_index_to_pick):
    batch_size, sample_size = node_index_to_pick.shape
    gathering_index = node_index_to_pick[:, :, None].expand(batch_size, sample_size, encoded_nodes.size(2))
    return encoded_nodes.gather(dim=1, index=gathering_index)


def reshape_by_heads(qkv, head_num):
    batch_s, n = qkv.size(0), qkv.size(1)
    return qkv.reshape(batch_s, n, head_num, -1).transpose(1, 2)


def multi_head_attention(q, k, v, rank3_ninf_mask=None):
    # q: (batch, head, n, key_dim); k, v: (batch, head, problem, key_dim); mask: (batch, n, problem)
    score = torch.matmul(q, k.transpose(2, 3)) / q.size(3)**(1/2)
    if rank3_ninf_mask is not None:
        score = score + rank3_ninf_mask[:, None, :, :]
    weights = nn.Softmax(dim=3)(score)
    out = torch.matmul(weights, v)
    batch_s, head_num, n, key_dim = out.shape
    return out.transpose(1, 2).reshape(batch_s, n, head_num * key_dim)


class AddAndInstanceNormalization(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.norm = nn.InstanceNorm1d(model_params['embedding_dim'], affine=True, track_running_stats=False)

    def forward(self, input1, input2):
        return self.norm((input1 + input2).transpose(1, 2)).transpose(1, 2)


class FeedForward(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.W1 = nn.Linear(model_params['embedding_dim'], model_params['ff_hidden_dim'])
        self.W2 = nn.Linear(model_params['ff_hidden_dim'], model_params['embedding_dim'])

    def forward(self, input1):
        return self.W2(F.relu(self.W1(input1)))
