
import torch
import torch.nn as nn
import torch.nn.functional as F
# from PFSPModel_LIB import MixedScore_MultiHeadAttention
from exp_config import direction_split


class PFSPModel(nn.Module):
    """Bi-NCO model for the PFSP.

    Streams: the encoder returns two job embeddings, 'f' (preceding role, E_pre) and
    't' (succeeding role, E_suc), plus a start token (stream f) and an end token
    (stream t) that serve as the previous element at the first step.

    decoder_wiring (see exp_config.py):
      'exchange' : forward  query from f (start token), keys/values from t;
                   backward query from t (end token), keys/values from f.
                   This is the wiring of the original PFSP code (Eqs. (5)-(6)).
      'shared'   : both directions use the forward wiring; a learned direction
                   embedding is added to the query (direction-token baseline, M3).

    Each rollout has its own noise vector z (Appendix A), which is concatenated to
    the query context and to the keys/values of that rollout.
    Rollouts: the first n_fwd samples are forward, the remaining n_bwd backward.
    """

    WIRING = {
        # direction: (query stream, key stream)
        'exchange': {'fwd': ('f', 't'), 'bwd': ('t', 'f')},
        'shared':   {'fwd': ('f', 't'), 'bwd': ('f', 't')},
    }

    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        self.job_size = model_params['job_size']
        self.machine_size = model_params['machine_size']
        self.n_fwd, self.n_bwd = direction_split(model_params)
        self.wiring = model_params.get('decoder_wiring', 'exchange')
        if self.wiring not in self.WIRING:
            raise ValueError('unknown decoder_wiring {} for PFSP'.format(self.wiring))
        self.use_dir_token = (self.wiring == 'shared')
        self.cross_encoder = Cross_Encoder(**model_params)
        self.decoder = PFSP_Decoder(**model_params)
        self.dz_cont = self.model_params['dz_cont']
        self.dz_cat = self.model_params['dz_cat']
        self.encoded = {}
        self.token = {}
        self.latent = {}

    def set_z(self, batch_size, sample_size):
        dz_cat = self.model_params['dz_cat']
        dz_cont = self.model_params['dz_cont']

        latent_c_var = torch.empty(batch_size, sample_size, dz_cont).uniform_(-1, 1)

        latent_d_var = torch.zeros((batch_size, sample_size, dz_cat), dtype=torch.float32)
        one_hot_idx = torch.randint(0, dz_cat, (batch_size, sample_size), dtype=torch.long)
        latent_d_var[torch.arange(batch_size).unsqueeze(1), torch.arange(sample_size).unsqueeze(0), one_hot_idx] = 1

        latent_var = torch.cat([latent_d_var, latent_c_var], dim=-1)
        return latent_var

    def _parts(self):
        parts = []
        if self.n_fwd > 0:
            parts.append(('fwd', 0, self.n_fwd))
        if self.n_bwd > 0:
            parts.append(('bwd', self.n_fwd, self.n_fwd + self.n_bwd))
        return parts

    def pre_forward(self, reset_state):
        enc_f, enc_t, start, end = self.cross_encoder(reset_state.problems)
        # shape: (batch, job, EMBEDDING_DIM), (batch, EMBEDDING_DIM)
        self.encoded = {'f': enc_f, 't': enc_t}
        self.token = {'f': start, 't': end}
        batch_size = start.size(0)
        latent_dimension = self.dz_cont + self.dz_cat

        kv = {}
        for d, lo, hi in self._parts():
            n_d = hi - lo
            _, k_stream = self.WIRING[self.wiring][d]
            self.latent[d] = self.set_z(batch_size, n_d)
            # shape: (batch, n_d, dz)
            latent_emb = self.latent[d].reshape(batch_size * n_d, 1, latent_dimension) \
                .expand(batch_size * n_d, self.job_size, latent_dimension)
            enc_rep = self.encoded[k_stream].repeat_interleave(n_d, dim=0)
            # shape: (batch*n_d, job, embedding)
            kv[d] = torch.cat([enc_rep, latent_emb], dim=-1)
        self.decoder.set_kv(kv)

    def forward(self, state):
        batch_size = state.BATCH_IDX.size(0)
        sample_size = state.BATCH_IDX.size(1)

        probs_list = []
        for d, lo, hi in self._parts():
            n_d = hi - lo
            q_stream, _ = self.WIRING[self.wiring][d]
            if state.current_node is None:
                last = self.token[q_stream].reshape(batch_size, 1, -1).repeat(1, n_d, 1)
            else:
                last = _get_encoding(self.encoded[q_stream], state.current_node[:, lo:hi])
            # shape: (batch, n_d, embedding)
            dir_idx = (0 if d == 'fwd' else 1) if self.use_dir_token else None
            probs_list.append(self.decoder(self.encoded[q_stream], last, self.latent[d],
                                           ninf_mask=state.ninf_mask[:, lo:hi], direction=d, dir_idx=dir_idx))
        probs = torch.cat(probs_list, dim=1)
        # shape: (batch, pomo, job)

        if self.training or self.model_params['eval_type'] == 'softmax':
            while True:
                selected = probs.reshape(batch_size * sample_size, -1).multinomial(1) \
                    .squeeze(dim=1).reshape(batch_size, sample_size)
                # shape: (batch, pomo)

                prob = probs[state.BATCH_IDX, state.SAMPLE_IDX, selected] \
                    .reshape(batch_size, sample_size)
                # shape: (batch, pomo)

                if (prob != 0).all():
                    break
        else:
            selected = probs.argmax(dim=2)
            # shape: (batch, pomo)
            prob = None

        return selected, prob

def _get_encoding(encoded_nodes, node_index_to_pick):
    # encoded_nodes.shape: (batch, problem, embedding)
    # node_index_to_pick.shape: (batch, pomo)

    batch_size = node_index_to_pick.size(0)
    pomo_size = node_index_to_pick.size(1)
    embedding_dim = encoded_nodes.size(2)

    gathering_index = node_index_to_pick[:, :, None].expand(batch_size, pomo_size, embedding_dim)
    # shape: (batch, pomo, embedding)

    picked_nodes = encoded_nodes.gather(dim=1, index=gathering_index)
    # shape: (batch, pomo, embedding)

    return picked_nodes


########################################
# ENCODER
########################################
class Cross_Encoder(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        encoder_layer_num = model_params['encoder_layer_num']
        self.layers = nn.ModuleList([EncoderLayer(**model_params) for _ in range(encoder_layer_num)])
        job_size = model_params['job_size']
        machine_size = model_params['machine_size']
        embedding_dim = model_params['embedding_dim']
        self.embedding1 = nn.Linear(machine_size, embedding_dim)
        self.embedding2 = nn.Linear(machine_size, embedding_dim)
        self.start = nn.Parameter(torch.empty(1, 1, embedding_dim))
        self.start.data.uniform_(-1, 1)
        self.end = nn.Parameter(torch.empty(1, 1, embedding_dim))
        self.end.data.uniform_(-1, 1)

    def compute_normalized_matrices(self, data):

        B, N, _ = data.shape
    
        # 배치마다 min, max 계산 (dim=(1,2)로 전체 N x N에서)
        min_vals = data.view(B, -1).min(dim=1)[0].view(B, 1, 1)
        max_vals = data.view(B, -1).max(dim=1)[0].view(B, 1, 1)

        # 0으로 나눔 방지 (max == min일 경우)
        range_vals = max_vals - min_vals
        range_vals[range_vals == 0] = 1.0

        # 정규화
        scaled_data = (data - min_vals) / range_vals
        
        return scaled_data
    
    def forward(self, data):
        # col_emb.shape: (batch, col_cnt, embedding)
        # row_emb.shape: (batch, row_cnt, embedding)
        # cost_mat.shape: (batch, row_cnt, col_cnt)
        start = self.start.repeat(data.size(0), 1, 1)
        end = self.end.repeat(data.size(0), 1, 1)

        scaled_data = self.compute_normalized_matrices(data)

        out1 = self.embedding1(scaled_data.float())
        out2 = self.embedding2(scaled_data.float())

        d_out1 = torch.cat((start, out1), dim=1)
        d_out2 = torch.cat((end, out2), dim=1)

        for layer in self.layers:
            d_out1, d_out2 = layer(d_out1, d_out2)

        return d_out1[:,1:], d_out2[:,1:], d_out1[:,0], d_out2[:,0]


class EncoderLayer(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.row_encoding_block = EncodingBlock(**model_params)
        self.col_encoding_block = EncodingBlock(**model_params)

    def forward(self, row_emb, col_emb):
        # row_emb.shape: (batch, row_cnt, embedding)
        # col_emb.shape: (batch, col_cnt, embedding)
        # cost_mat.shape: (batch, row_cnt, col_cnt)
        row_emb_out = self.row_encoding_block(row_emb, col_emb)
        col_emb_out = self.col_encoding_block(col_emb, row_emb)

        return row_emb_out, col_emb_out


class EncodingBlock(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']
        head_num = self.model_params['head_num']
        qkv_dim = self.model_params['qkv_dim']

        self.Wq = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wk = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.multi_head_combine = nn.Linear(head_num * qkv_dim, embedding_dim)

        self.add_n_normalization_1 = Add_And_Normalization_Module(**model_params)
        self.feed_forward = Feed_Forward_Module(**model_params)
        self.add_n_normalization_2 = Add_And_Normalization_Module(**model_params)

    def forward(self, row_emb, col_emb):
        # NOTE: row and col can be exchanged, if cost_mat.transpose(1,2) is used
        # input1.shape: (batch, row_cnt, embedding)
        # input2.shape: (batch, col_cnt, embedding)
        # cost_mat.shape: (batch, row_cnt, col_cnt)
        head_num = self.model_params['head_num']

        q = reshape_by_heads(self.Wq(row_emb), head_num=head_num)
        # q shape: (batch, head_num, row_cnt, qkv_dim)
        k = reshape_by_heads(self.Wk(col_emb), head_num=head_num)
        v = reshape_by_heads(self.Wv(col_emb), head_num=head_num)
        # kv shape: (batch, head_num, col_cnt, qkv_dim)
        out_concat = multi_head_attention(q, k, v)
        # shape: (batch, row_cnt, head_num*qkv_dim)

        multi_head_out = self.multi_head_combine(out_concat)
        # shape: (batch, row_cnt, embedding)

        out1 = self.add_n_normalization_1(row_emb, multi_head_out)
        out2 = self.feed_forward(out1)
        out3 = self.add_n_normalization_2(out1, out2)

        return out3
        # shape: (batch, row_cnt, embedding)
########################################
# DECODER
########################################

class PFSP_Decoder(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']
        head_num = self.model_params['head_num']
        qkv_dim = self.model_params['qkv_dim']

        self.dz_cont = self.model_params['dz_cont']
        self.dz_cat = self.model_params['dz_cat']
        dz = self.dz_cont + self.dz_cat

        # self.Wq_first = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wq = nn.Linear(dz+2*embedding_dim, head_num * qkv_dim, bias=False)
        self.Wk = nn.Linear(dz+embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv = nn.Linear(dz+embedding_dim, head_num * qkv_dim, bias=False)
        self.Wp = nn.Linear(dz+embedding_dim, head_num * qkv_dim, bias=False)
        
        self.Wz = nn.Linear(self.dz_cont + self.dz_cat, head_num * qkv_dim, bias=False)
        
        self.multi_head_combine = nn.Linear(head_num * qkv_dim, embedding_dim)

        self.k = {}  # saved keys per direction, for multi-head attention
        self.v = {}  # saved values per direction
        self.single_head_key = {}  # saved per direction, for single-head attention

        # direction token, created only for the 'shared' wiring (M3)
        if model_params.get('decoder_wiring', 'exchange') == 'shared':
            self.dir_embedding = nn.Parameter(torch.zeros(2, head_num * qkv_dim))
            nn.init.normal_(self.dir_embedding, std=0.1)
        else:
            self.dir_embedding = None

        self.feed_forward = Feed_Forward_Module(**model_params)

    def set_kv(self, kv):
        # kv[direction].shape: (batch*n_d, job, embedding+dz)
        head_num = self.model_params['head_num']
        self.k, self.v, self.single_head_key = {}, {}, {}
        for d, enc in kv.items():
            self.k[d] = reshape_by_heads(self.Wk(enc), head_num=head_num)
            self.v[d] = reshape_by_heads(self.Wv(enc), head_num=head_num)
            self.single_head_key[d] = self.Wp(enc).transpose(1, 2)

    def forward(self, encoded_node, encoded_last_node, latent_vector, ninf_mask, direction, dir_idx=None):
        # encoded_node.shape: (batch, job, embedding), query stream
        # encoded_last_node.shape: (batch, n_d, embedding)
        # latent_vector.shape: (batch, n_d, dz)
        # ninf_mask.shape: (batch, n_d, job)
        k = self.k[direction]
        v = self.v[direction]
        single_head_key = self.single_head_key[direction]

        head_num = self.model_params['head_num']
        batch_size = encoded_last_node.size(0)
        trajectory_size = encoded_last_node.size(1)

        valid = (ninf_mask == 0).float()          # allowed=1
        unvisited_node = valid @ encoded_node
        cnt = valid.sum(dim=-1, keepdim=True).clamp_min(1.0)
        unvisited_node_avg = unvisited_node / cnt

        context_embedding = self.Wq(torch.cat([encoded_last_node, unvisited_node_avg, latent_vector], dim=-1))
        if dir_idx is not None:
            context_embedding = context_embedding + self.dir_embedding[dir_idx]
        reshaped_context_emb = context_embedding.reshape(batch_size*trajectory_size, 1, context_embedding.size(-1))
        reshaped_ninf_mask = ninf_mask.reshape(batch_size*trajectory_size, 1, ninf_mask.size(-1))

        q = reshape_by_heads(reshaped_context_emb, head_num=head_num)

        out_concat = multi_head_attention(q, k, v, rank3_ninf_mask=reshaped_ninf_mask)

        mh_atten_out = self.multi_head_combine(out_concat)

        updated_context = self.feed_forward(mh_atten_out)
        pointer = mh_atten_out+updated_context

        score = torch.matmul(pointer, single_head_key)

        sqrt_embedding_dim = self.model_params['sqrt_embedding_dim']
        logit_clipping = self.model_params['logit_clipping']

        score_scaled = score / sqrt_embedding_dim
        score_clipped = logit_clipping * torch.tanh(score_scaled)
        score_masked = score_clipped + reshaped_ninf_mask
        score_masked = score_masked.reshape(batch_size, trajectory_size, reshaped_ninf_mask.size(-1))

        probs = F.softmax(score_masked, dim=2)

        return probs


########################################
# NN SUB CLASS / FUNCTIONS
########################################

def reshape_by_heads(qkv, head_num):
    # q.shape: (batch, n, head_num*key_dim)   : n can be either 1 or PROBLEM_SIZE

    batch_s = qkv.size(0)
    n = qkv.size(1)

    q_reshaped = qkv.reshape(batch_s, n, head_num, -1)
    # shape: (batch, n, head_num, key_dim)

    q_transposed = q_reshaped.transpose(1, 2)
    # shape: (batch, head_num, n, key_dim)

    return q_transposed


def multi_head_attention(q, k, v, rank2_ninf_mask=None, rank3_ninf_mask=None):
    # q shape: (batch, head_num, n, key_dim)   : n can be either 1 or PROBLEM_SIZE
    # k,v shape: (batch, head_num, problem, key_dim)
    # rank2_ninf_mask.shape: (batch, problem)
    # rank3_ninf_mask.shape: (batch, group, problem)

    batch_s = q.size(0)
    head_num = q.size(1)
    n = q.size(2)
    key_dim = q.size(3)

    input_s = k.size(2)

    score = torch.matmul(q, k.transpose(2, 3))
    # shape: (batch, head_num, n, problem)

    score_scaled = score / torch.sqrt(torch.tensor(key_dim, dtype=torch.float))

    if rank2_ninf_mask is not None:
        score_scaled = score_scaled + rank2_ninf_mask[:, None, None, :].expand(batch_s, head_num, n, input_s)
    if rank3_ninf_mask is not None:
        score_scaled = score_scaled + rank3_ninf_mask[:, None, :, :].expand(batch_s, head_num, n, input_s)

    weights = nn.Softmax(dim=3)(score_scaled)
    # shape: (batch, head_num, n, problem)

    out = torch.matmul(weights, v)
    # shape: (batch, head_num, n, key_dim)

    out_transposed = out.transpose(1, 2)
    # shape: (batch, n, head_num, key_dim)

    out_concat = out_transposed.reshape(batch_s, n, head_num * key_dim)
    # shape: (batch, n, head_num*key_dim)

    return out_concat

class Add_And_Normalization_Module(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        embedding_dim = model_params['embedding_dim']
        self.norm = nn.InstanceNorm1d(embedding_dim, affine=True, track_running_stats=False)

    def forward(self, input1, input2):
        # input.shape: (batch, problem, embedding)

        added = input1 + input2
        # shape: (batch, problem, embedding)

        transposed = added.transpose(1, 2)
        # shape: (batch, embedding, problem)

        normalized = self.norm(transposed)
        # shape: (batch, embedding, problem)

        back_trans = normalized.transpose(1, 2)
        # shape: (batch, problem, embedding)

        return back_trans


class Feed_Forward_Module(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        embedding_dim = model_params['embedding_dim']
        ff_hidden_dim = model_params['ff_hidden_dim']

        self.W1 = nn.Linear(embedding_dim, ff_hidden_dim)
        self.W2 = nn.Linear(ff_hidden_dim, embedding_dim)

    def forward(self, input1):
        # input.shape: (batch, problem, embedding)

        return self.W2(F.relu(self.W1(input1)))
