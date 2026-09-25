

import torch
import torch.nn as nn
import torch.nn.functional as F
from ATSPModel_LIB import MixedScore_MultiHeadAttention
from exp_config import direction_split


class BOPN_Model(nn.Module):
    """Bi-NCO model for the ATSP.

    Streams: the encoder returns two embeddings per city. The 'f' (row) stream is
    computed from the cost matrix D and represents a city in its preceding role
    (E_pre); the 't' (column) stream is computed from D^T and represents a city in
    its succeeding role (E_suc).

    decoder_wiring (see exp_config.py):
      'exchange' : forward  query E_pre, keys/values E_suc; backward query E_suc,
                   keys/values E_pre (Eqs. (5)-(6) of the paper).
      'shared'   : both directions use the forward wiring; a learned direction
                   embedding is added to the query (direction-token baseline, M3).
      'legacy'   : query and keys come from the same stream (forward f/f,
                   backward t/t), as in the original ATSP code.

    Rollouts: the first n_fwd samples are forward, the remaining n_bwd backward.
    """

    WIRING = {
        # direction: (query stream, key stream, first-node stream)
        'exchange': {'fwd': ('f', 't', 't'), 'bwd': ('t', 'f', 'f')},
        'shared':   {'fwd': ('f', 't', 't'), 'bwd': ('f', 't', 't')},
        'legacy':   {'fwd': ('f', 'f', 't'), 'bwd': ('t', 't', 'f')},
    }

    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        self.node_cnt = model_params['node_cnt']
        self.n_fwd, self.n_bwd = direction_split(model_params)
        self.wiring = model_params.get('decoder_wiring', 'exchange')
        if self.wiring not in self.WIRING:
            raise ValueError('unknown decoder_wiring {}'.format(self.wiring))
        self.use_dir_token = (self.wiring == 'shared')
        self.start_mode = model_params.get('start_mode', 'random')  # 'random' | 'fixed'
        self.cross_encoder = Cross_Encoder(**model_params)
        self.decoder = Decoder(**model_params)
        self.encoded = {}

    def pre_forward(self, reset_state):
        encoded_f, encoded_t = self.cross_encoder(reset_state.problems)
        # shape: (batch, problem, EMBEDDING_DIM)
        self.encoded = {'f': encoded_f, 't': encoded_t}
        self.decoder.set_kv(encoded_f, encoded_t)

    def _parts(self):
        parts = []
        if self.n_fwd > 0:
            parts.append(('fwd', 0, self.n_fwd))
        if self.n_bwd > 0:
            parts.append(('bwd', self.n_fwd, self.n_fwd + self.n_bwd))
        return parts

    def forward(self, state):
        batch_size = state.BATCH_IDX.size(0)
        sample_size = state.BATCH_IDX.size(1)

        if state.current_node is None:
            if self.start_mode == 'fixed':
                selected = torch.zeros(size=(batch_size, sample_size), dtype=torch.long)
            else:
                selected = torch.randint(low=0, high=self.node_cnt, size=(batch_size, sample_size), dtype=torch.long)
            prob = torch.ones(size=(batch_size, sample_size))

            first = {}
            for d, lo, hi in self._parts():
                stream = self.WIRING[self.wiring][d][2]
                first[d] = _get_encoding(self.encoded[stream], selected[:, lo:hi])
                # shape: (batch, n_d, embedding)
            self.decoder.set_q1(first.get('fwd'), first.get('bwd'))

        else:
            probs_list = []
            for d, lo, hi in self._parts():
                q_stream, k_stream, _ = self.WIRING[self.wiring][d]
                encoded_last_node = _get_encoding(self.encoded[q_stream], state.current_node[:, lo:hi])
                # shape: (batch, n_d, embedding)
                dir_idx = (0 if d == 'fwd' else 1) if self.use_dir_token else None
                probs_list.append(self.decoder(encoded_last_node, ninf_mask=state.ninf_mask[:, lo:hi],
                                               key_stream=k_stream, direction=d, dir_idx=dir_idx))
            probs = torch.cat(probs_list, dim=1)
            # shape: (batch, pomo, problem)

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
        embedding_dim = model_params['embedding_dim']

        self.node_idx_projection = nn.Linear(1, embedding_dim)
        self.edge_mtrx_projection = nn.Linear(1, embedding_dim)

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

        batch_size, num_nodes, _ = data.shape

        input_emb = self.node_idx_projection(torch.rand((batch_size, num_nodes, 1)))

        out1, out2 = input_emb, input_emb
        
        scaled_data = self.compute_normalized_matrices(data)
        
        edge_emb = self.edge_mtrx_projection(scaled_data.float().unsqueeze(-1))

        for layer in self.layers:
            out1, out2 = layer(out1, out2, edge_emb)

        return out1, out2


class EncoderLayer(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.row_encoding_block = EncodingBlock(**model_params)
        self.col_encoding_block = EncodingBlock(**model_params)

    def forward(self, row_emb, col_emb, edge_emb):
        # row_emb.shape: (batch, row_cnt, embedding)
        # col_emb.shape: (batch, col_cnt, embedding)
        # cost_mat.shape: (batch, row_cnt, col_cnt)
        row_emb_out = self.row_encoding_block(row_emb, col_emb, edge_emb)
        col_emb_out = self.col_encoding_block(col_emb, row_emb, edge_emb.transpose(2,1))

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

        self.mixed_score_MHA = MixedScore_MultiHeadAttention(**model_params)

    def forward(self, row_emb, col_emb, edge_emb):
        # NOTE: row and col can be exchanged, if cost_mat.transpose(1,2) is used
        # input1.shape: (batch, row_cnt, embedding)
        # input2.shape: (batch, col_cnt, embedding)
        # cost_mat.shape: (batch, row_cnt, col_cnt)
        head_num = self.model_params['head_num']

        q = reshape_by_heads(self.Wq(row_emb), head_num=head_num)
        # q shape: (batch, head_num, row_cnt, qkv_dim)
        k = reshape_by_heads(self.Wk(col_emb), head_num=head_num)
        v = reshape_by_heads(self.Wv(col_emb), head_num=head_num)

        out_concat = self.mixed_score_MHA(q, k, v, edge_emb)

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

class Decoder(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']
        head_num = self.model_params['head_num']
        qkv_dim = self.model_params['qkv_dim']

        self.Wq_1 = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wq_0 = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wk = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)

        self.multi_head_combine = nn.Linear(head_num * qkv_dim, embedding_dim)

        # direction token, created only for the 'shared' wiring (M3)
        if model_params.get('decoder_wiring', 'exchange') == 'shared':
            self.dir_embedding = nn.Parameter(torch.zeros(2, head_num * qkv_dim))
            nn.init.normal_(self.dir_embedding, std=0.1)
        else:
            self.dir_embedding = None

        self.k = {}  # saved keys per stream, for multi-head attention
        self.v = {}  # saved values per stream
        self.single_head_key = {}  # saved per stream, for single-head attention
        self.q1 = {}  # saved first-node query per direction

    def set_kv(self, encoded_nodes_f, encoded_nodes_t):
        # encoded_nodes.shape: (batch, problem, embedding)
        head_num = self.model_params['head_num']
        for name, enc in (('f', encoded_nodes_f), ('t', encoded_nodes_t)):
            self.k[name] = reshape_by_heads(self.Wk(enc), head_num=head_num)
            self.v[name] = reshape_by_heads(self.Wv(enc), head_num=head_num)
            # shape: (batch, head_num, problem, qkv_dim)
            self.single_head_key[name] = enc.transpose(1, 2)
            # shape: (batch, embedding, problem)

    def set_q1(self, encoded_q1_fwd, encoded_q1_bwd):
        # encoded_q1.shape: (batch, n_d, embedding), or None if the direction is unused
        head_num = self.model_params['head_num']
        self.q1 = {}
        if encoded_q1_fwd is not None:
            self.q1['fwd'] = reshape_by_heads(self.Wq_1(encoded_q1_fwd), head_num=head_num)
        if encoded_q1_bwd is not None:
            self.q1['bwd'] = reshape_by_heads(self.Wq_1(encoded_q1_bwd), head_num=head_num)
        # shape: (batch, head_num, n_d, qkv_dim)

    def forward(self, encoded_q0, ninf_mask, key_stream, direction, dir_idx=None):
        # encoded_q0.shape: (batch, n_d, embedding)
        # ninf_mask.shape: (batch, n_d, problem)
        head_num = self.model_params['head_num']
        k = self.k[key_stream]
        v = self.v[key_stream]
        single_head_key = self.single_head_key[key_stream]

        #  Multi-Head Attention
        #######################################################
        q0 = reshape_by_heads(self.Wq_0(encoded_q0), head_num=head_num)
        # shape: (batch, head_num, n_d, qkv_dim)

        q = self.q1[direction] + q0
        if dir_idx is not None:
            q = q + self.dir_embedding[dir_idx].reshape(1, head_num, 1, -1)
        # shape: (batch, head_num, n_d, qkv_dim)

        out_concat = multi_head_attention(q, k, v, rank3_ninf_mask=ninf_mask)
        # shape: (batch, n_d, head_num*qkv_dim)

        mh_atten_out = self.multi_head_combine(out_concat)
        # shape: (batch, n_d, embedding)

        #  Single-Head Attention, for probability calculation
        #######################################################
        score = torch.matmul(mh_atten_out, single_head_key)
        # shape: (batch, n_d, problem)

        sqrt_embedding_dim = self.model_params['sqrt_embedding_dim']
        logit_clipping = self.model_params['logit_clipping']

        score_scaled = score / sqrt_embedding_dim
        score_clipped = logit_clipping * torch.tanh(score_scaled)
        score_masked = score_clipped + ninf_mask

        probs = F.softmax(score_masked, dim=2)
        # shape: (batch, n_d, problem)

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


def multi_head_attention(q, k, v, rank2_ninf_mask=None, rank3_ninf_mask=None, mtrx=None):
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

    if mtrx is not None:
        score_scaled += mtrx.permute(0, 3, 1, 2)


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