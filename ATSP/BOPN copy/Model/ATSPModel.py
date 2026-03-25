

import torch
import torch.nn as nn
import torch.nn.functional as F
from ATSPModel_LIB import MixedScore_MultiHeadAttention


class BOPN_Model(nn.Module):

    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        self.node_cnt = model_params['node_cnt']
        self.trajectory_size = model_params['trajectory_size']
        self.encoder = Encoder(**model_params)
        self.decoder = Decoder(**model_params)
        self.encoded_nodes = None
        # shape: (batch, problem, EMBEDDING_DIM)

    def pre_forward(self, reset_state):
        self.encoded_nodes = self.encoder(reset_state.problems)
        # shape: (batch, problem, EMBEDDING_DIM)
        self.decoder.set_kv(self.encoded_nodes)

    def forward(self, state):
        batch_size = state.BATCH_IDX.size(0)
        sample_size = state.BATCH_IDX.size(1)

        if state.current_node is None:
            # node = torch.arange(self.node_cnt)
            # selected = torch.cat((node,node),dim=0)[None, :].expand(batch_size, sample_size)
            selected = torch.randint(
                low=0,
                high=self.node_cnt,              # self.node_cnt 미포함
                size=(batch_size, sample_size),  
                dtype=torch.long
            )
            prob = torch.ones(size=(batch_size, sample_size))

            encoded_first = _get_encoding(self.encoded_nodes, selected)
            # shape: (batch, pomo, embedding)
            self.decoder.set_q1(encoded_first)

        else:
            encoded_last_node_f = _get_encoding(self.encoded_nodes, state.current_node[:,:self.trajectory_size])
            encoded_last_node_t = _get_encoding(self.encoded_nodes, state.current_node[:,self.trajectory_size:])
            # shape: (batch, pomo, embedding)
            probs_Forward = self.decoder(self.encoded_nodes,encoded_last_node_f, ninf_mask=state.ninf_mask[:,:self.trajectory_size])
            probs_Backward = self.decoder(self.encoded_nodes, encoded_last_node_t, ninf_mask=state.ninf_mask[:,self.trajectory_size:], Backward=True)
            # shape: (batch, pomo, problem)

            probs = torch.cat((probs_Forward, probs_Backward), dim=1)

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
class Encoder(nn.Module):
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

        out = self.node_idx_projection(torch.rand((batch_size, num_nodes, 1)))
        
        scaled_data = self.compute_normalized_matrices(data)
        
        edge_emb = self.edge_mtrx_projection(scaled_data.float().unsqueeze(-1))

        for layer in self.layers:
            out = layer(out, edge_emb)

        return out


class EncoderLayer(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.encoding_block = EncodingBlock(**model_params)

    def forward(self, node_emb, edge_emb):
        # row_emb.shape: (batch, row_cnt, embedding)
        # col_emb.shape: (batch, col_cnt, embedding)
        # cost_mat.shape: (batch, row_cnt, col_cnt)
        node_emb = self.encoding_block(node_emb, edge_emb)

        return node_emb

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

    def forward(self, node_emb, edge_emb):
        # NOTE: row and col can be exchanged, if cost_mat.transpose(1,2) is used
        # input1.shape: (batch, row_cnt, embedding)
        # input2.shape: (batch, col_cnt, embedding)
        # cost_mat.shape: (batch, row_cnt, col_cnt)
        head_num = self.model_params['head_num']

        q = reshape_by_heads(self.Wq(node_emb), head_num=head_num)
        # q shape: (batch, head_num, row_cnt, qkv_dim)
        k = reshape_by_heads(self.Wk(node_emb), head_num=head_num)
        v = reshape_by_heads(self.Wv(node_emb), head_num=head_num)

        out_concat = self.mixed_score_MHA(q, k, v, edge_emb)

        # shape: (batch, row_cnt, head_num*qkv_dim)

        multi_head_out = self.multi_head_combine(out_concat)
        # shape: (batch, row_cnt, embedding)

        out1 = self.add_n_normalization_1(node_emb, multi_head_out)
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

        self.Wq_f = nn.Linear(3*embedding_dim, head_num * qkv_dim, bias=False)
        self.Wk_f = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv_f = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wp_f = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wq_t = nn.Linear(3*embedding_dim, head_num * qkv_dim, bias=False)
        self.Wk_t = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv_t = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wp_t = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)

        self.multi_head_combine = nn.Linear(head_num * qkv_dim, embedding_dim)

        self.k = None  # saved key, for multi-head attention
        self.v = None  # saved value, for multi-head_attention
        self.single_head_key = None  # saved, for single-head attention


    def set_kv(self, encoded_nodes):
        # encoded_nodes.shape: (batch, problem, embedding)
        head_num = self.model_params['head_num']

        self.k_f = reshape_by_heads(self.Wk_f(encoded_nodes), head_num=head_num)
        self.v_f = reshape_by_heads(self.Wv_f(encoded_nodes), head_num=head_num)
        # shape: (batch, head_num, pomo, qkv_dim)
        self.single_head_key_f = self.Wp_f(encoded_nodes).transpose(1, 2)
        # shape: (batch, embedding, problem)

        self.k_t = reshape_by_heads(self.Wk_t(encoded_nodes), head_num=head_num)
        self.v_t = reshape_by_heads(self.Wv_t(encoded_nodes), head_num=head_num)
        # shape: (batch, head_num, pomo, qkv_dim)
        self.single_head_key_t = self.Wp_t(encoded_nodes).transpose(1, 2)
        # shape: (batch, embedding, problem)

    def set_q1(self, encoded_q1):
        # encoded_q.shape: (batch, n, embedding)  # n can be 1 or pomo
        head_num = self.model_params['head_num']
        trajectory_size = self.model_params['trajectory_size']

        self.q1_f = encoded_q1[:,:trajectory_size]
        self.q1_t = encoded_q1[:,trajectory_size:]
        # shape: (batch, head_num, n, qkv_dim)

    def forward(self, encoded_node, encoded_q0, ninf_mask, Backward = False):
        # encoded_last_node.shape: (batch, pomo, embedding)
        # ninf_mask.shape: (batch, pomo, problem)
        head_num = self.model_params['head_num']

        valid = (ninf_mask == 0).float()          # [100,128,20], allowed=1
        masked_sum_node = valid @ encoded_node         # [100,128,256]
        cnt = valid.sum(dim=-1, keepdim=True).clamp_min(1.0)
        unvisited_node = masked_sum_node / cnt

        # backward와 forward 분기처리 잘못 되었음.
        # 그러나 단순 인코더 출처 구분이니 문제는 안됨 추후 수정
        if Backward == True:
            k = self.k_f
            v = self.v_f
            single_head_key = self.single_head_key_f
            q_context = torch.cat((unvisited_node, encoded_q0, self.q1_t), dim=-1)
            q = reshape_by_heads(self.Wq_t(q_context), head_num=head_num) 
        else:
            k = self.k_t
            v = self.v_t
            single_head_key = self.single_head_key_t
            q_context = torch.cat((unvisited_node, encoded_q0, self.q1_f), dim=-1)
            q = reshape_by_heads(self.Wq_f(q_context), head_num=head_num)
           
        #  Multi-Head Attention
        #######################################################
        
        # shape: (batch, head_num, pomo, qkv_dim)

        out_concat = multi_head_attention(q, k, v, rank3_ninf_mask=ninf_mask)
        # shape: (batch, pomo, head_num*qkv_dim)

        mh_atten_out = self.multi_head_combine(out_concat)
        # shape: (batch, pomo, embedding)

        #  Single-Head Attention, for probability calculation
        #######################################################
        score = torch.matmul(mh_atten_out, single_head_key)
        # shape: (batch, pomo, problem)

        sqrt_embedding_dim = self.model_params['sqrt_embedding_dim']
        logit_clipping = self.model_params['logit_clipping']

        score_scaled = score / sqrt_embedding_dim
        # shape: (batch, pomo, problem)

        score_clipped = logit_clipping * torch.tanh(score_scaled)

        score_masked = score_clipped + ninf_mask

        probs = F.softmax(score_masked, dim=2)
        # shape: (batch, pomo, problem)

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