
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
        self.cross_encoder = Cross_Encoder(**model_params)
        self.decoder = Decoder(**model_params)
        self.encoded_nodes_j = None
        self.encoded_nodes_m = None
        # shape: (batch, problem, EMBEDDING_DIM)

    def pre_forward(self, reset_state):
        self.encoded_nodes_f, self.encoded_nodes_t = self.cross_encoder(reset_state.problems)
        # shape: (batch, problem, EMBEDDING_DIM)
        self.decoder.set_kv(self.encoded_nodes_f, self.encoded_nodes_t)

    def forward(self, state):
        batch_size = state.BATCH_IDX.size(0)
        sample_size = state.BATCH_IDX.size(1)

        if state.current_node is None:
            node = torch.arange(self.node_cnt)
            selected = torch.cat((node,node),dim=0)[None, :].expand(batch_size, sample_size)
            prob = torch.ones(size=(batch_size, sample_size))

            encoded_first_row_f = _get_encoding(self.encoded_nodes_f, selected[:,:self.node_cnt])
            encoded_first_row_t = _get_encoding(self.encoded_nodes_t, selected[:,self.node_cnt:])
            # shape: (batch, pomo, embedding)
            self.decoder.set_q1(encoded_first_row_f, encoded_first_row_t)

        else:
            encoded_last_node_f = _get_encoding(self.encoded_nodes_f, state.current_node[:,:self.trajectory_size])
            encoded_last_node_t = _get_encoding(self.encoded_nodes_t, state.current_node[:,self.trajectory_size:])
            # shape: (batch, pomo, embedding)
            probs_Forward = self.decoder(encoded_last_node_f, ninf_mask=state.ninf_mask[:,:self.trajectory_size])
            probs_Backward = self.decoder(encoded_last_node_t, ninf_mask=state.ninf_mask[:,self.trajectory_size:], Backward=True)
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

        self.k = None  # saved key, for multi-head attention
        self.v = None  # saved value, for multi-head_attention
        self.single_head_key = None  # saved, for single-head attention
        self.q1_f = None  # saved q1, for multi-head attention
        self.q1_t = None  # saved q1, for multi-head attention

    def set_kv(self, encoded_nodes_f, encoded_nodes_t):
        # encoded_nodes.shape: (batch, problem, embedding)
        head_num = self.model_params['head_num']

        self.k_f = reshape_by_heads(self.Wk(encoded_nodes_f), head_num=head_num)
        self.v_f = reshape_by_heads(self.Wv(encoded_nodes_f), head_num=head_num)
        # shape: (batch, head_num, pomo, qkv_dim)
        self.single_head_key_f = encoded_nodes_f.transpose(1, 2)
        # shape: (batch, embedding, problem)

        self.k_t = reshape_by_heads(self.Wk(encoded_nodes_t), head_num=head_num)
        self.v_t = reshape_by_heads(self.Wv(encoded_nodes_t), head_num=head_num)
        # shape: (batch, head_num, pomo, qkv_dim)
        self.single_head_key_t = encoded_nodes_t.transpose(1, 2)
        # shape: (batch, embedding, problem)

    def set_q1(self, encoded_q1_f,encoded_q1_t):
        # encoded_q.shape: (batch, n, embedding)  # n can be 1 or pomo
        head_num = self.model_params['head_num']

        self.q1_f = reshape_by_heads(self.Wq_1(encoded_q1_f), head_num=head_num)
        self.q1_t = reshape_by_heads(self.Wq_1(encoded_q1_t), head_num=head_num)
        # shape: (batch, head_num, n, qkv_dim)

    def forward(self, encoded_q0, ninf_mask, Backward = False):
        # encoded_last_node.shape: (batch, pomo, embedding)
        # ninf_mask.shape: (batch, pomo, problem)
        head_num = self.model_params['head_num']

        if Backward == True:
            k = self.k_t
            v = self.v_t
            single_head_key = self.single_head_key_t
            q1 = self.q1_f
        else:
            k = self.k_f
            v = self.v_f
            single_head_key = self.single_head_key_f
            q1 = self.q1_t
            

        #  Multi-Head Attention
        #######################################################
        q0 = reshape_by_heads(self.Wq_0(encoded_q0), head_num=head_num)
        # shape: (batch, head_num, pomo, qkv_dim)

        q = q1 + q0
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
