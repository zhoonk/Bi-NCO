
import torch
import torch.nn as nn
import torch.nn.functional as F
# from PFSPModel_LIB import MixedScore_MultiHeadAttention


class PFSPModel(nn.Module):

    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        self.job_size = model_params['job_size']
        self.machine_size = model_params['machine_size']
        self.trajectory_size = model_params['trajectory_size']
        self.cross_encoder = Cross_Encoder(**model_params)
        self.decoder = PFSP_Decoder(**model_params)
        self.encoded_nodes_j = None
        self.encoded_nodes_m = None
        self.start = None
        self.end = None
        self.dz_cont = self.model_params['dz_cont']
        self.dz_cat = self.model_params['dz_cat']
        # shape: (batch, problem, EMBEDDING_DIM)

    def set_z(self, batch_size, sample_size):
        head_num = self.model_params['head_num']
        dz_cat = self.model_params['dz_cat']
        dz_cont = self.model_params['dz_cont']

        latent_c_var = torch.empty(batch_size, sample_size, dz_cont).uniform_(-1, 1)

        latent_d_var = torch.zeros((batch_size, sample_size, dz_cat), dtype=torch.float32)
        one_hot_idx = torch.randint(0, dz_cat, (batch_size, sample_size), dtype=torch.long)
        latent_d_var[torch.arange(batch_size).unsqueeze(1), torch.arange(sample_size).unsqueeze(0), one_hot_idx] = 1

        latent_var = torch.cat([latent_d_var, latent_c_var], dim=-1)
        return latent_var

    def pre_forward(self, reset_state):
        self.encoded_nodes_f, self.encoded_nodes_t, self.start, self.end = self.cross_encoder(reset_state.problems)
        # shape: (batch, problem, EMBEDDING_DIM)

        latent_cond_encoded_f = self.encoded_nodes_f.repeat_interleave(self.trajectory_size, dim=0) # shape: (batch*pomo, job, embedding)
        latent_cond_encoded_t = self.encoded_nodes_t.repeat_interleave(self.trajectory_size, dim=0) # shape: (batch*pomo, job, embedding)

        latent_dimension = self.dz_cont + self.dz_cat

        batch_size = self.start.size(0)

        self.latent_vector_f = self.set_z(batch_size, self.trajectory_size)
        re_latent_emb_f = self.latent_vector_f.reshape(batch_size*self.trajectory_size, 1, latent_dimension)
        latent_emb_f = re_latent_emb_f.expand(batch_size*self.trajectory_size, self.job_size, latent_dimension) # shape: (batch*pomo, job, embed)
        
        self.latent_vector_t = self.set_z(batch_size, self.trajectory_size)
        re_latent_emb_t = self.latent_vector_t.reshape(batch_size*self.trajectory_size, 1, latent_dimension)
        latent_emb_t = re_latent_emb_t.expand(batch_size*self.trajectory_size, self.job_size, latent_dimension) # shape: (batch*pomo, job, embed)


        #서로 내적하는 것이기에 kv로 쓸 땐 쿼리에 맞춰서 넣어줘야 함 == 원래 논문에 따라 똑같이 맞춰줘야 함
        latent_kv_f = torch.cat([latent_cond_encoded_f, latent_emb_t], dim=-1)
        latent_kv_t = torch.cat([latent_cond_encoded_t, latent_emb_f], dim=-1)

        self.decoder.set_kv(latent_kv_f, latent_kv_t)

    def forward(self, state):
        batch_size = state.BATCH_IDX.size(0)
        sample_size = state.BATCH_IDX.size(1)

        if state.current_node is None: 
            start = self.start.reshape(batch_size,1,self.start.size(-1)).repeat(1,self.trajectory_size,1)
            end = self.end.reshape(batch_size,1,self.end.size(-1)).repeat(1,self.trajectory_size,1)

            probs_Forward = self.decoder(self.encoded_nodes_f, start, self.latent_vector_f, ninf_mask=state.ninf_mask[:,:self.trajectory_size])
            probs_Backward = self.decoder(self.encoded_nodes_t, end, self.latent_vector_t, ninf_mask=state.ninf_mask[:,self.trajectory_size:],Backward = True)

            probs = torch.cat((probs_Forward,probs_Backward),dim=1)

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
        else:
            encoded_last_node_f = _get_encoding(self.encoded_nodes_f, state.current_node[:,:self.trajectory_size])
            encoded_last_node_t = _get_encoding(self.encoded_nodes_t, state.current_node[:,self.trajectory_size:])
            # shape: (batch, pomo, embedding)
            probs_Forward = self.decoder(self.encoded_nodes_f, encoded_last_node_f, self.latent_vector_f, ninf_mask=state.ninf_mask[:,:self.trajectory_size])
            probs_Backward = self.decoder(self.encoded_nodes_t, encoded_last_node_t, self.latent_vector_t, ninf_mask=state.ninf_mask[:,self.trajectory_size:], Backward=True)
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

        self.k = None  # saved key, for multi-head attention
        self.v = None  # saved value, for multi-head_attention
        self.single_head_key = None  # saved, for single-head attention

        self.feed_forward = Feed_Forward_Module(**model_params)

    def set_kv(self, encoded_nodes_f, encoded_nodes_t):
        # encoded_nodes.shape: (batch, problem, embedding)
        head_num = self.model_params['head_num']

        self.k_f = reshape_by_heads(self.Wk(encoded_nodes_f), head_num=head_num)
        self.v_f = reshape_by_heads(self.Wv(encoded_nodes_f), head_num=head_num)
        # shape: (batch, head_num, pomo, qkv_dim)
        self.single_head_key_f = self.Wp(encoded_nodes_f).transpose(1, 2)
        # shape: (batch, embedding, problem)

        self.k_t = reshape_by_heads(self.Wk(encoded_nodes_t), head_num=head_num)
        self.v_t = reshape_by_heads(self.Wv(encoded_nodes_t), head_num=head_num)
        # shape: (batch, head_num, pomo, qkv_dim)
        self.single_head_key_t = self.Wp(encoded_nodes_t).transpose(1, 2)
        # shape: (batch, embedding, problem)
        

    def forward(self, encoded_node, encoded_last_node, latent_vector, ninf_mask, Backward = False):
        # encoded_last_node.shape: (batch, pomo, embedding)
        # ninf_mask.shape: (batch, pomo, problem)

        if Backward == True:
            k = self.k_f
            v = self.v_f
            single_head_key = self.single_head_key_f
        else:
            k = self.k_t
            v = self.v_t
            single_head_key = self.single_head_key_t

        head_num = self.model_params['head_num']
        batch_size = encoded_last_node.size(0)
        trajectory_size = encoded_last_node.size(1)

        valid = (ninf_mask == 0).float()          # [100,128,20], allowed=1

        unvisited_node = valid @ encoded_node         # [100,128,256]
        cnt = valid.sum(dim=-1, keepdim=True).clamp_min(1.0)
        unvisited_node_avg = unvisited_node / cnt  

        context_embedding = self.Wq(torch.cat([encoded_last_node, unvisited_node_avg, latent_vector], dim=-1))
        reshaped_context_emb = context_embedding.reshape(batch_size*trajectory_size,1,context_embedding.size(-1))
        reshaped_ninf_mask = ninf_mask.reshape(batch_size*trajectory_size,1, ninf_mask.size(-1))

        q = reshape_by_heads(reshaped_context_emb, head_num=head_num)

        #  Multi-Head Attention
        #######################################################
        
        out_concat = multi_head_attention(q, k, v, rank3_ninf_mask=reshaped_ninf_mask)
        # shape: (batch, pomo, head_num*qkv_dim)

        mh_atten_out = self.multi_head_combine(out_concat)
        # shape: (batch, pomo, embedding)

        updated_context = self.feed_forward(mh_atten_out)
        pointer = mh_atten_out+updated_context

        #  Single-Head Attention, for probability calculation
        #######################################################
        score = torch.matmul(pointer, single_head_key)
        # shape: (batch, pomo, problem)

        sqrt_embedding_dim = self.model_params['sqrt_embedding_dim']
        logit_clipping = self.model_params['logit_clipping']

        score_scaled = score / sqrt_embedding_dim
        # shape: (batch, pomo, problem)

        score_clipped = logit_clipping * torch.tanh(score_scaled)

        score_masked = score_clipped + reshaped_ninf_mask
        score_masked = score_masked.reshape(batch_size,trajectory_size,reshaped_ninf_mask.size(-1))

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
