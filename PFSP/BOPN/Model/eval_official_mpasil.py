"""2026-11-12: evaluate the official MP-ASIL PFSP checkpoints (Supplementary Material/Code/PFSP, checkpoint-1000.pt)
on our Taillard files with the UB reference. Only the model definition (PFSPModel.py, PFSPModel_Lib.py) of the
official code is imported; the environment below reproduces its masking and makespan computation.
Usage: python eval_official_mpasil.py <official_code_dir> <data_dir> <out_tsv>"""
import sys, os, csv, torch
CODE, DATA, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
sys.path.insert(0, CODE)
from PFSPModel import PFSPModel
from dataclasses import dataclass
torch.set_default_tensor_type('torch.cuda.FloatTensor')

@dataclass
class State:
    BATCH_IDX: torch.Tensor
    POMO_IDX: torch.Tensor
    current_node: torch.Tensor = None
    ninf_mask: torch.Tensor = None

@dataclass
class Reset:
    problems: torch.Tensor

def makespan(p, seq):
    # p: (b, job, mc); seq: (b, pomo, job)
    b, k, n = seq.shape; m = p.size(2)
    t = torch.gather(p.unsqueeze(1).expand(b, k, n, m), 2, seq.unsqueeze(-1).expand(b, k, n, m))
    c = torch.zeros(b, k, m)
    for j in range(n):
        prev = torch.zeros(b, k)
        new = torch.zeros(b, k, m)
        for i in range(m):
            prev = torch.maximum(prev, c[:, :, i]) + t[:, :, j, i]
            new[:, :, i] = prev
        c = new
    return c[:, :, -1]

def params(mc, pomo, eval_type):
    return dict(embedding_dim=128, sqrt_embedding_dim=128**0.5, encoder_layer_num=6, qkv_dim=16, sqrt_qkv_dim=4.0,
                head_num=8, logit_clipping=10, ff_hidden_dim=512, ms_hidden_dim=16, ms_layer1_init=(1/2)**0.5,
                ms_layer2_init=(1/16)**0.5, eval_type=eval_type, one_hot_seed_cnt=mc, latent_cont_size=4,
                latent_disc_size=12, temperature=1, pomo_size=pomo, mc_cnt=mc, eval_pomo_size=pomo)

@torch.no_grad()
def run(model, probs_all, pomo, seed, bs=10):
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    out = []
    for lo in range(0, probs_all.size(0), bs):
        p = probs_all[lo:lo+bs]; b, n, m = p.shape
        zc = torch.empty(b, pomo, 4).uniform_(-1, 1)
        zd = torch.zeros(b, pomo, 12); zd[torch.arange(b)[:, None], torch.arange(pomo)[None, :], torch.randint(0, 12, (b, pomo))] = 1
        model.pre_forward(Reset(p), torch.cat([zd, zc], -1))
        st = State(torch.arange(b)[:, None].expand(b, pomo), torch.arange(pomo)[None, :].expand(b, pomo))
        st.ninf_mask = torch.zeros(b, pomo, n); seq = []
        for _ in range(n):
            sel, _ = model(st)
            seq.append(sel); st.current_node = sel
            st.ninf_mask[st.BATCH_IDX, st.POMO_IDX, sel] = float('-inf')
        out.append(makespan(p, torch.stack(seq, 2)).min(dim=1).values)
    return torch.cat(out)

rows = []
for mc, sizes in [(10, [20, 50, 100, 200]), (5, [20, 50, 100])]:
    sd = torch.load(os.path.join(CODE, f'result/saved_PFSP20x{mc}/checkpoint-1000.pt'), map_location='cuda', weights_only=True)
    for mode, pomo, et in [('softmax128', 128, 'softmax'), ('argmax128', 128, 'argmax'), ('G', 1, 'argmax')]:
        model = PFSPModel(**params(mc, pomo, et)).cuda(); model.load_state_dict(sd); model.eval()
        for n in sizes:
            d = torch.load(os.path.join(DATA, f'tai{n}x{mc}.pt'))
            prob = d['data'].float().cuda(); ub = d['ub'].float().cuda()
            best = run(model, prob, pomo, 1234)
            gaps = ((best - ub) / ub * 100).tolist()
            rows.append((f'20x{mc}', f'{n}x{mc}', mode, sum(gaps)/len(gaps)))
            with open(os.path.join(os.path.dirname(OUT), f'per_instance_official_{mode}_tai{n}x{mc}.csv'), 'w') as f:
                w = csv.writer(f); w.writerow(['instance', 'cost', 'ref'])
                for i, (c, r) in enumerate(zip(best.tolist(), ub.tolist())): w.writerow([i, c, r])
            print(rows[-1], flush=True)
with open(OUT, 'w') as f:
    f.write('train\ttest\tmode\tgap\n'); [f.write('\t'.join(map(str, r)) + '\n') for r in rows]
