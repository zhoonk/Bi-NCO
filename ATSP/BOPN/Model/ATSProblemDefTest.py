import torch
import numpy as np


def get_random_problems(batch_size, node_cnt, aug_factor):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    problems = torch.load(
    "/home/inuai_11/Bi-NCO/ATSP/BOPN/Dataset/ATSP50/ATSP50.pth",
    map_location="cpu",
    weights_only=True
    )
    problems = problems.to(device)

    return problems[:100]