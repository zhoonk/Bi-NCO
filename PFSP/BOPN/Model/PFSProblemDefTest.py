
import torch
import numpy as np


def get_random_problems(batch_size, job_size, machine_size):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    problems = torch.load(
    "/home/inuai_11/Bi-NCO/PFSP/BOPN/Dataset/j50m20/PFSP50by20.pth",
    map_location="cpu",
    weights_only=True
    )
    problems = problems.to(device)

    return problems[:100]
