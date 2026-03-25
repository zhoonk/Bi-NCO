
import torch
import numpy as np


def get_random_problems(batch_size, job_size, machine_size):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # problems = torch.load(
    # f"/home/inuai_11/Bi-NCO/PFSP/BOPN/Dataset/j{job_size}m{machine_size}/PFSP{job_size}by{machine_size}.pth",
    # map_location="cpu",
    # weights_only=True
    # )
    problems = torch.load(
    f"/home/inuai_11/Bi-NCO/PFSP/BOPN/Dataset/taidata/tai{job_size}x{machine_size}_with_ub.pt",
    map_location="cpu",
    weights_only=True
    )
    problems = problems["data"].to(device)

    return problems
