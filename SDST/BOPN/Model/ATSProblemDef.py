import torch
import numpy as np


def get_random_problems(batch_size, node_cnt, aug_factor):

    if aug_factor is not None:
        device = torch.device(f'cuda:{torch.cuda.current_device()}' if torch.cuda.is_available() else 'cpu')
        scaled_problems = torch.load('dist_matrices.pt', map_location=device, weights_only=True)
    else:

        int_min = 0
        int_max = 1000*1000
        scaler = 1000*1000

        problems = torch.randint(low=int_min, high=int_max, size=(batch_size, node_cnt, node_cnt), dtype=torch.float64)
        # shape: (batch, node, node)
        problems[:, torch.arange(node_cnt), torch.arange(node_cnt)] = 0

        while True:
            old_problems = problems.clone()

            problems, _ = (problems[:, :, None, :] + problems[:, None, :, :].transpose(2, 3)).min(dim=3)
            # shape: (batch, node, node)

            if (problems == old_problems).all():
                break

        # Scale
        scaled_problems = problems / scaler


    return scaled_problems


