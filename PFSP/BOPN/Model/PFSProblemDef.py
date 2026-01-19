
import torch
import numpy as np


def get_random_problems(batch_size, job_size, machine_size):

    problems = torch.randint(1,100,size=(batch_size, job_size, machine_size))
    #problem scaling

    return problems

def augment_PFSP(problems):
    # problems.shape: (batch, problem, 2)
    problems_flip = problems.flip(dims=(2,))

    aug_problems = torch.cat((problems, problems_flip), dim=0)
    # shape: (8*batch, problem, 2)
   
    return aug_problems