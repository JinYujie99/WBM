import numpy as np
import torch

def probability_simplex_projection(x):
    descending_idx = torch.argsort(x, descending=True)
    u = x[descending_idx]
    rho= 0.
    lambda_= 1.
    for i in range(u.shape[0]):
        value = u[i] + (1- u[:(i+1)].sum())/(i+1)
        if value>0:
            rho+=1
            lambda_-=u[i]
        else:
            break
    return torch.max(x + lambda_/rho, torch.zeros_like(x))

