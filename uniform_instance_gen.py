import numpy as np
import torch
from Params import configs

def permute_rows(x):
    '''
    x is a np array
    '''
    ix_i = np.tile(np.arange(x.shape[0]), (x.shape[1], 1)).T
    ix_j = np.random.sample(x.shape).argsort(axis=1)
    return x[ix_i, ix_j]

def uni_instance_gen(n_j, n_m, low, high, device='cpu'):
    # Generate processing times and machine assignments
    times = np.random.randint(low=low, high=high, size=(n_j, n_m))
    machines = np.expand_dims(np.arange(1, n_m+1), axis=0).repeat(repeats=n_j, axis=0)
    machines = permute_rows(machines)

    # Generate node features (fea): [processing time, machine ID]
    num_nodes = n_j * n_m
    fea = np.zeros((num_nodes, 2))  # 2 features: time, machine
    for j in range(n_j):
        for m in range(n_m):
            node_idx = j * n_m + m
            fea[node_idx, 0] = times[j, m] / configs.et_normalize_coef
            fea[node_idx, 1] = machines[j, m] / configs.wkr_normalize_coef
    fea = fea.astype(np.float32)

    # Generate edge_index for processing order (directed edges within each job)
    edge_index = []
    for j in range(n_j):
        for m in range(n_m - 1):
            src = j * n_m + m
            dst = j * n_m + (m + 1)
            edge_index.append([src, dst])
    edge_index = np.array(edge_index).T
    edge_index = torch.tensor(edge_index, dtype=torch.long).to(device)

    # Generate candidate and mask (simplified: first operation of each job)
    candidate = np.array([j * n_m for j in range(n_j)])  # First operation of each job
    mask = np.zeros(n_j, dtype=bool)  # All candidates are valid initially

    return edge_index, fea, candidate, mask

def override(fn):
    """
    override decorator
    """
    return fn