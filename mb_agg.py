import torch
import scipy.sparse as sp
from Params import configs

def aggr_obs(edge_index_mb, n_node):
    # edge_index_mb is a list of edge_index tensors, each of shape [2, num_edges]
    # n_node is the number of nodes per graph (n_j * n_m)
    batch_edge_index = []
    batch_offset = 0
    for edge_index in edge_index_mb:
        # Offset node indices for each graph in the batch
        offset_edge_index = edge_index + batch_offset
        batch_edge_index.append(offset_edge_index)
        batch_offset += n_node
    # Concatenate all edge indices
    batch_edge_index = torch.cat(batch_edge_index, dim=1)
    return batch_edge_index.to(edge_index_mb[0].device)

import torch
import scipy.sparse as sp
from Params import configs

def g_pool_cal(graph_pool_type, batch_size, n_nodes, device):
    if graph_pool_type == 'job':
        n_j = configs.n_j
        n_m = configs.n_m
        num_nodes = n_j * n_m
        row, col = [], []
        for job_id in range(n_j):
            for op_id in range(n_m):
                node_idx = job_id * n_m + op_id
                row.append(job_id)
                col.append(node_idx)
        data = [1.0 / n_m] * len(row)  # Mean pooling within each job
        idx = torch.tensor([row, col], dtype=torch.long)
        graph_pool = torch.sparse_coo_tensor(idx, torch.tensor(data, dtype=torch.float), size=(n_j, num_nodes)).to(device)
        return graph_pool
    else:
        raise ValueError("Unsupported graph_pool_type: {}".format(graph_pool_type))

if __name__ == '__main__':
    print('Go home.')