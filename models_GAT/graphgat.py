import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

class GraphGAT(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=3, heads=1):
        super(GraphGAT, self).__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.heads = heads
        self.output_dim = output_dim
        
        self.convs.append(GATConv(input_dim, hidden_dim, heads=heads, concat=True))
        self.bns.append(nn.BatchNorm1d(hidden_dim * heads))
        
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_dim * heads, hidden_dim, heads=heads, concat=True))
            self.bns.append(nn.BatchNorm1d(hidden_dim * heads))
        
        self.convs.append(GATConv(hidden_dim * heads, output_dim, heads=1, concat=False))
    
    def forward(self, x, edge_index, graph_pool):
        #print(f"[GNN Input] edge_index.shape: {edge_index.shape}, dtype: {edge_index.dtype}")
        assert isinstance(edge_index, torch.Tensor), "edge_index is not a Tensor!"
        assert edge_index.dtype in (torch.int64, torch.int32), f"edge_index has wrong dtype: {edge_index.dtype}"
        assert edge_index.dim() == 2 and edge_index.shape[0] == 2, f"edge_index shape wrong: {edge_index.shape}"

        for conv, bn in zip(self.convs[:-1], self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
        h_nodes = self.convs[-1](x, edge_index)

        #print(f"GAT h_nodes shape: {h_nodes.shape}")
        #print(f"GAT graph_pool shape: {graph_pool.shape}")
        assert h_nodes.shape[0] == graph_pool.shape[1], f"Graph node count mismatch: h_nodes={h_nodes.shape[0]}, expected={graph_pool.shape[1]}"

        h_pooled = torch.sparse.mm(graph_pool, h_nodes) if graph_pool is not None else h_nodes[:self.n_j]  # Fallback to first n_j nodes
        #print("Input node feature shape:", x.shape) #查看fea的形狀
        return h_pooled, h_nodes