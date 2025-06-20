import torch
import torch.nn as nn
import torch.nn.functional as F
from models_GAT.graphgat import GraphGAT
from models_GAT.mlp_GAT import MultiLayerPerceptron
#from Params import configs

class ActorCritic(nn.Module):
    def __init__(self, n_j, n_m, node_input_dim, graph_hidden_dim, graph_output_dim, 
                 mlp_hidden_dim, mlp_output_dim, num_layers=3, heads=1, device='cpu'):
        super(ActorCritic, self).__init__()
        self.n_j = n_j
        self.n_m = n_m
        self.n_ops_perjob = n_m
        self.device = device
        
        self.gnn = GraphGAT(node_input_dim, graph_hidden_dim, graph_output_dim, 
                           num_layers=num_layers, heads=heads).to(device)
        #輸入維度
        with torch.no_grad():
            dummy_feat = torch.randn(1, graph_output_dim - 1).to(self.device)  # 模擬 candidate node feature
            dummy_pool = torch.randn(1, graph_output_dim).to(self.device)      # 模擬 h_pooled
            concat_dim = dummy_feat.shape[-1] + dummy_pool.shape[-1]

        self.actor_input_dim = concat_dim
        self.actor_mlp = MultiLayerPerceptron(self.actor_input_dim, mlp_hidden_dim, 
                                              1, num_layers=3, activation='tanh').to(device)  # Output 1 for each candidate
        self.critic_mlp = MultiLayerPerceptron(graph_output_dim, mlp_hidden_dim, 
                                             1, num_layers=3, activation='tanh').to(device)
    
    def forward(self, x, edge_index, graph_pool, candidate, mask):
        # GNN forward pass
        h_pooled, h_nodes = self.gnn(x, edge_index, graph_pool)  # h_pooled: [n_j, graph_output_dim], h_nodes: [n_nodes, graph_output_dim]
        assert h_nodes.shape[0] == graph_pool.shape[1], f"Mismatch: h_nodes has {h_nodes.shape[0]} nodes, graph_pool expects {graph_pool.shape[1]}"
        
        # Actor: Process candidate features
        dummy = candidate.unsqueeze(0)  # [1, n_candidates]
        candidate_feature = torch.gather(h_nodes, 0, dummy.expand(h_nodes.size(0), -1)).transpose(0, 1)  # [n_candidates, graph_output_dim]
        
        # Select h_pooled based on candidate job indices
        job_indices = (candidate // self.n_m).unsqueeze(-1).expand(-1, h_pooled.size(-1))  # [n_candidates, graph_output_dim]
        h_pooled_selected = torch.gather(h_pooled, 0, job_indices)  # [n_candidates, graph_output_dim]
        
        # Concatenate features
        concate_fea = torch.cat((candidate_feature, h_pooled_selected), dim=-1)  # [n_candidates, 2 * graph_output_dim]
        
        print("candidate_feature shape:", candidate_feature.shape)         # ← [2, ?]
        print("h_pooled_selected shape:", h_pooled_selected.shape)         # ← [2, ?]
        print("concat feature shape:", concate_fea.shape)                  # ← [2, ?] ← 應該是 16

        # Compute action scores
        candidate_scores = self.actor_mlp(concate_fea)  # [n_candidates, 1]
        
        # Apply mask and compute policy
        if mask is not None:
            candidate_scores = candidate_scores.masked_fill(mask.unsqueeze(-1).expand_as(candidate_scores), float('-inf'))
        pi = F.softmax(candidate_scores, dim=0).squeeze(-1)  # [n_candidates]
        
        # Critic: Compute value
        v = self.critic_mlp(h_pooled).squeeze(-1)  # [n_j]
        
        return pi, v
