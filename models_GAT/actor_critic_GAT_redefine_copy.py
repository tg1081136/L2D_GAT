import ipdb
import torch
import torch.nn as nn
import torch.nn.functional as F
from models_GAT.graphgat import GraphGAT
from models_GAT.mlp_GAT import MultiLayerPerceptron

class ActorCritic(nn.Module):
    def __init__(self, n_j, n_m, node_input_dim, graph_hidden_dim, graph_output_dim,
                 mlp_hidden_dim, mlp_output_dim, num_layers=3, heads=1, device='cpu'):
        super(ActorCritic, self).__init__()
        self.mlp_hidden_dim = mlp_hidden_dim
        self.n_j = n_j
        self.n_m = n_m
        self.device = device
        
        self.gnn = GraphGAT(node_input_dim, graph_hidden_dim, graph_output_dim,
                           num_layers=num_layers, heads=heads).to(device)

        self.actor_mlp = None  # 延遲初始化
        self.critic_mlp = MultiLayerPerceptron(graph_output_dim, self.mlp_hidden_dim,
                                              1, num_layers=3, activation='tanh', norm_type='layernorm').to(device)
        
        
    def forward(self, x, edge_index, graph_pool, omega, mask):
        # ===== 1. GNN encoder =====
        h_pooled, h_nodes = self.gnn(x, edge_index, graph_pool)  # [n_jobs * n_machines, dim] / [n_jobs, dim]

        # ===== 2. Critic =====
        h_global = torch.mean(h_pooled, dim=0, keepdim=True)  # [1, dim]
        v = self.critic_mlp(h_global).squeeze(-1)  # scalar state value

        # ===== 3. 靜態 candidate 構建 =====
        n_j = self.n_j
        n_m = self.n_m
        device = x.device

        candidate = torch.arange(n_j, device=device)  # [0, 1, ..., n_j - 1]
        omega_tensor = omega.to(device)               # shape: [n_j]

        # ===== 🛡️ sanity check before indexing =====
        if omega_tensor.shape[0] != n_j:
            raise ValueError(f"[ERROR] omega 長度錯誤: {omega_tensor.shape[0]} ≠ {n_j}")

        if not torch.all((0 <= omega_tensor) & (omega_tensor < n_m)):
            overflow_jobs = torch.where(omega_tensor >= n_m)[0].tolist()
            raise ValueError(f"[ERROR] 以下 job 的 omega 超過 machine 數 n_m={n_m} ➜ job_ids={overflow_jobs}")

        node_idx = candidate * n_m + omega_tensor  # [n_j]

        # ===== 4. 建立 actor 輸入 =====
        candidate_feature = h_nodes[node_idx]            # [n_j, dim]
        h_pooled_selected = h_pooled[candidate]          # [n_j, dim]
        concat_fea = torch.cat((candidate_feature, h_pooled_selected), dim=-1)  # [n_j, 2*dim]

        if self.actor_mlp is None:
            self.actor_mlp = MultiLayerPerceptron(
                input_dim=concat_fea.size(-1),
                hidden_dim=self.mlp_hidden_dim,
                output_dim=1,
                num_layers=3,
                activation='tanh',
                norm_type='layernorm'
            ).to(device)

        candidate_scores = self.actor_mlp(concat_fea).squeeze(-1)  # [n_j]
        candidate_scores = torch.clamp(candidate_scores, min=-10, max=10)

        # ===== 5. 採用 job-level mask =====
        if mask is not None:
            if not mask.any():
                raise ValueError("No valid jobs available in mask")
            elif mask.sum() == 1:
                pi = torch.zeros_like(candidate_scores)
                pi[mask] = 1.0
            else:
                candidate_scores = candidate_scores.masked_fill(~mask, float('-inf'))
                pi = F.softmax(candidate_scores, dim=-1)
        else:
            pi = F.softmax(candidate_scores, dim=-1)

        return pi.unsqueeze(0), v