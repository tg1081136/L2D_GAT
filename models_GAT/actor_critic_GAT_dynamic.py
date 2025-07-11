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

    def forward(self, x, edge_index, graph_pool, candidate, mask):
        #print(f"[DEBUG] x.shape: {x.shape}")
        h_pooled, h_nodes = self.gnn(x, edge_index, graph_pool)
        #print(f"[DEBUG] h_pooled.shape: {h_pooled.shape}, h_nodes.shape: {h_nodes.shape}")
        
        h_global = torch.mean(h_pooled, dim=0, keepdim=True)  # [1, graph_output_dim]
        #print(f"[DEBUG] h_global.shape: {h_global.shape}")
        v = self.critic_mlp(h_global).squeeze(-1)  # [] 或 [1]
        #print(f"[DEBUG] v.shape: {v.shape}, v: {v}")
        
        if candidate.dim() == 2:
            candidate = candidate.squeeze(0)
        if mask.dim() == 2:
            mask = mask.squeeze(0)
        #print(f"[DEBUG] candidate.shape: {candidate.shape}, mask.shape: {mask.shape}")
        
        if len(mask) != len(candidate):
            #print(f"[ERROR] Mismatch: mask length={len(mask)}, candidate length={len(candidate)}")
            ipdb.set_trace()
        
        candidate_feature = h_nodes[candidate]  # [n_candidates, graph_output_dim]
        job_indices = candidate // self.n_m
        h_pooled_selected = h_pooled[job_indices]  # [n_candidates, graph_output_dim]
        concat_fea = torch.cat((candidate_feature, h_pooled_selected), dim=-1)  # [n_candidates, 2 * graph_output_dim]
        
        if self.actor_mlp is None:
            input_dim = concat_fea.size(-1)
            self.actor_mlp = MultiLayerPerceptron(input_dim, self.mlp_hidden_dim,
                                                1, num_layers=3, activation='tanh', norm_type='layernorm').to(self.device)
        
        candidate_scores = self.actor_mlp(concat_fea).squeeze(-1)  # [n_candidates]
        candidate_scores = torch.clamp(candidate_scores, min=-10, max=10)
        
        
        if mask is not None:
            valid_count = mask.sum()
            #print(f"[DEBUG] mask: {mask}, valid candidates: {valid_count}")
            if valid_count == 0:
                #print("[ERROR] All candidates are masked out, entering ipdb")
                ipdb.set_trace()
                # 應由環境保證至少一個有效動作，暫不使用均勻分佈
                raise ValueError("No valid candidates in mask")
            elif valid_count == 1:
                #print("[DEBUG] Only one valid candidate.")
                pi = torch.zeros_like(candidate_scores)
                pi[mask] = 1.0
            else:
                candidate_scores = candidate_scores.masked_fill(~mask, float('-inf'))
                pi = F.softmax(candidate_scores, dim=-1)
        else:
            pi = F.softmax(candidate_scores, dim=-1)

        if pi.dim() == 0:
            pi = pi.unsqueeze(0)
            
        #print(f"[DEBUG] pi.shape: {pi.shape}, pi: {pi}")
        return pi.unsqueeze(0), v