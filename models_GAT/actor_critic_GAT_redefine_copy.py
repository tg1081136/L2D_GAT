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
        h_pooled, h_nodes = self.gnn(x, edge_index, graph_pool)
        #print(f"h_pooled shape: {h_pooled.shape}, values: {h_pooled}, any NaN: {torch.isnan(h_pooled).any()}")
        #print(f"h_nodes shape: {h_nodes.shape}, values: {h_nodes}, any NaN: {torch.isnan(h_nodes).any()}")

        # 抽取 candidate node feature
        candidate_feature = h_nodes[candidate]  # [n_candidates, graph_output_dim]
        #print(f"candidate_feature shape: {candidate_feature.shape}, values: {candidate_feature}, any NaN: {torch.isnan(candidate_feature).any()}")

        # 對應的 job index 決定用哪個 pooled 向量
        job_indices = (candidate // self.n_m)
        h_pooled_selected = h_pooled[job_indices]  # [n_candidates, graph_output_dim]
        #print(f"h_pooled_selected shape: {h_pooled_selected.shape}, values: {h_pooled_selected}, any NaN: {torch.isnan(h_pooled_selected).any()}")

        # 合併兩個來源的特徵
        concat_fea = torch.cat((candidate_feature, h_pooled_selected), dim=-1)  # [n_candidates, 2 * graph_output_dim]
        #print(f"concat_fea shape: {concat_fea.shape}, values: {concat_fea}, any NaN: {torch.isnan(concat_fea).any()}")

        # 延遲初始化 actor_mlp
        if self.actor_mlp is None:
            input_dim = concat_fea.size(-1)
            self.actor_mlp = MultiLayerPerceptron(input_dim, self.mlp_hidden_dim,
                                                 1, num_layers=3, activation='tanh', norm_type='layernorm').to(self.device)
            # 檢查權重（若支援）
            try:
                if hasattr(self.actor_mlp, 'weight') and torch.isnan(self.actor_mlp.weight).any():
                    print("NaN detected in actor_mlp weights")
            except AttributeError:
                print("actor_mlp has no weight attribute, skipping check")

        candidate_scores = self.actor_mlp(concat_fea).squeeze(-1)
        #print(f"candidate_scores shape: {candidate_scores.shape}, values: {candidate_scores}, any NaN: {torch.isnan(candidate_scores).any()}")
        #print(f"candidate_scores min: {candidate_scores.min()}, max: {candidate_scores.max()}")

        #if mask.all():
            #print("⚠️ Warning: all candidates are masked out.")

        if mask is not None:
            candidate_scores = candidate_scores.masked_fill(mask, float('-inf'))
            #print(f"candidate_scores after mask shape: {candidate_scores.shape}, values: {candidate_scores}, any NaN: {torch.isnan(candidate_scores).any()}")

        if torch.isinf(candidate_scores).all() or torch.isnan(candidate_scores).any():
            pi = torch.ones_like(candidate_scores).squeeze(-1)
            pi = pi / pi.sum(dim = -1, keepdim=True) 
        else:
            pi = F.softmax(candidate_scores, dim = -1)
        #print(f"🎯 pi shape: {pi.shape}, values: {pi}, sum: {pi.sum()}")

        v = self.critic_mlp(h_pooled).squeeze(-1)  # [n_j]
        #print(f"v shape: {v.shape}, values: {v}, any NaN: {torch.isnan(v).any()}")
        return pi, v