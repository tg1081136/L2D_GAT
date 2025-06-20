import torch
import torch.nn as nn
import torch.nn.functional as F
from models_GAT.graphgat import GraphGAT
from models_GAT.mlp_GAT import MultiLayerPerceptron

class ActorCritic(nn.Module):
    def __init__(self, n_j, n_m, node_input_dim, graph_hidden_dim, graph_output_dim,
                 mlp_hidden_dim, mlp_output_dim, num_layers, heads, device):
        super(ActorCritic, self).__init__()
        self.graph_model = GraphGAT(node_input_dim, graph_hidden_dim, graph_output_dim, num_layers, heads).to(device)
        self.n_j = n_j
        self.n_m = n_m
        self.mlp_hidden_dim = mlp_hidden_dim

        self.actor_mlp = MultiLayerPerceptron(2 * graph_output_dim, mlp_hidden_dim, 1, num_layers).to(device)  # 輸出 1 維
        self.critic_mlp = MultiLayerPerceptron(2 * graph_output_dim, mlp_hidden_dim, 1, num_layers).to(device)
        self.device = device

    def forward(self, x, edge_index, graph_pool, candidate, mask):
        h_pooled, h_nodes = self.graph_model(x, edge_index, graph_pool.to(self.device))
        print(f"h_pooled shape: {h_pooled.shape}, values: {h_pooled}, any NaN: {torch.isnan(h_pooled).any()}")
        print(f"h_nodes shape: {h_nodes.shape}, values: {h_nodes}, any NaN: {torch.isnan(h_nodes).any()}")

        candidate_feature = h_nodes[candidate]
        job_indices = (candidate // self.n_m)
        h_pooled_selected = h_pooled[job_indices]
        concat_fea = torch.cat((candidate_feature, h_pooled_selected), dim=-1)

        if self.actor_mlp is None:
            input_dim = concat_fea.size(-1)
            self.actor_mlp = MultiLayerPerceptron(input_dim, self.mlp_hidden_dim,
                                                1, num_layers=3, activation='tanh', norm_type='layernorm').to(self.device)

        candidate_scores = self.actor_mlp(concat_fea).squeeze(-1)
        print(f"candidate_scores shape: {candidate_scores.shape}, values: {candidate_scores}, any NaN: {torch.isnan(candidate_scores).any()}")
        print(f"candidate_scores min: {candidate_scores.min()}, max: {candidate_scores.max()}")

        if mask is not None:
            candidate_scores = torch.where(mask, torch.full_like(candidate_scores, float('-inf')), candidate_scores)
            print(f"candidate_scores after mask shape: {candidate_scores.shape}, values: {candidate_scores}, any NaN: {torch.isnan(candidate_scores).any()}")

        
        if torch.isinf(candidate_scores).all():
            print("All candidate_scores are inf or nan, using uniform fallback")
            pi = torch.ones_like(candidate_scores) / candidate.size(0)
        else:
            pi = F.softmax(candidate_scores, dim=0)
            print(f"pi shape: {pi.shape}, values: {pi}, sum: {pi.sum()}")

        v = self.critic_mlp(h_pooled).squeeze(-1)
        print(f"v shape: {v.shape}, values: {v}, any NaN: {torch.isnan(v).any()}")
        return pi, v