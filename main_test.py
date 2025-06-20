import torch
import numpy as np
from models_GAT.graphgat import GraphGAT
from models_GAT.mlp_GAT import MultiLayerPerceptron
from models_GAT.actor_critic_GAT_redefine import ActorCritic  # 假設你已經封裝這個類別
from uniform_instance_gen import uni_instance_gen

# ✅ 環境參數
n_j = 2
n_m = 3
feature_dim = 2
graph_hidden_dim = 8
graph_output_dim = 8
mlp_hidden_dim = 16
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ✅ 資料生成
edge_index, fea, candidate, mask = uni_instance_gen(n_j, n_m, low=1, high=10, device=device)
assert edge_index.max().item() < fea.shape[0], "Edge index refers to non-existent node!"
fea = torch.tensor(fea, dtype=torch.float).to(device)
candidate = torch.tensor(candidate, dtype=torch.long).to(device)
mask = torch.tensor([True, False], device=device)
#mask = torch.tensor(mask, dtype=torch.bool).to(device)


# ✅ graph_pool 建立
row = np.repeat(np.arange(n_j), n_m)
col = np.arange(n_j * n_m)
value = np.full(n_j * n_m, 1 / n_m)
indices = torch.from_numpy(np.stack([row, col], axis=0)).long().to(device)
values = torch.tensor(value, dtype=torch.float).to(device)
graph_pool = torch.sparse_coo_tensor(indices, values, (n_j, n_j * n_m)).to(device)

# ✅ 模型初始化
model = ActorCritic(
    n_j=n_j,
    n_m=n_m,
    node_input_dim=feature_dim,
    graph_hidden_dim=graph_hidden_dim,
    graph_output_dim=graph_output_dim,
    mlp_hidden_dim=mlp_hidden_dim,
    mlp_output_dim=1,
    num_layers=3,
    heads=2,
    device=device
)

print(f"🧾 fea shape: {fea.shape}")
print(f"🧾 edge_index shape: {edge_index.shape}")
print(f"🧾 graph_pool shape: {graph_pool.shape}")
print(f"🧾 candidate: {candidate}")
print(f"🧾 mask: {mask}\n")
print("edge_index max:", edge_index.max().item())  # 應該 <= 5

# ✅ 前向傳遞
pi, v = model(fea, edge_index, graph_pool, candidate, mask)

# ✅ 結果輸出
print(f"\n🧩 candidate logits (pi): {pi.shape} → {pi.detach().cpu().numpy()}")
print(f"🧠 value estimate (v): {v.shape} → {v.detach().cpu().numpy()}")