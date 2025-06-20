from mb_agg import *
from agent_utils import eval_actions
from agent_utils import select_action
from models_GAT.actor_critic_GAT_redefine_copy import ActorCritic
from copy import deepcopy
import torch
import time
import torch.nn as nn
import numpy as np
from Params import configs
from validation import validate
from torch_geometric.utils import from_scipy_sparse_matrix
import scipy.sparse as sp

device = torch.device(configs.device)

def adj_to_edge_index(adj, device):
    adj_sparse = sp.csr_matrix(adj)
    edge_index, _ = from_scipy_sparse_matrix(adj_sparse)
    return edge_index.to(device)

class Memory:
    def __init__(self):
        self.adj_mb = []
        self.fea_mb = []
        self.candidate_mb = []
        self.mask_mb = []
        self.a_mb = []
        self.r_mb = []
        self.done_mb = []
        self.logprobs = []

    def clear_memory(self):
        del self.adj_mb[:]
        del self.fea_mb[:]
        del self.candidate_mb[:]
        del self.mask_mb[:]
        del self.a_mb[:]
        del self.r_mb[:]
        del self.done_mb[:]
        del self.logprobs[:]

class PPO:
    def __init__(self,
                 lr,
                 gamma,
                 k_epochs,
                 eps_clip,
                 n_j,
                 n_m,
                 node_input_dim,
                 graph_hidden_dim,
                 graph_output_dim,
                 mlp_hidden_dim,
                 mlp_output_dim,
                 num_layers=3,
                 heads=1):
        self.lr = lr
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs

        self.policy = ActorCritic(n_j=n_j,
                                  n_m=n_m,
                                  node_input_dim=node_input_dim,
                                  graph_hidden_dim=graph_hidden_dim,
                                  graph_output_dim=graph_output_dim,
                                  mlp_hidden_dim=mlp_hidden_dim,
                                  mlp_output_dim=mlp_output_dim,
                                  num_layers=num_layers,
                                  heads=heads,
                                  device=device)
        # dummy forward to trigger actor_mlp 建構
        with torch.no_grad():
            dummy_fea = torch.randn(5, node_input_dim).to(device)
            dummy_edge = torch.tensor([[0, 1], [1, 0]], dtype=torch.long).to(device)
            dummy_pool = torch.eye(1, 5).to(device)
            dummy_candidate = torch.tensor([0], dtype=torch.long).to(device)
            dummy_mask = torch.tensor([False]).to(device)
            self.policy(dummy_fea, dummy_edge, dummy_pool, dummy_candidate, dummy_mask)

        # ← 此時 actor_mlp 才真的建起來
        self.policy_old = deepcopy(self.policy)
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,
                                                        step_size=configs.decay_step_size,
                                                        gamma=configs.decay_ratio)
        self.V_loss_2 = nn.MSELoss()

    def update(self, memories, n_tasks, g_pool):
        vloss_coef = configs.vloss_coef
        ploss_coef = configs.ploss_coef
        entloss_coef = configs.entloss_coef

        rewards_all_env = []
        edge_index_mb_t_all_env = []
        fea_mb_t_all_env = []
        candidate_mb_t_all_env = []
        mask_mb_t_all_env = []
        a_mb_t_all_env = []
        old_logprobs_mb_t_all_env = []
        for i in range(len(memories)):
            rewards = []
            discounted_reward = 0
            for reward, is_terminal in zip(reversed(memories[i].r_mb), reversed(memories[i].done_mb)):
                if is_terminal:
                    discounted_reward = 0
                discounted_reward = reward + (self.gamma * discounted_reward)
                rewards.insert(0, discounted_reward)
            rewards = torch.tensor(rewards, dtype=torch.float).to(device)
            rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)
            rewards_all_env.append(rewards)
            # Directly use stored edge_index, no need for adj_to_edge_index
            edge_index_mb_t = memories[i].adj_mb  # adj_mb now stores edge_index
            edge_index_mb_t_all_env.append(edge_index_mb_t)
            fea_mb_t = torch.stack(memories[i].fea_mb).to(device)
            fea_mb_t = fea_mb_t.reshape(-1, fea_mb_t.size(-1))
            fea_mb_t_all_env.append(fea_mb_t)
            candidate_mb_t_all_env.append(torch.stack(memories[i].candidate_mb).to(device).squeeze())
            mask_mb_t_all_env.append(torch.stack(memories[i].mask_mb).to(device).squeeze())
            a_mb_t_all_env.append(torch.stack(memories[i].a_mb).to(device).squeeze())
            old_logprobs_mb_t_all_env.append(torch.stack(memories[i].logprobs).to(device).squeeze().detach())

        mb_g_pool = g_pool_cal(graph_pool_type=configs.graph_pool_type,
                               batch_size=None,
                               n_nodes=configs.n_j * configs.n_m,
                               device=device)
        for _ in range(self.k_epochs):
            loss_sum = 0
            vloss_sum = 0
            for i in range(len(memories)):
                pis, vals = self.policy(x=fea_mb_t_all_env[i],
                                        edge_index=edge_index_mb_t_all_env[i],
                                        graph_pool=mb_g_pool,
                                        candidate=candidate_mb_t_all_env[i],
                                        mask=mask_mb_t_all_env[i])
                logprobs, ent_loss = eval_actions(pis.squeeze(), a_mb_t_all_env[i])
                ratios = torch.exp(logprobs - old_logprobs_mb_t_all_env[i].detach())
                advantages = rewards_all_env[i] - vals.view(-1).detach()
                surr1 = ratios * advantages
                surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
                v_loss = self.V_loss_2(vals.squeeze(), rewards_all_env[i])
                p_loss = - torch.min(surr1, surr2).mean()
                ent_loss = - ent_loss.clone()
                loss = vloss_coef * v_loss + ploss_coef * p_loss + entloss_coef * ent_loss
                loss_sum += loss
                vloss_sum += v_loss
            self.optimizer.zero_grad()
            loss_sum.mean().backward()
            self.optimizer.step()
        self.policy_old.load_state_dict(self.policy.state_dict())
        if configs.decayflag:
            self.scheduler.step()
        return loss_sum.mean().item(), vloss_sum.mean().item()

def main():
    from JSSP_Env import SJSSP
    envs = [SJSSP(n_j=configs.n_j, n_m=configs.n_m, device=device) for _ in range(configs.num_envs)]
    from uniform_instance_gen import uni_instance_gen
    data_generator = uni_instance_gen

    #print(f"📌 Used mlp_hidden_dim = {configs.mlp_hidden_dim}")
    # Validate data format from data_generator
    sample_data = data_generator(n_j=configs.n_j, n_m=configs.n_m, low=configs.low, high=configs.high)
    assert len(sample_data) == 4, "data_generator should return (edge_index, fea, omega, mask)"
    assert isinstance(sample_data[1], (np.ndarray, torch.Tensor)), "fea should be a NumPy array or Tensor"

    dataLoaded = np.load('./DataGen/generatedData{}_{}_Seed{}.npy'.format(configs.n_j, configs.n_m, configs.np_seed_validation))
    vali_data = []
    for i in range(dataLoaded.shape[0]):
        adj, fea = dataLoaded[i][0], dataLoaded[i][1]
        edge_index = adj_to_edge_index(adj, device)  # Assuming adj_to_edge_index is defined
        candidate = np.array([j * configs.n_m for j in range(configs.n_j)])
        mask = np.ones(configs.n_j, dtype=bool)
        vali_data.append((edge_index, fea, candidate, mask))

    torch.manual_seed(configs.torch_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(configs.torch_seed)
    np.random.seed(configs.np_seed_train)

    memories = [Memory() for _ in range(configs.num_envs)]

    ppo = PPO(configs.lr, configs.gamma, configs.k_epochs, configs.eps_clip,
              n_j=configs.n_j, n_m=configs.n_m, node_input_dim=configs.input_dim,
              graph_hidden_dim=configs.hidden_dim, graph_output_dim=configs.output_dim,
              mlp_hidden_dim=configs.mlp_hidden_dim, mlp_output_dim=configs.mlp_output_dim,
              num_layers=configs.num_layers, heads=configs.heads)

    g_pool_step = g_pool_cal(graph_pool_type=configs.graph_pool_type,
                            batch_size=None,
                            n_nodes=configs.n_j * configs.n_m,
                            device=device)
    
    log = []
    validation_log = []
    optimal_gaps = []
    record = 100000
    for i_update in range(configs.max_updates):
        t3 = time.time()
        ep_rewards = [0 for _ in range(configs.num_envs)]
        # Initialize states
        edge_index_envs = []
        fea_envs = []
        candidate_envs = []
        mask_envs = []
        for i, env in enumerate(envs):
            state, info = env.reset(data_generator(n_j=configs.n_j, n_m=configs.n_m, low=configs.low, high=configs.high))
            edge_index, fea, candidate, mask = state
            edge_index_envs.append(edge_index)
            fea_envs.append(fea)
            candidate_envs.append(candidate)
            mask_envs.append(mask)
            ep_rewards[i] = -env.initQuality

        while True:
            fea_tensor_envs = [torch.from_numpy(np.copy(fea)).to(device) if isinstance(fea, np.ndarray) else fea.to(device) for fea in fea_envs]
            edge_index_envs = [edge_index.to(device) for edge_index in edge_index_envs]
            candidate_tensor_envs = [torch.from_numpy(np.copy(candidate)).to(device, dtype=torch.int64) for candidate in candidate_envs]
            mask_tensor_envs = [torch.from_numpy(np.copy(mask)).to(device) if isinstance(mask, np.ndarray) else mask.to(device) for mask in mask_envs]
            
            with torch.no_grad():
                action_envs = []
                a_idx_envs = []
                for i in range(configs.num_envs):
                    pi, _ = ppo.policy_old(x=fea_tensor_envs[i],
                                           edge_index=edge_index_envs[i],
                                           graph_pool=g_pool_step,
                                           candidate=candidate_tensor_envs[i].unsqueeze(0),
                                           mask=mask_tensor_envs[i].unsqueeze(0))
                    action, a_idx = select_action(pi, candidate_envs[i], memories[i])
                    action_envs.append(action)
                    a_idx_envs.append(a_idx)
            
            # Update states
            edge_index_envs = []
            fea_envs = []
            candidate_envs = []
            mask_envs = []
            edge_index_envs = []

            for i in range(configs.num_envs):
                # 執行一步環境
                edge_index, fea, reward, done, info = envs[i].step(action_envs[i].item())

                # ✅ 將 step 前用到的資訊存入記憶體
                memories[i].adj_mb.append(edge_index)  # ← 注意！不是 edge_index_envs[i]，而是這一輪剛 step 出來的
                memories[i].fea_mb.append(fea_tensor_envs[i])
                memories[i].candidate_mb.append(candidate_tensor_envs[i])
                memories[i].mask_mb.append(mask_tensor_envs[i])
                memories[i].a_mb.append(a_idx_envs[i])
                memories[i].r_mb.append(reward)
                memories[i].done_mb.append(done)
                ep_rewards[i] += reward

                # ✅ 更新下次要用的狀態
                candidate = info["omega"]
                mask = info["mask"]
                edge_index_envs.append(edge_index)
                fea_envs.append(fea)
                candidate_envs.append(candidate)
                mask_envs.append(mask)
            if envs[0].done():
                break
        # ... (rest of the main function)
        for j in range(configs.num_envs):
            ep_rewards[j] -= envs[j].posRewards

        loss, v_loss = ppo.update(memories, configs.n_j * configs.n_m, configs.graph_pool_type)
        for memory in memories:
            memory.clear_memory()
        mean_rewards_all_env = sum(ep_rewards) / len(ep_rewards)
        log.append([i_update, mean_rewards_all_env])
        if (i_update + 1) % 100 == 0:
            file_writing_obj = open('./log_{}_{}_{}_{}.txt'.format(
                configs.n_j, configs.n_m, configs.low, configs.high), 'w')
            file_writing_obj.write(str(log))

        print('Episode {}\t Last reward: {:.2f}\t Mean_Vloss: {:.8f}'.format(
            i_update + 1, mean_rewards_all_env, v_loss))
        
        t4 = time.time()
        if (i_update + 1) % 100 == 0:
            vali_result = - validate(vali_data, ppo.policy).mean()
            validation_log.append(vali_result)
            if vali_result < record:
                torch.save(ppo.policy.state_dict(), './{}.pth'.format(
                    str(configs.n_j) + '_' + str(configs.n_m) + '_' + str(configs.low) + '_' + str(configs.high)))
                record = vali_result
            print('The validation quality is:', vali_result)
            file_writing_obj1 = open(
                './vali_{}_{}_{}_{}.txt'.format(
                    configs.n_j, configs.n_m, configs.low, configs.high), 'w')
            file_writing_obj1.write(str(validation_log))
        t5 = time.time()

if __name__ == '__main__':
    total1 = time.time()
    main()
    total2 = time.time()