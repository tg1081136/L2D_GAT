from mb_agg import *
from agent_utils import eval_actions
from agent_utils import select_action
from models_GAT.actor_critic_GAT_redefine_copy import ActorCritic
from copy import deepcopy
import ipdb
import torch
import time
import torch.nn as nn
import numpy as np
from Params import configs
from validation import validate
from torch_geometric.utils import from_scipy_sparse_matrix
import scipy.sparse as sp
import csv

device = torch.device(configs.device)

def adj_to_edge_index(adj, device):
    adj_sparse = sp.csr_matrix(adj)
    edge_index, _ = from_scipy_sparse_matrix(adj_sparse)
    return edge_index.to(device)

class Memory:
    def __init__(self):
        self.adj_mb = []
        self.fea_mb = []
        self.omega_mb = []
        self.mask_mb = []
        self.a_mb = []
        self.r_mb = []
        self.done_mb = []
        self.logprobs = []

    def clear_memory(self):
        del self.adj_mb[:]
        del self.fea_mb[:]
        del self.omega_mb[:]
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
        with torch.no_grad():
            n_nodes = n_j * n_m
            dummy_fea = torch.zeros(n_nodes, node_input_dim).to(device)
            dummy_edge = torch.randint(0, n_nodes, (2, 10), dtype=torch.long).to(device)  # 隨便造一組合理邊
            dummy_pool = torch.eye(n_j).repeat(1, int(n_nodes / n_j)).to(device)[:n_j]    # 每個 job 對應 pool

            dummy_omega = torch.zeros(n_j, dtype=torch.long).to(device)                   # 每個 job 初始 op = 0
            dummy_mask = torch.ones(n_j, dtype=torch.bool).to(device)                     # 全部合法 job

            self.policy(dummy_fea, dummy_edge, dummy_pool, dummy_omega, dummy_mask)

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
        #candidate_mb_t_all_env = []
        omega_mb_t_all_env = []
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

            edge_index_mb_t = memories[i].adj_mb
            
            edge_index_mb_t_all_env.append(edge_index_mb_t)

            fea_mb_t = torch.stack(memories[i].fea_mb).to(device)
            fea_mb_t_all_env.append(fea_mb_t)
            omega_mb_t_all_env.append(torch.stack(memories[i].omega_mb).to(device))  
            #candidate_mb_t_all_env.append(torch.stack(memories[i].candidate_mb).to(device).squeeze())
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
                T = len(memories[i].r_mb)
                all_vals = []
                all_pis = []
                for t in range(T):
                    x = fea_mb_t_all_env[i][t]
                    if x.dim() == 3:
                        x = x.squeeze(0)
                    assert isinstance(x, torch.Tensor), f"x is not a tensor: {type(x)}"
                    assert x.dim() == 2, f"x dimension error: {x.shape}"
                    
                    omega = omega_mb_t_all_env[i][t].clone().to(device)

                    try:
                        pis, vals = self.policy(
                            x=x,
                            edge_index=edge_index_mb_t_all_env[i][t],
                            graph_pool=mb_g_pool,
                            omega=omega,
                            mask=mask_mb_t_all_env[i][t]
                        )
                        #print(f"[DEBUG] t={t}, pis.shape: {pis.shape}, vals.shape: {vals.shape}, vals: {vals}")
                        all_vals.append(vals if vals.dim() == 0 else vals.squeeze())
                        all_pis.append(pis.squeeze(0))
                    except Exception as e:
                        print(f"[ERROR] Policy forward failed: {e}")
                        ipdb.set_trace()
                        raise
                
                try:
                    vals = torch.stack(all_vals)
                    #print(f"[DEBUG] vals.shape: {vals.shape}, rewards.shape: {rewards_all_env[i].shape}")
                    if vals.shape != rewards_all_env[i].shape:
                        print(f"[ERROR] Shape mismatch: vals.shape={vals.shape}, rewards.shape={rewards_all_env[i].shape}")
                        ipdb.set_trace()
                    assert vals.shape == rewards_all_env[i].shape, f"vals and rewards shape mismatch: {vals.shape} vs {rewards_all_env[i].shape}"
                except Exception as e:
                    print(f"[ERROR] Vals stacking failed: {e}")
                    ipdb.set_trace()
                    raise
                
                pis = torch.stack(all_pis)
                logprobs = []
                entropies = []
                for t in range(T):
                    probs = pis[t]
                    action = a_mb_t_all_env[i][t]
                    mask = mask_mb_t_all_env[i][t]
                    #candidate = candidate_mb_t_all_env[i][t]
                    #print(f"[DEBUG] t={t}, probs: {probs}, action: {action}, mask: {mask}, candidate: {candidate}")
                    
                    valid_indices = torch.where(mask)[0]
                    if len(valid_indices) == 1 and action not in valid_indices:
                        print(f"[ERROR] Invalid action {action} for mask {mask}, valid indices: {valid_indices}")
                        ipdb.set_trace()
                    
                    try:
                        logp, entropy = eval_actions(probs, action)
                    except Exception as e:
                        print(f"[ERROR] Eval_actions failed: {e}")
                        #print(f"[DEBUG] t={t}, probs: {probs}, action: {action}, candidate: {candidate}, mask: {mask}")
                        ipdb.set_trace()
                        raise
                    
                    logprobs.append(logp)
                    entropies.append(entropy)
                
                logprobs = torch.stack(logprobs)
                ent_loss = torch.stack(entropies).mean()
                ratios = torch.exp(logprobs - old_logprobs_mb_t_all_env[i].detach())
                advantages = rewards_all_env[i] - vals.detach()
                surr1 = ratios * advantages
                surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
                v_loss = self.V_loss_2(vals, rewards_all_env[i])
                p_loss = -torch.min(surr1, surr2).mean()
                ent_loss = -ent_loss.clone()
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
    from uniform_instance_gen import uni_instance_gen
    data_generator = uni_instance_gen

    configs.num_envs = 1  # 設置為 1 簡化調試
    envs = [SJSSP(n_j=configs.n_j, n_m=configs.n_m, device=device) for _ in range(configs.num_envs)]
    
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
    
    max_steps =configs.max_steps  
    log = []
    record = float('inf')
    
    for i_update in range(configs.max_updates):
        #print(f"[INFO] Update {i_update} started")
        t3 = time.time()
        ep_rewards = [0.0 for _ in range(configs.num_envs)]
        t_step = 0
        edge_index_envs = []
        fea_envs = []
        omega_envs = []
        #candidate_envs = []
        mask_envs = []

        for i, env in enumerate(envs):
            state, info = env.reset(data_generator(n_j=configs.n_j, n_m=configs.n_m, low=configs.low, high=configs.high))
            edge_index, fea, omega, mask = state
            #print(f"[DEBUG] env={i} reset: candidate={candidate}, mask={mask}, done={env.done()}")
            if env.done():
                print(f"[ERROR] env={i} done after reset")
                ipdb.set_trace()
            if not np.any(mask):
                print(f"[ERROR] No valid candidates in reset mask for env {i}, candidate={candidate}, mask={mask}")
                ipdb.set_trace()

            '''
            if len(mask) != len(candidate):
                print(f"[ERROR] Mismatch: mask length={len(mask)}, candidate length={len(candidate)}")
                ipdb.set_trace()
            '''

            edge_index_envs.append(edge_index)
            fea_envs.append(fea)
            omega_envs.append(omega)
            #candidate_envs.append(candidate)
            mask_envs.append(mask)
            ep_rewards[i] = -env.initQuality

        # rollout the env
        while True:
            t_step += 1
            #print(f"[INFO] Time step: {t_step}")
            
            fea_tensor_envs = [torch.tensor(fea, device=device, dtype=torch.float32) if isinstance(fea, np.ndarray) else fea.to(device) for fea in fea_envs]
            edge_index_envs = [edge_index.to(device) for edge_index in edge_index_envs]
            omega_tensor_envs = [torch.tensor(omega, dtype=torch.long, device=device) for omega in omega_envs]
            mask_tensor_envs = [torch.tensor(mask, dtype=torch.bool, device=device) for mask in mask_envs]
            #candidate_tensor_envs = [torch.tensor(candidate, device=device, dtype=torch.int64) for candidate in candidate_envs]
        

            with torch.no_grad():
                action_envs = []
                a_idx_envs = []
                for i in range(configs.num_envs):
                    if envs[i].done():
                        print(f"[WARNING] env={i} is done, skipping")
                        continue
                    
                    if not mask_tensor_envs[i].any():
                        print(f"[ERROR] No valid candidates in mask for env {i}, omega={omega_envs[i]}, mask={mask_tensor_envs[i]}")
                        ipdb.set_trace()

                    for j in range(len(omega_envs[i])):
                        if omega_envs[i][j] >= configs.n_m:
                            mask_envs[i][j] = False  # 🛡️ 更新原始 mask
                    mask_tensor_envs[i] = torch.tensor(mask_envs[i], dtype=torch.bool, device=device)  # 🆕 重建 tensor
                    
                    pi, _ = ppo.policy_old(
                        x=fea_tensor_envs[i],
                        edge_index=edge_index_envs[i],
                        graph_pool=g_pool_step,
                        omega=omega_tensor_envs[i],
                        #candidate=candidate_tensor_envs[i],
                        mask=mask_tensor_envs[i]
                    )

                    if pi.dim() == 0:
                        pi = pi.unsqueeze(0)
                    elif pi.dim() > 1:
                        pi = pi.squeeze(0)

                    action, a_idx = select_action(
                        pi.squeeze(0),
                        #torch.tensor(candidate_envs[i], dtype=torch.long, device=device),
                        torch.tensor(mask_envs[i], dtype=torch.bool, device=device),
                        memories[i]
                    )
                    print(f"[DEBUG] env={i}, pi: {pi.squeeze(0)}, omega: {omega_envs[i]}, mask: {mask_envs[i]}, action: {action}, a_idx: {a_idx}")
                    
                    valid_indices = np.where(mask_envs[i])[0]
                    if len(valid_indices) == 0:
                        #print(f"[ERROR] No valid candidates after select_action for env {i}")
                        ipdb.set_trace()
                    elif len(valid_indices) == 1 and a_idx != valid_indices[0]:
                        #print(f"[ERROR] Invalid a_idx {a_idx}, expected {valid_indices[0]}")
                        ipdb.set_trace()

                    job_id = a_idx  # 選到的 job index

                    if job_id >= configs.n_j or omega_envs[i][job_id] >= configs.n_m:
                        print(f"[CRASH GUARD] job_id={job_id}, omega={omega_envs[i][job_id]} 超範圍")
                        ipdb.set_trace()

                    node_id = job_id * configs.n_m + omega_envs[i][job_id]  
                    
                    action_envs.append(node_id)
                    a_idx_envs.append(a_idx)
            
            new_edge_index_envs = []
            new_fea_envs = []
            new_omega_envs = []
            #new_candidate_envs = []
            new_mask_envs = []
            
            for i in range(configs.num_envs):
                if envs[i].done():
                    continue
                edge_index, fea, reward, done, info = envs[i].step(action_envs[i].item())
                candidate = info["omega"]
                mask = info["mask"]
                #print(f"[DEBUG] env={i}, step action: {action_envs[i]}, candidate: {candidate}, mask: {mask}, reward: {reward}, done: {done}")
                
                #candidate_tensor_envs[i] = torch.tensor(candidate, dtype=torch.long, device=device)
                omega_tensor_envs = [torch.tensor(omega, dtype=torch.long, device=device) for omega in omega_envs]
                mask_tensor_envs[i] = torch.tensor(mask, dtype=torch.bool, device=device)
                
                if not isinstance(candidate, (list, np.ndarray, torch.Tensor)):
                    print(f"[ERROR] Invalid candidate type: {type(candidate)}")
                    ipdb.set_trace()
                if not isinstance(mask, (list, np.ndarray, torch.Tensor)):
                    print(f"[ERROR] Invalid mask type: {type(mask)}")
                    ipdb.set_trace()
                if len(mask) != len(candidate):
                    print(f"[ERROR] Mismatch: mask length={len(mask)}, candidate length={len(candidate)}")
                    ipdb.set_trace()
                if not done and not np.any(mask):
                    print(f"[ERROR] No valid candidates in step mask for env {i}, candidate={candidate}, mask={mask}")
                    ipdb.set_trace()
                
                assert isinstance(edge_index, torch.Tensor), f"edge_index type: {type(edge_index)}"
                assert edge_index.dtype in (torch.int64, torch.int32), f"edge_index dtype: {edge_index.dtype}"
                assert edge_index.shape[0] == 2, f"edge_index shape: {edge_index.shape}"
                
                memories[i].adj_mb.append(edge_index.clone().detach())
                memories[i].fea_mb.append(fea_tensor_envs[i])

                memories[i].mask_mb.append(mask_tensor_envs[i].clone().view(-1))
                memories[i].omega_mb.append(torch.tensor(omega_envs[i], dtype=torch.long).clone().to(device))
                #memories[i].candidate_mb.append(candidate_tensor_envs[i].clone().view(-1))

                #print(f"動作張量形狀: {a_idx_envs[i].shape}, 值: {a_idx_envs[i]}")
                memories[i].a_mb.append(a_idx_envs[i].clone().detach().view(-1).to(device))
                memories[i].r_mb.append(reward)
                memories[i].done_mb.append(done)
                ep_rewards[i] += reward
                
                new_edge_index_envs.append(edge_index)
                new_fea_envs.append(fea)
                new_omega_envs.append(omega)
                new_mask_envs.append(mask)
            
            edge_index_envs = new_edge_index_envs
            fea_envs = new_fea_envs
            omega_envs = new_omega_envs
            mask_envs = new_mask_envs
            
            if t_step >= max_steps or all(env.done() for env in envs):
                #print(f"[INFO] Stopping after {t_step} steps")
                #print(f"[DEBUG] Episode rewards: {ep_rewards}")
                #print(f"[DEBUG] Memory length: {[len(m.r_mb) for m in memories]}")
                break
        
        for j in range(configs.num_envs):
            ep_rewards[j] -= envs[j].posRewards

        if all(len(m.r_mb) == 0 for m in memories):
            print(f"[ERROR] Memories are empty, cannot update PPO")
            ipdb.set_trace()

        loss, v_loss = ppo.update(memories, configs.n_j * configs.n_m, configs.graph_pool_type)
        for memory in memories:
            memory.clear_memory()
        
        mean_rewards_all_env = sum(ep_rewards) / len(ep_rewards)
        #log.append([i_update, mean_rewards_all_env])
        log.append([i_update, mean_rewards_all_env, v_loss])
        
        if mean_rewards_all_env < record:
            record = mean_rewards_all_env
            torch.save(ppo.policy.state_dict(), './{}.pth'.format(
                str(configs.n_j) + '_' + str(configs.n_m) + '_' + str(configs.low) + '_' + str(configs.high)))
            #print(f"[INFO] Saved model with reward: {mean_rewards_all_env}")
        
        if (i_update + 1) % 100 == 0:
            with open('./log_{}_{}_{}_{}.txt'.format(
                configs.n_j, configs.n_m, configs.low, configs.high), 'w') as f:
                f.write(str(log))

        print('Episode {}\t Last reward: {:.2f}\t Mean_Vloss: {:.8f}'.format(
            i_update + 1, mean_rewards_all_env, v_loss))
        
        t4 = time.time()

        if (i_update + 1) % 100 == 0:
            dataLoaded = np.load('./DataGen/generatedData_{}_{}_Seed{}.npy'.format(
                configs.n_j, configs.n_m, configs.np_seed_validation), allow_pickle=True)
            vali_data = []
            for i in range(dataLoaded.shape[0]):
                adj, fea = dataLoaded[i][0], dataLoaded[i][1]
                edge_index = adj_to_edge_index(adj, device)
                candidate = np.array([j * configs.n_m for j in range(configs.n_j)])
                mask = np.ones(configs.n_j, dtype=bool)
                vali_data.append((edge_index, fea, candidate, mask))
            
            vali_result = -validate(vali_data, ppo.policy).mean()
            print(f"[INFO] Validation quality: {vali_result}")
            if vali_result < record:
                record = vali_result
                torch.save(ppo.policy.state_dict(), './{}.pth'.format(
                    str(configs.n_j) + '_' + str(configs.n_m) + '_' + str(configs.low) + '_' + str(configs.high)))
            with open('./vali_{}_{}_{}_{}.txt'.format(
                configs.n_j, configs.n_m, configs.low, configs.high), 'w') as f:
                f.write(str(vali_result)) 

        t5 = time.time()
        # print('Training:', t4 - t3)
        # print('Validation:', t5 - t4)

if __name__ == '__main__':
    total1 = time.time()
    main()
    total2 = time.time()