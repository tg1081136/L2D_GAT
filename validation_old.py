import numpy as np
import torch
from Params import configs
from JSSP_Env import SJSSP
from mb_agg import g_pool_cal
from agent_utils import greedy_select_action
import ipdb

def to_tensor(x, dtype=None, device=None, sparse=False):
    if isinstance(x, np.ndarray):
        t = torch.from_numpy(x)
    elif isinstance(x, torch.Tensor):
        t = x.cpu() if not x.device.type == 'cpu' else x  # 確保能轉 dtype 後轉 device
    else:
        raise TypeError(f"Unsupported type: {type(x)}")
    if dtype:
        t = t.to(dtype)
    t = t.to(device)
    return t.to_sparse() if sparse else t

def validate(vali_set, model):
    N_JOBS = configs.n_j
    N_MACHINES = configs.n_m
    device = torch.device(configs.device)
    
    env = SJSSP(n_j=N_JOBS, n_m=N_MACHINES, device=device)
    # 修正：將 batch_size 設為 None，與主程式一致
    g_pool_step = g_pool_cal(graph_pool_type=configs.graph_pool_type,
                             batch_size=None,
                             n_nodes=env.number_of_tasks,
                             device=device)
    make_spans = []
    
    for data in vali_set:
        state, info = env.reset(data)
        adj, fea, candidate, mask = state
        rewards = -env.initQuality
        print(f"[DEBUG] validate reset: candidate={candidate}, mask={mask}, done={env.done()}")
        if env.done():
            print(f"[ERROR] Environment done after reset")
            ipdb.set_trace()
        if not np.any(mask):
            print(f"[ERROR] No valid candidates in reset mask, candidate={candidate}, mask={mask}")
            ipdb.set_trace()
        if len(mask) != len(candidate):
            print(f"[ERROR] Mismatch: mask length={len(mask)}, candidate length={len(candidate)}")
            ipdb.set_trace()
        
        while True:
            adj_tensor = to_tensor(adj, dtype=torch.long, device=device)
            fea_tensor = to_tensor(fea, dtype=torch.float32, device=device)
            candidate_tensor = to_tensor(candidate, dtype=torch.long, device=device)
            mask_tensor = to_tensor(mask, dtype=torch.bool, device=device)
            print(f"[DEBUG] validate step: adj_tensor.shape={adj_tensor.shape}, fea_tensor.shape={fea_tensor.shape}, "
                  f"candidate_tensor.shape={candidate_tensor.shape}, mask_tensor.shape={mask_tensor.shape}")
            
            with torch.no_grad():
                pi, _ = model(
                    x=fea_tensor,
                    edge_index=adj_tensor,
                    graph_pool=g_pool_step,
                    candidate=candidate_tensor.unsqueeze(0),
                    mask=mask_tensor.unsqueeze(0)
                )
                print(f"[DEBUG] validate step: pi.shape={pi.shape}, pi={pi}")
            
            action = greedy_select_action(pi, candidate)
            print(f"[DEBUG] validate step: action={action}, candidate={candidate}, mask={mask}")
            
            adj, fea, reward, done, info = env.step(action.item())
            candidate = info["omega"]
            mask = info["mask"]
            rewards += reward
            print(f"[DEBUG] validate step: reward={reward}, done={done}, candidate={candidate}, mask={mask}")

            if env.done() or not np.any(mask) or len(mask) != len(candidate):
                print(f"[WARNING] Skipping invalid validation sample")
                continue  # 跳過這個 data
            '''
            if not done and not np.any(mask):
                print(f"[ERROR] No valid candidates in step mask, candidate={candidate}, mask={mask}")
                ipdb.set_trace()
            if len(mask) != len(candidate):
                print(f"[ERROR] Mismatch: mask length={len(mask)}, candidate length={len(candidate)}")
                ipdb.set_trace()
            
            if done:
                break
                
            '''
            make_spans.append(rewards - env.posRewards)
    
    return np.array(make_spans)

if __name__ == '__main__':
    from uniform_instance_gen import uni_instance_gen
    import numpy as np
    import time
    import argparse
    from Params import configs
    from PPO_jssp_multiInstances import PPO

    parser = argparse.ArgumentParser(description='Arguments for ppo_jssp')
    parser.add_argument('--Pn_j', type=int, default=2, help='Number of jobs of instances to test')
    parser.add_argument('--Pn_m', type=int, default=3, help='Number of machines instances to test')
    parser.add_argument('--Nn_j', type=int, default=2, help='Number of jobs on which to be loaded net are trained')
    parser.add_argument('--Nn_m', type=int, default=3, help='Number of machines on which to be loaded net are trained')
    parser.add_argument('--low', type=int, default=1, help='LB of duration')
    parser.add_argument('--high', type=int, default=99, help='UB of duration')
    parser.add_argument('--seed', type=int, default=200, help='Cap seed for validate set generation')
    parser.add_argument('--n_vali', type=int, default=100, help='validation set size')
    params = parser.parse_args()

    N_JOBS_P = params.Pn_j
    N_MACHINES_P = params.Pn_m
    LOW = params.low
    HIGH = params.high
    N_JOBS_N = params.Nn_j
    N_MACHINES_N = params.Nn_m

    # 修正：使用與 PPO_jssp_multiInstances.py 一致的參數
    ppo = PPO(
        lr=configs.lr,
        gamma=configs.gamma,
        k_epochs=configs.k_epochs,
        eps_clip=configs.eps_clip,
        n_j=N_JOBS_P,
        n_m=N_MACHINES_P,
        node_input_dim=configs.input_dim,
        graph_hidden_dim=configs.hidden_dim,
        graph_output_dim=configs.output_dim,
        mlp_hidden_dim=configs.mlp_hidden_dim,
        mlp_output_dim=configs.mlp_output_dim,
        num_layers=configs.num_layers,
        heads=configs.heads
    )

    path = './{}.pth'.format(str(N_JOBS_N) + '_' + str(N_MACHINES_N) + '_' + str(LOW) + '_' + str(HIGH))
    ppo.policy.load_state_dict(torch.load(path))

    SEEDs = range(0, params.seed, 10)
    result = []
    for SEED in SEEDs:
        np.random.seed(SEED)
        vali_data = [uni_instance_gen(n_j=N_JOBS_P, n_m=N_MACHINES_P, low=LOW, high=HIGH) for _ in range(params.n_vali)]
        makespan = -validate(vali_data, ppo.policy)
        print(f"[INFO] Seed {SEED}: makespan mean = {makespan.mean()}")
        result.append(makespan.mean())
    
    print(f"[INFO] Average makespan across seeds: {np.mean(result)}")