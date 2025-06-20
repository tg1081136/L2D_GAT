from mb_agg import *
from agent_utils import *
import torch
import argparse
from Params import configs
import time
import numpy as np

device = torch.device(configs.device)

parser = argparse.ArgumentParser(description='Arguments for ppo_jssp')
parser.add_argument('--Pn_j', type=int, default=15, help='Number of jobs of instances to test')
parser.add_argument('--Pn_m', type=int, default=15, help='Number of machines instances to test')
parser.add_argument('--Nn_j', type=int, default=15, help='Number of jobs on which to be loaded net are trained')
parser.add_argument('--Nn_m', type=int, default=15, help='Number of machines on which to be loaded net are trained')
parser.add_argument('--low', type=int, default=1, help='LB of duration')
parser.add_argument('--high', type=int, default=99, help='UB of duration')
parser.add_argument('--seed', type=int, default=200, help='Seed for validate set generation')
params = parser.parse_args()

N_JOBS_P = params.Pn_j
N_MACHINES_P = params.Pn_m
LOW = params.low
HIGH = params.high
SEED = params.seed
N_JOBS_N = params.Nn_j
N_MACHINES_N = params.Nn_m

from JSSP_Env import SJSSP
from PPO_jssp_multiInstances import PPO
env = SJSSP(n_j=N_JOBS_P, n_m=N_MACHINES_P)

ppo = PPO(configs.lr, configs.gamma, configs.k_epochs, configs.eps_clip,
          n_j=N_JOBS_P,
          n_m=N_MACHINES_P,
          num_layers=configs.num_layers,
          neighbor_pooling_type=configs.neighbor_pooling_type,
          input_dim=configs.input_dim,
          hidden_dim=configs.hidden_dim,
          num_mlp_layers_feature_extract=configs.num_mlp_layers_feature_extract,
          num_mlp_layers_actor=configs.num_mlp_layers_actor,
          hidden_dim_actor=configs.hidden_dim_actor,
          num_mlp_layers_critic=configs.num_mlp_layers_critic,
          hidden_dim_critic=configs.hidden_dim_critic)
path = './SavedNetwork/{}.pth'.format(str(N_JOBS_N) + '_' + str(N_MACHINES_N) + '_' + str(LOW) + '_' + str(HIGH))
ppo.policy.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
g_pool_step = g_pool_cal(graph_pool_type=configs.graph_pool_type,
                         batch_size=torch.Size([1, env.number_of_tasks, env.number_of_tasks]),
                         n_nodes=env.number_of_tasks,
                         device=device)

from uniform_instance_gen import uni_instance_gen
np.random.seed(SEED)

dataLoaded = np.load('./DataGen/generatedData' + str(N_JOBS_P) + '_' + str(N_MACHINES_P) + '_Seed' + str(SEED) + '.npy')
dataset = []
for i in range(dataLoaded.shape[0]):
    dataset.append((dataLoaded[i][0], dataLoaded[i][1]))

def test(dataset):
    result = []
    t1 = time.time()
    for i, data in enumerate(dataset):
        # gymnasium 的 reset 返回 (state, info)
        state, info = env.reset(data)
        adj, fea, candidate, mask = state  # candidate 是 omega
        ep_reward = -env.max_endTime
        while True:
            fea_tensor = torch.from_numpy(fea).to(device)
            adj_tensor = torch.from_numpy(adj).to(device)
            candidate_tensor = torch.from_numpy(candidate).to(device)
            #print("Final candidate type before conversion:", type(candidate), candidate)
            mask_tensor = torch.from_numpy(mask).to(device)

            with torch.no_grad():
                pi, _ = ppo.policy(x=fea_tensor,
                                   graph_pool=g_pool_step,
                                   padded_nei=None,
                                   adj=adj_tensor,
                                   candidate=candidate_tensor.unsqueeze(0),
                                   mask=mask_tensor.unsqueeze(0))
                action = greedy_select_action(pi, candidate)

            # 正確解包 step 的返回值
            adj, fea, reward, done, truncated, info = env.step(action)
            # 從 info 中更新 candidate 和 mask
            candidate = info["omega"]
            mask = info["mask"]
            ep_reward += reward

            if done:
                break

        print('Instance' + str(i + 1) + ' makespan:', -ep_reward + env.posRewards)
        result.append(-ep_reward + env.posRewards)

    t2 = time.time()
    print(t2 - t1)
    file_writing_obj = open('./' + 'drltime_' + str(N_JOBS_N) + 'x' + str(N_MACHINES_N) + '_' + str(N_JOBS_P) + 'x' + str(N_MACHINES_P) + '.txt', 'w')
    file_writing_obj.write(str((t2 - t1)/len(dataset)))

    np.save('drlResult_' + str(N_JOBS_N) + 'x' + str(N_MACHINES_N) + '_' + str(N_JOBS_P) + 'x' + str(N_MACHINES_P) + '_Seed' + str(SEED), np.array(result, dtype=np.single))

if __name__ == '__main__':
    import cProfile
    cProfile.run('test(dataset)', filename='restats')