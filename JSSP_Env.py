import gymnasium as gym
import numpy as np
import torch
from gymnasium.utils import EzPickle
from uniform_instance_gen import override
from updateEntTimeLB import calEndTimeLB
from Params import configs
from permissibleLS import permissibleLeftShift
from updateAdjMat import getActionNbghs
from torch_geometric.utils import from_scipy_sparse_matrix
import scipy.sparse as sp
import ipdb

class SJSSP(gym.Env, EzPickle):
    def __init__(self, n_j, n_m, device):
        EzPickle.__init__(self)
        self.step_count = 0
        self.number_of_jobs = n_j
        self.number_of_machines = n_m
        self.number_of_tasks = self.number_of_jobs * self.number_of_machines
        # the task id for first column
        self.first_col = np.arange(start=0, stop=self.number_of_tasks, step=1).reshape(self.number_of_jobs, -1)[:, 0]
        # the task id for last column
        self.last_col = np.arange(start=0, stop=self.number_of_tasks, step=1).reshape(self.number_of_jobs, -1)[:, -1]
        self.getEndTimeLB = calEndTimeLB
        self.getNghbs = getActionNbghs
        self.device = device

    def done(self):
        if len(self.partial_sol_sequeence) == self.number_of_tasks:
            return True
        return False
    
    @override
    def step(self, action):
        if action not in self.partial_sol_sequeence:
            # UPDATE BASIC INFO:
            row = action // self.number_of_machines
            col = action % self.number_of_machines
            self.step_count += 1
            self.finished_mark[row, col] = 1
            dur_a = self.dur[row, col]
            self.partial_sol_sequeence.append(action)

            # UPDATE STATE:
            # permissible left shift
            startTime_a, flag = permissibleLeftShift(a=action, durMat=self.dur, mchMat=self.m, mchsStartTimes=self.mchsStartTimes, opIDsOnMchs=self.opIDsOnMchs)
            self.flags.append(flag)

            # omega 和 mask 更新
            job_id = action // self.number_of_machines

            if action not in self.last_col:
                # 不是最後一個任務，更新 omega 到下一個任務
                self.omega[job_id] += 1
            else:
                # 是最後一個任務，將該工作標記為完成
                self.mask[job_id] = False  # 或者根據你的mask含義決定是True還是False

            #print(f"[DEBUG] job_id={job_id}, action={action} in last_col={action in self.last_col}, omega={self.omega}, mask={self.mask}")

            self.temp1[row, col] = startTime_a + dur_a
            self.LBs = calEndTimeLB(self.temp1, self.dur_cp)

            precd, succd = self.getNghbs(action, self.opIDsOnMchs)

            # adj matrix
            self.adj[action] = 0
            self.adj[action, action] = 1
            if action not in self.first_col:
                self.adj[action, action - 1] = 1
            self.adj[action, precd] = 1
            self.adj[succd, action] = 1

            if flag and precd != action and succd != action:
                self.adj[succd, precd] = 0
            
            adj_sparse = sp.csr_matrix(self.adj)
            self.edge_index = from_scipy_sparse_matrix(adj_sparse)[0].to(dtype=torch.long, device=self.device).clone().detach()

        # prepare for return
        fea = np.concatenate((self.LBs.reshape(-1, 1) / configs.et_normalize_coef,
                              self.finished_mark.reshape(-1, 1)), axis=1)
        reward = -(self.LBs.max() - self.max_endTime)
        if reward == 0:
            reward = configs.rewardscale
            self.posRewards += reward
        self.max_endTime = self.LBs.max()

        #print(f"[DEBUG] step: action={action}, candidate={self.omega}, mask={self.mask}, reward={reward}, done={self.done()}")
        
        # 檢查是否還有有效的候選者
        if not self.done() and not np.any(self.mask):
            print(f"[ERROR] No valid candidates in step mask, candidate={self.omega}, mask={self.mask}")
            ipdb.set_trace()

        return self.edge_index, fea, reward, self.done(), {"omega": self.omega, "mask": self.mask}

    @override
    def reset(self, data):
        self.step_count = 0
        self.edge_index, fea, self.omega, self.mask = data

        # Extract dur and m from fea
        if isinstance(fea, torch.Tensor):
            fea = fea.cpu().numpy()
        self.dur = fea[:, 0] * configs.et_normalize_coef
        if np.isnan(self.dur).any():
            print("[ERROR] NaN detected in dur")
            ipdb.set_trace()

        self.dur = self.dur.reshape(self.number_of_jobs, self.number_of_machines).astype(np.float32)
        self.dur_cp = np.copy(self.dur)
        self.m = fea[:, 1] * configs.wkr_normalize_coef
        if np.isnan(self.m).any():
            print("[ERROR] NaN detected in m")
            ipdb.set_trace()

        self.m = self.m.reshape(self.number_of_jobs, self.number_of_machines).astype(np.int32)
        self.partial_sol_sequeence = []
        self.flags = []
        self.posRewards = 0

        # 確保 omega 和 mask 的正確初始化
        self.omega = self.omega.astype(np.int64)
        self.mask = self.mask.astype(bool)
        
        # 檢查初始狀態的一致性
        if len(self.omega) != self.number_of_jobs:
            print(f"[ERROR] omega length mismatch: {len(self.omega)} vs {self.number_of_jobs}")
            ipdb.set_trace()
        if len(self.mask) != self.number_of_jobs:
            print(f"[ERROR] mask length mismatch: {len(self.mask)} vs {self.number_of_jobs}")
            ipdb.set_trace()
        
        # 檢查初始狀態：所有工作都應該可用
        if not np.all(self.mask):
            print(f"[WARNING] Not all jobs are available in initial state: {self.mask}")
        
        # Initialize adjacency matrix and sync with edge_index
        conj_nei_up_stream = np.eye(self.number_of_tasks, k=-1, dtype=np.float32)
        conj_nei_low_stream = np.eye(self.number_of_tasks, k=1, dtype=np.float32)
        conj_nei_up_stream[self.first_col] = 0
        conj_nei_low_stream[self.last_col] = 0
        self_as_nei = np.eye(self.number_of_tasks, dtype=np.float32)
        self.adj = self_as_nei + conj_nei_up_stream
        adj_sparse = sp.csr_matrix(self.adj)
        self.edge_index = from_scipy_sparse_matrix(adj_sparse)[0].to(dtype=torch.long, device=self.device).clone().detach()
        
        self.LBs = np.cumsum(self.dur, axis=1, dtype=np.float32)
        self.initQuality = self.LBs.max() if not configs.init_quality_flag else 0
        self.max_endTime = self.initQuality
        self.finished_mark = np.zeros_like(self.m, dtype=np.float32)
        self.temp1 = np.zeros_like(self.dur, dtype=np.float32)
        self.mchsStartTimes = -configs.high * np.ones_like(self.dur.transpose(), dtype=np.int32)
        self.opIDsOnMchs = -self.number_of_jobs * np.ones_like(self.dur.transpose(), dtype=np.int32)
        
        fea = np.concatenate((self.LBs.reshape(-1, 1) / configs.et_normalize_coef,
                              self.finished_mark.reshape(-1, 1)), axis=1)
        
        #print(f"[DEBUG] reset: candidate={self.omega}, mask={self.mask}, done={self.done()}")
        
        if self.done():
            print("[ERROR] Environment done after reset")
            ipdb.set_trace()
        if not np.any(self.mask):
            print(f"[ERROR] No valid candidates in reset mask, candidate={self.omega}, mask={self.mask}")
            ipdb.set_trace()
        if len(self.mask) != len(self.omega):
            print(f"[ERROR] Mismatch: mask length={len(self.mask)}, candidate length={len(self.omega)}")
            ipdb.set_trace()
        
        return (self.edge_index, fea, self.omega, self.mask), {}