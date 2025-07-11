from torch.distributions.categorical import Categorical
import ipdb
import torch

def select_action(pi, mask, memory):
    if pi.dim() == 0:
        pi = pi.unsqueeze(0)

    valid_indices = torch.where(mask)[0]

    if len(valid_indices) == 0:
        raise ValueError("No valid jobs in mask")
    elif len(valid_indices) == 1:
        job_id = valid_indices[0]
        logprob = torch.log(pi[job_id] + 1e-10)
    else:
        dist = torch.distributions.Categorical(probs=pi)
        job_id = dist.sample()
        logprob = dist.log_prob(job_id)

    memory.logprobs.append(logprob)

    return job_id.item(), job_id

def eval_actions(probs, action):
    #print(f"[DEBUG] eval_actions: probs: {probs}, action: {action}")
    valid_indices = torch.where(probs > 0)[0]
    if len(valid_indices) == 1:
        valid_action = valid_indices[0]
        if action != valid_action:
            #print(f"[WARNING] Invalid action {action}, expected {valid_action}, setting logp=0, entropy=0")
            return torch.tensor(0.0, device=probs.device), torch.tensor(0.0, device=probs.device)
    try:
        softmax_dist = torch.distributions.Categorical(probs=probs)
        logp = softmax_dist.log_prob(action).reshape(-1)
        entropy = softmax_dist.entropy().reshape(-1)
        return logp, entropy
    except ValueError as e:
        #print(f"[ERROR] Categorical failed: {e}")
        #print(f"[DEBUG] probs: {probs}, action: {action}, valid_indices: {valid_indices}")
        ipdb.set_trace()
        raise
'''
def select_action(p, cadidate, memory):
    dist = Categorical(p.squeeze())
    s = dist.sample()
    if memory is not None: memory.logprobs.append(dist.log_prob(s))
    return cadidate[s], s

# evaluate the actions
def eval_actions(p, actions):
    softmax_dist = Categorical(p)
    ret = softmax_dist.log_prob(actions).reshape(-1)
    entropy = softmax_dist.entropy().mean()
    return ret, entropy
'''

# select action method for test
def greedy_select_action(p, candidate):
    _, index = p.squeeze().max(0)
    action = candidate[index]
    return action


# select action method for test
def sample_select_action(p, candidate):
    dist = Categorical(p.squeeze())
    s = dist.sample()
    return candidate[s]


def select_action_dy(pi, candidate, mask, memory):
    if pi.dim() == 0:
        pi = pi.unsqueeze(0)
    #print(f"[DEBUG] select_action: pi={pi}, candidate={candidate}, mask={mask}")
    valid_indices = torch.where(mask)[0]
    if len(valid_indices) == 0:
        #print("[ERROR] No valid candidates in mask, entering ipdb")
        ipdb.set_trace()
        raise ValueError("No valid candidates in mask")
    elif len(valid_indices) == 1:
        action_idx = valid_indices[0]
        action = candidate[action_idx]
        logprob = torch.log(pi[action_idx] + 1e-10)
        #print(f"[DEBUG] Single valid action: action_idx={action_idx}, action={action}")
    else:
        dist = torch.distributions.Categorical(probs=pi)
        action_idx = dist.sample()
        action = candidate[action_idx]
        logprob = dist.log_prob(action_idx)
        #print(f"[DEBUG] Sampled action: action_idx={action_idx}, action={action}")
    
    memory.logprobs.append(logprob)
    return action, action_idx
