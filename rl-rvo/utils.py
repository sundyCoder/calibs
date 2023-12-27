import torch
import sys
import logging
import os
from torch.autograd import Variable
from torch.nn import functional as F
import numpy as np
import socket
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from config import *



def transform_buffer(buff):    #   [([tensor, tensor, ...], [action, action], [],), (), ()]
    obs_batch, a_batch, r_batch, d_batch, l_batch, v_batch = [], [], [], [], [], []
    
    for e in buff:
        obs_batch.append(e[0])                #  [[tensor, ...], [], []]     400 x 8  n_tensor
        a_batch.append(e[1])
        r_batch.append(e[2])                # 400 x 8  1
        d_batch.append(e[3])
        l_batch.append(e[4])
        v_batch.append(e[5])

    a_batch = np.asarray(a_batch)                  # 400 x 8  2
    r_batch = np.asarray(r_batch)
    d_batch = np.asarray(d_batch)
    l_batch = np.asarray(l_batch)
    v_batch = np.asarray(v_batch)

    return obs_batch, a_batch, r_batch, d_batch, l_batch, v_batch

def generate_action(env, state_list, policy, action_bound):
    if env.index == 0:       
        v, a, logprob, mean = policy(state_list)
        v, a, logprob = v.data.cpu().numpy(), a.data.cpu().numpy(), logprob.data.cpu().numpy()
        scaled_action = np.clip(a, a_min=action_bound[0], a_max=action_bound[1])        
        
    else:
        v = None
        a = None
        scaled_action = None
        logprob = None

    return v, a, logprob, scaled_action

def generate_action_test(env, state_list, policy, action_bound):
    if env.index == 0:
        _, _, _, mean = policy(state_list)
        mean = mean.data.cpu().numpy()
        scaled_action = np.clip(mean, a_min=action_bound[0], a_max=action_bound[1])
    else:
        mean = None
        scaled_action = None

    return mean, scaled_action


def generate_train_data(rewards, gamma, values, last_value, dones, lam):
    num_step = rewards.shape[0]
    num_env = rewards.shape[1]
    values = list(values)
    values.append(last_value)
    values = np.asarray(values).reshape((num_step+1,num_env))

    targets = np.zeros((num_step, num_env))
    gae = np.zeros((num_env,))

    for t in range(num_step - 1, -1, -1):
        delta = rewards[t, :] + gamma * values[t + 1, :] * (1 - dones[t, :]) - values[t, :]
        gae = delta + gamma * lam * (1 - dones[t, :]) * gae

        targets[t, :] = gae + values[t, :]

    advs = targets - values[:-1, :]
    return targets, advs
    
    
def ppo_update_v1(policy, optimizer, batch_size, memory, epoch, coeff_entropy=0.02, clip_value=0.2,
               num_step=128, num_env=4, act_size=2):
    obs, actions, logprobs, targets, values, rewards, advs = memory

    advs = (advs - advs.mean()) / advs.std()
    #important
    # obs = [[tensor, tensor, tensor, ...], [], []]         num_step * num_env,  tensor
    actions = actions.reshape(num_step*num_env, act_size)
    logprobs = logprobs.reshape(num_step*num_env, 1)
    advs = advs.reshape(num_step*num_env, 1)
    targets = targets.reshape(num_step*num_env, 1)

    for update in range(epoch): # epoch= update_num  / n / (horizion / batch)
        sampler = BatchSampler(SubsetRandomSampler(list(range(advs.shape[0]))), batch_size=batch_size, drop_last=False) # batch_size=32
        for i, index in enumerate(sampler):
            #   row, col =  item // num_env,    item % num_env
            sampled_obs = []
            for item in index:
                row, col = item // num_env, item % num_env
                sampled_obs.append(obs[row][col])
            sampled_actions = Variable(torch.from_numpy(actions[index])).float().cuda()
            sampled_logprobs = Variable(torch.from_numpy(logprobs[index])).float().cuda()
            sampled_targets = Variable(torch.from_numpy(targets[index])).float().cuda()
            sampled_advs = Variable(torch.from_numpy(advs[index])).float().cuda()
            
            new_value, new_logprob, dist_entropy = policy.evaluate_actions(sampled_obs, sampled_actions)
            
            sampled_logprobs = sampled_logprobs.view(-1, 1)
            ratio = torch.exp(new_logprob - sampled_logprobs) # important weight

            sampled_advs = sampled_advs.view(-1, 1)
            surrogate1 = ratio * sampled_advs
            surrogate2 = torch.clamp(ratio, 1 - clip_value, 1 + clip_value) * sampled_advs
            policy_loss = -torch.min(surrogate1, surrogate2).mean()

            sampled_targets = sampled_targets.view(-1, 1)
            value_loss = F.mse_loss(new_value, sampled_targets)

            loss = policy_loss + 20 * value_loss - coeff_entropy * dist_entropy
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            info_p_loss, info_v_loss, info_entropy = float(policy_loss.detach().cpu().numpy()), float(value_loss.detach().cpu().numpy()), float(dist_entropy.detach().cpu().numpy())

            
            
    print('update')
    
    
def ppo_update(policy, optimizer, batch_size, memory, epoch, coeff_entropy=0.02, clip_value=0.2,
               num_step=128, num_env=4, act_size=2):
    obs, actions, logprobs, targets, values, rewards, advs = memory

    advs = (advs - advs.mean()) / advs.std()
        
    for i in range(num_env):
        sample_obs, sample_actions, sample_logprobs, sample_targets, sample_advs = [], [], [], [], []
        for j in range(num_step):
            sample_obs.append(obs[j][i])
            sample_actions.append(actions[j][i])
            sample_logprobs.append(logprobs[j][i])
            sample_targets.append(targets[j][i])
            sample_advs.append(advs[j][i])
        sample_actions = np.asarray(sample_actions)
        sample_logprobs = np.asarray(sample_logprobs)
        sample_targets = np.asarray(sample_targets)
        sample_advs = np.asarray(sample_advs)
        
        sample_actions = sample_actions.reshape(num_step, act_size)
        sample_logprobs = sample_logprobs.reshape(num_step, 1)
        sample_targets = sample_targets.reshape(num_step, 1)
        sample_advs = sample_advs.reshape(num_step, 1)

        sampled_actions = Variable(torch.from_numpy(sample_actions)).float().cuda()
        sampled_logprobs = Variable(torch.from_numpy(sample_logprobs)).float().cuda()
        sampled_targets = Variable(torch.from_numpy(sample_targets)).float().cuda()
        sampled_advs = Variable(torch.from_numpy(sample_advs)).float().cuda()
    
        for k in range(TRAIN_ITERS):
            new_value, new_logprob, dist_entropy = policy.evaluate_actions(sample_obs, sampled_actions)
            sampled_logprobs = sampled_logprobs.view(-1, 1)
            ratio = torch.exp(new_logprob - sampled_logprobs) # important weight
            
            sampled_advs = sampled_advs.view(-1, 1)
            surrogate1 = ratio * sampled_advs
            surrogate2 = torch.clamp(ratio, 1 - clip_value, 1 + clip_value) * sampled_advs
            policy_loss = -torch.min(surrogate1, surrogate2).mean()

            sampled_targets = sampled_targets.view(-1, 1)
            value_loss = F.mse_loss(new_value, sampled_targets)

            loss = policy_loss + 20 * value_loss - coeff_entropy * dist_entropy
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            info_p_loss, info_v_loss, info_entropy = float(policy_loss.detach().cpu().numpy()), float(value_loss.detach().cpu().numpy()), float(dist_entropy.detach().cpu().numpy())
    
        """
        for k in range(TRAIN_ITERS):
            new_value, new_logprob, dist_entropy = policy.evaluate_actions(sampled_obs, sampled_actions)
            ratio = torch.exp(new_logprob - sampled_logprobs)
            surrogate1 = ratio * sampled_advs
            surrogate2 = torch.clamp(ratio, 1 - clip_value, 1 + clip_value) * sampled_advs
            policy_loss = -torch.min(surrogate1, surrogate2).mean()
            Kl = (sampled_logprobs - new_logprob).mean().item()
            if Kl > TARGET_KL:
                break
            pi_optimizer.zero_grad()
            policy_loss.backward()
            pi_optimizer.step()
        for l in range(TRAIN_ITERS):
            new_value, new_logprob, dist_entropy = policy.evaluate_actions(sampled_obs, sampled_actions)
            value_loss = F.mse_loss(new_value, sampled_targets)
            v_optimizer.zero_grad()
            value_loss.backward()
            v_optimizer.step()
        """
    
    
    print('update')

