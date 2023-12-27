import torch
import sys
import logging
import os
from torch.autograd import Variable
from torch.nn import functional as F
import numpy as np
import socket
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler



def transform_buffer(buff):
    s_batch, goal_batch, obs_batch, a_batch, r_batch, d_batch, l_batch, v_batch = [], [], [], [], [], [], [], []
    s_temp, goal_temp, obs_temp = [], [], []
    
    for e in buff:
        for state in e[0]:
            s_temp.append(state[0])
            goal_temp.append(state[1])
            obs_temp.append(state[2])
        s_batch.append(s_temp)
        goal_batch.append(goal_temp)
        obs_batch.append(obs_temp)
        s_temp = []
        goal_temp = []
        obs_temp = []
        
        
        a_batch.append(e[1])
        r_batch.append(e[2])
        d_batch.append(e[3])
        l_batch.append(e[4])
        v_batch.append(e[5])

    s_batch = np.asarray(s_batch)
    goal_batch = np.asarray(goal_batch)
    obs_batch = np.asarray(obs_batch)
    a_batch = np.asarray(a_batch)
    r_batch = np.asarray(r_batch)
    d_batch = np.asarray(d_batch)
    l_batch = np.asarray(l_batch)
    v_batch = np.asarray(v_batch)

    return s_batch, goal_batch, obs_batch, a_batch, r_batch, d_batch, l_batch, v_batch

def generate_action(env, state_list, policy, action_bound):
    if env.index == 0:
        s_list, goal_list, obs_list = [], [], []
        for i in state_list:
            s_list.append(i[0])
            goal_list.append(i[1])
            obs_list.append(i[2])
        s_list = np.asarray(s_list)
        goal_list = np.asarray(goal_list)
        obs_list = np.asarray(obs_list)
        s_list = Variable(torch.from_numpy(s_list)).float().cuda()
        goal_list = Variable(torch.from_numpy(goal_list)).float().cuda()
        obs_list = Variable(torch.from_numpy(obs_list)).float().cuda()
        
        v, a, logprob, mean = policy(s_list, goal_list, obs_list)
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
        s_list, goal_list, speed_list = [], [], []
        for i in state_list:
            s_list.append(i[0])
            goal_list.append(i[1])
            speed_list.append(i[2])

        s_list = np.asarray(s_list)
        goal_list = np.asarray(goal_list)
        speed_list = np.asarray(speed_list)

        s_list = Variable(torch.from_numpy(s_list)).float().cuda()
        goal_list = Variable(torch.from_numpy(goal_list)).float().cuda()
        speed_list = Variable(torch.from_numpy(speed_list)).float().cuda()

        _, _, _, mean = policy(s_list, goal_list, speed_list)
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
    
    
def ppo_update(policy, optimizer, batch_size, memory, epoch,
               coeff_entropy=0.02, clip_value=0.2,
               num_step=128, num_env=4, act_size=2, T=5, robot_state_num=6, obs_state_num=6):
    states, goal, obs, actions, logprobs, targets, values, rewards, advs = memory

    advs = (advs - advs.mean()) / advs.std()
    #important
    states = states.reshape((num_step*num_env, robot_state_num, T, 1))
    goal = goal.reshape((num_step*num_env, 2))
    obs = obs.reshape((num_step*num_env, obs_state_num, T, num_env-1))
    
    actions = actions.reshape(num_step*num_env, act_size)
    logprobs = logprobs.reshape(num_step*num_env, 1)
    advs = advs.reshape(num_step*num_env, 1)
    targets = targets.reshape(num_step*num_env, 1)

    for update in range(epoch): # epoch=2
        sampler = BatchSampler(SubsetRandomSampler(list(range(advs.shape[0]))), batch_size=batch_size, drop_last=False) # batch_size=32
        for i, index in enumerate(sampler):
            sampled_states = Variable(torch.from_numpy(states[index])).float().cuda()
            sampled_goal = Variable(torch.from_numpy(goal[index])).float().cuda()
            sampled_obs = Variable(torch.from_numpy(obs[index])).float().cuda()

            sampled_actions = Variable(torch.from_numpy(actions[index])).float().cuda()
            sampled_logprobs = Variable(torch.from_numpy(logprobs[index])).float().cuda()
            sampled_targets = Variable(torch.from_numpy(targets[index])).float().cuda()
            sampled_advs = Variable(torch.from_numpy(advs[index])).float().cuda()


            new_value, new_logprob, dist_entropy = policy.evaluate_actions(sampled_states, sampled_goal, sampled_obs, sampled_actions)

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
