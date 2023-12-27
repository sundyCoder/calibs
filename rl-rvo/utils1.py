import torch
import sys
import logging
import os
from torch.autograd import Variable
from torch.nn import functional as F
import numpy as np
import socket
from config import *

import scipy
import scipy.signal
import time


class Buff():
    def __init__(self, obs_dim, act_dim, size, gamma=0.99, lam=0.95):
        self.obs_buf = [0] * size
        self.act_buf = np.zeros((size, act_dim), dtype=np.float32)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size
        
    def store(self, obs, act, rew, val, logp):
        assert self.ptr < self.max_size     # buffer has to have room so you can store
        self.obs_buf[self.ptr] = obs#.cpu().numpy()
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.ptr += 1
        
    def finish_path(self, last_val=0):

        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)
        
        # the next two lines implement GAE-Lambda advantage calculation
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = self.discount_cumsum(deltas, self.gamma * self.lam)
        
        # the next line computes rewards-to-go, to be targets for the value function
        self.ret_buf[path_slice] = self.discount_cumsum(rews, self.gamma)[:-1]        
        self.path_start_idx = self.ptr
        
    def discount_cumsum(self, x, discount):
        return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]
        
    def get(self):
        # print(self.ptr, self.max_size)
        if self.ptr == self.max_size:    # buffer has to be full before you can get
            self.ptr, self.path_start_idx = 0, 0

            act_ten = torch.as_tensor(self.act_buf, dtype=torch.float32)
            ret_ten = torch.as_tensor(self.ret_buf, dtype=torch.float32)
            adv_ten = torch.as_tensor(self.adv_buf, dtype=torch.float32)
            logp_ten = torch.as_tensor(self.logp_buf, dtype=torch.float32)
            obs_tensor_list = list(map(lambda o: torch.as_tensor(o, dtype=torch.float32), self.obs_buf))

            data = dict(obs=obs_tensor_list, act=act_ten, ret=ret_ten,
                        adv=adv_ten, logp=logp_ten)

            return data
        else:
            return None
    
    def complete(self):
        self.ptr, self.path_start_idx = 0, 0
        
    
def ppo_update(policy, pi_optimizer, v_optimizer, data_list, clip_value=0.2, num_env=4, max_update_num=10, use_gpu=True):
               
    randn = np.arange(num_env)
    np.random.shuffle(randn)
    
    update_num = 0
    for r in randn:
        data = data_list[r]
        update_num += 1
        
        if update_num > max_update_num:
            continue
            
        for i in range(TRAIN_A_ITERS):
            pi_optimizer.zero_grad()
            act, adv, logp_old = data['act'], data['adv'], data['logp']
            obs = data['obs']
            # print(len(obs), type(obs), obs[0].size())
            # print(act.size())
            # print(adv.size())
            # print(logp_old.size())
            logp_old = logp_old.cuda()
            adv = adv.cuda()
            pi, logp = policy.pi(obs, act)
            ratio = torch.exp(logp - logp_old)
            clip_adv = torch.clamp(ratio, 1-clip_value, 1+clip_value) * adv
            policy_loss = -(torch.min(ratio * adv, clip_adv)).mean()

            Kl = (logp_old - logp).mean().item()
            
            ent = pi.entropy().mean().item()
            clipped = ratio.gt(1+clip_value) | ratio.lt(1-clip_value)
            clipfrac = torch.as_tensor(clipped, dtype=torch.float32).mean().item()
            
            if Kl > TARGET_KL:
                break
            policy_loss.backward()
            pi_optimizer.step()
        for l in range(TRAIN_V_ITERS):
            obs, ret = data['obs'], data['ret']
            ret = ret.cuda()
            value_loss = ((policy.v(obs) - ret)**2).mean()
            v_optimizer.zero_grad()
            value_loss.backward()
            v_optimizer.step()
    print("update------------")
    
def generate_action(env, state_list, policy):
    if env.index == 0:       
        a, v, loga = policy(state_list)              
    else:
        v = None
        a = None
        loga = None
    return a, v, loga
    
def generate_action_test(env, state_list, policy):
    if env.index == 0:       
        a = policy.get_actions(state_list)              
    else:
        a = None
    return a

