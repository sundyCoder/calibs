import torch
import torch.nn as nn
import numpy as np
from torch.distributions.normal import Normal
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
import time

def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []

    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]

    return nn.Sequential(*layers)

class RNN_AC(nn.Module):

    def __init__(self, action_dim=2, state_dim=6, rnn_input_dim=8, 
    rnn_hidden_dim=256, hidden_sizes_ac=(256, 256), hidden_sizes_v=(256, 256), 
    activation=nn.ReLU, output_activation=nn.Tanh, output_activation_v= nn.Identity, use_gpu=True):
        super(RNN_AC, self).__init__()
        self.use_gpu = use_gpu
        obs_dim = (rnn_hidden_dim + state_dim)
        
        self.state_dim = state_dim
        self.input_dim = rnn_input_dim
        self.hidden_dim = rnn_hidden_dim

        self.rnn_net = nn.GRU(self.input_dim, self.hidden_dim, batch_first=True, bidirectional=True)
        des_dim = state_dim + self.hidden_dim
        self.ln = nn.LayerNorm(des_dim)
        
        self.net_out = mlp([obs_dim] + list(hidden_sizes_ac) + [action_dim], activation, output_activation)
        log_std = -1 * np.ones(action_dim, dtype=np.float32)
        self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))
        
        self.v_net = mlp([obs_dim] + list(hidden_sizes_v) + [1], activation, output_activation_v)
        
    def forward(self, obs):
        if isinstance(obs, list):
            obs = self.obs_rnn_list(obs)
        else:
            obs = self.obs_rnn(obs)
        mean = self.net_out(obs)
        std = torch.exp(self.log_std)
        pi = Normal(mean, std)
        action = pi.sample() 
        # print(pi.log_prob(action).sum(-1))
        logprob = pi.log_prob(action).sum(axis=-1)
        # v = torch.squeeze(self.v_net(obs), -1)
        v = self.v_net(obs)     
        return v, action, logprob, mean
    
    def evaluate_actions(self, obs, action):
        v, _, _, mean = self.forward(obs)
        std = torch.exp(self.log_std)
        pi = Normal(mean, std)
        # evaluate
        logprob = pi.log_prob(action).sum(axis=-1)
        dist_entropy = pi.entropy().sum(axis=-1).mean()          
        return v, logprob, dist_entropy
        
    def obs_rnn(self, obs):

        obs = torch.as_tensor(obs, dtype=torch.float32)  

        if self.use_gpu:
            obs=obs.cuda() 

        moving_state = obs[self.state_dim:]
        robot_state = obs[:self.state_dim]
        mov_len = int(moving_state.size()[0] / self.input_dim)
        rnn_input = torch.reshape(moving_state, (1, mov_len, self.input_dim))


        output, hn = self.rnn_net(rnn_input)
    
        hnv = torch.squeeze(hn)
        hnv = torch.sum(hnv, 0)
        
        rnn_obs = torch.cat((robot_state, hnv))
        rnn_obs = self.ln(rnn_obs)

        return rnn_obs  

    def obs_rnn_list(self, obs_tensor_list):     # [(6+8a,), (6+8b, ), (6+8c, ), ...]    tensor
        
        mov_len = [(len(obs_tensor)-self.state_dim)/self.input_dim for obs_tensor in obs_tensor_list]   # [a, b, c, ...]
        obs_pad = pad_sequence(obs_tensor_list, batch_first = True)     # (n, 6+8*max(a,b,c))
        robot_state_batch = obs_pad[:, :self.state_dim]                 # tensor (n, 6)
        batch_size = len(obs_tensor_list)
        if self.use_gpu:
            robot_state_batch=robot_state_batch.cuda()

        def obs_tensor_reform(obs_tensor):
            mov_tensor = obs_tensor[self.state_dim:]
            mov_tensor_len = int(len(mov_tensor)/self.input_dim)
            re_mov_tensor = torch.reshape(mov_tensor, (mov_tensor_len, self.input_dim)) 
            return re_mov_tensor
        
        re_mov_list = list(map(lambda o: obs_tensor_reform(o), obs_tensor_list))        # [(a, 8), (b, 8), (c, 8), ...]
        re_mov_pad = pad_sequence(re_mov_list, batch_first = True)                      # (n, max(a, b, c), 8)

        if self.use_gpu:
            re_mov_pad=re_mov_pad.cuda()

        moving_state_pack = pack_padded_sequence(re_mov_pad, mov_len, batch_first=True, enforce_sorted=False)
        

        output, hn= self.rnn_net(moving_state_pack)

        hnv = torch.squeeze(hn)

        hnv = torch.sum(hnv, 0)
        
        fc_obs_batch = torch.cat((robot_state_batch, hnv), 1)
        fc_obs_batch = self.ln(fc_obs_batch)

        return fc_obs_batch
