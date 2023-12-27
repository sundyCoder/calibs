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
    activation=nn.ReLU, output_activation=nn.Tanh, output_activation_v= nn.Identity, use_gpu=True, drop_p=0):
        super(RNN_AC,self).__init__()
        self.use_gpu = use_gpu
        torch.cuda.synchronize()
        obs_dim = (rnn_hidden_dim + state_dim)
        
        rnn = rnn_Reader(state_dim, rnn_input_dim, rnn_hidden_dim, use_gpu=use_gpu)
        
        self.pi = GaussianActor(obs_dim, action_dim, hidden_sizes_ac, activation, output_activation, rnn_reader=rnn, use_gpu=use_gpu)
        self.v = Critic(obs_dim, hidden_sizes_v, activation, output_activation_v, rnn_reader=rnn, use_gpu=use_gpu)
        
    def forward(self, obs):
        with torch.no_grad():
            pi_dis = self.pi._distribution(obs)
            a = pi_dis.sample()
            logp_a = self.pi._log_prob_from_distribution(pi_dis, a)
            v = self.v(obs)
            
            if self.use_gpu:
                a = a.cpu()
                logp_a = logp_a.cpu()
                v = v.cpu()
                
        return a.numpy(), v.numpy(), logp_a.numpy()
        
    def get_actions(self, obs):
        with torch.no_grad():
            pi = self.pi._distribution(obs, std_factor = 0.000001)
            
            action = pi.sample()
            if self.use_gpu:
                action = action.cpu()
        return action.numpy()
            

class rnn_Reader(nn.Module):
    def __init__(self, state_dim, input_dim, hidden_dim, use_gpu):
        super(rnn_Reader, self).__init__()
        
        self.state_dim = state_dim
        self.rnn_net = nn.GRU(input_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.use_gpu = use_gpu
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        des_dim = state_dim + hidden_dim
        self.ln = nn.LayerNorm(des_dim)
        if use_gpu: 
            self.rnn_net = self.rnn_net.cuda()
            self.ln = self.ln.cuda()
            
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

    def obs_rnn_list(self, obs_tensor_list):
        #print(obs_tensor_list)
        mov_len = [(len(obs_tensor)-self.state_dim)/self.input_dim for obs_tensor in obs_tensor_list]
        obs_pad = pad_sequence(obs_tensor_list, batch_first = True)
        robot_state_batch = obs_pad[:, :self.state_dim] 
        batch_size = len(obs_tensor_list)
        if self.use_gpu:
            robot_state_batch=robot_state_batch.cuda()

        def obs_tensor_reform(obs_tensor):
            mov_tensor = obs_tensor[self.state_dim:]
            mov_tensor_len = int(len(mov_tensor)/self.input_dim)
            re_mov_tensor = torch.reshape(mov_tensor, (mov_tensor_len, self.input_dim)) 
            return re_mov_tensor
        
        re_mov_list = list(map(lambda o: obs_tensor_reform(o), obs_tensor_list))
        re_mov_pad = pad_sequence(re_mov_list, batch_first = True)

        if self.use_gpu:
            re_mov_pad=re_mov_pad.cuda()

        # print(re_mov_pad.shape, mov_len)
        moving_state_pack = pack_padded_sequence(re_mov_pad, mov_len, batch_first=True, enforce_sorted=False)
        

        output, hn= self.rnn_net(moving_state_pack)

        hnv = torch.squeeze(hn)
        hnv = torch.sum(hnv, 0)
        
        fc_obs_batch = torch.cat((robot_state_batch, hnv), 1)
        fc_obs_batch = self.ln(fc_obs_batch)

        return fc_obs_batch

class Actor(nn.Module):
    def _distribution(self, obs):
        raise NotImplementedError

    def _log_prob_from_distribution(self, pi, act):
        raise NotImplementedError

    def forward(self, obs, act=None, std_factor=1):
        # Produce action distributions for given observations, and 
        # optionally compute the log likelihood of given actions under
        # those distributions.
        pi = self._distribution(obs, std_factor)
        logp_a = None

        if act is not None:   
            logp_a = self._log_prob_from_distribution(pi, act)

        return pi, logp_a
      
class GaussianActor(Actor):
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, output_activation, rnn_reader=None, use_gpu=False):
        super(GaussianActor, self).__init__()
        self.rnn_reader = rnn_reader
        self.use_gpu = use_gpu
        self.net_out=mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation, output_activation)
        
        log_std = -1 * np.ones(act_dim, dtype=np.float32)
        
        if use_gpu:
            self.log_std = torch.nn.Parameter(torch.as_tensor(log_std, device=torch.device('cuda')))
            self.net_out=self.net_out.cuda()
        else:
            self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))
            
    def _distribution(self, obs, std_factor=1):
        if isinstance(obs, list):
            obs = self.rnn_reader.obs_rnn_list(obs)
            net_out = self.net_out(obs)
        else:
            obs = self.rnn_reader.obs_rnn(obs)
            net_out = self.net_out(obs)
        
        mu = net_out 
        std = torch.exp(self.log_std)
        std = std_factor * std
        
        return Normal(mu, std) 
        
    def _log_prob_from_distribution(self, pi, act):

        if self.use_gpu:
            act = act.cuda()

        return pi.log_prob(act).sum(axis=-1)
        
class Critic(nn.Module):
    def __init__(self, obs_dim, hidden_sizes, activation, output_activation, rnn_reader=None, use_gpu=False):
        super(Critic, self).__init__()
        self.v_net = mlp([obs_dim] + list(hidden_sizes) + [1], activation, output_activation)

        if use_gpu:
            self.v_net = self.v_net.cuda()

        self.rnn_reader = rnn_reader

    def forward(self, obs):

        if self.rnn_reader != None:
            if isinstance(obs, list):
                obs = self.rnn_reader.obs_rnn_list(obs)
            else:
                obs = self.rnn_reader.obs_rnn(obs)
        v = torch.squeeze(self.v_net(obs), -1)

        return v 