import logging
import itertools
import torch
import torch.nn as nn
from torch.nn.functional import softmax, relu, sigmoid, tanh
from torch.nn import Parameter
import numpy as np
import math
from config import *

def mlp(input_dim, mlp_dims, last_relu=False):
    layers = []
    mlp_dims = [input_dim] + mlp_dims
    for i in range(len(mlp_dims) - 1):
        layers.append(nn.Linear(mlp_dims[i], mlp_dims[i + 1]))
        if i != len(mlp_dims) - 2 or last_relu:
            layers.append(nn.ReLU())
    net = nn.Sequential(*layers)
    return net

def log_normal_density(x, mean, log_std, std):
    """returns guassian density given x on log scale"""

    variance = std.pow(2)
    log_density = -(x - mean).pow(2) / (2 * variance) - 0.5 * np.log(2 * np.pi) - log_std    # num_env * frames * act_size
    log_density = log_density.sum(dim=-1, keepdim=True) # num_env * frames * 1
    return log_density


class TCN_GACN(nn.Module):
    def __init__(self, robot_state_dim, obs_state_dim, action_space, goal_dim):
        """ The current code might not be compatible with models trained with previous version
        """
        super(TCN_GACN, self).__init__()
        X_dim = 32   
        wr_dims = [64, 32]
        wh_dims = [64, 32]
        final_state_dim = 32



        self.num_layer = 2
        self.X_dim = X_dim
        self.skip_connection = False


        self.logstd = nn.Parameter(torch.zeros(action_space))
        
        self.w_r = mlp(robot_state_dim, wr_dims, last_relu=True)
        self.w_h = mlp(obs_state_dim, wh_dims, last_relu=True)
        
        self.tcn_robot = nn.Sequential(
            nn.Conv2d(
                robot_state_dim,
                64,
                (3, 1),
                (1, 1),
                (0, 0),
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                64,
                32,
                (3, 1),
                (1, 1),
                (0, 0),
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                32,
                32,
                (1, 1),
                (1, 1),
                (0, 0),
            ),
            nn.ReLU(inplace=True),
        )   #(None, f=32, T=t_0, v=1)
        
        self.tcn_other = nn.Sequential(
            nn.Conv2d(
                obs_state_dim,
                64,
                (3, 1),
                (1, 1),
                (0, 0),
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                64,
                32,
                (3, 1),
                (1, 1),
                (0, 0),
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                32,
                32,
                (1, 1),
                (1, 1),
                (0, 0),
            ),
            nn.ReLU(inplace=True),
        )   #(None, f=32, T=t_0, v=3)
        self.goal_mlp = mlp(goal_dim, [64, 32], last_relu=True)   #(None, v=1, f=32)


        self.w_a = Parameter(torch.randn(32, 32))
        # TODO: try other dim size
        embedding_dim = 32
        self.Wsa = torch.nn.ParameterList()
        for i in range(self.num_layer): # 2
            if i == 0:
                self.Wsa.append(Parameter(torch.randn(self.X_dim, embedding_dim)))
            elif i == self.num_layer - 1:
                self.Wsa.append(Parameter(torch.randn(embedding_dim, final_state_dim)))
            else:
                self.Wsa.append(Parameter(torch.randn(embedding_dim, embedding_dim)))
        self.a_fc = nn.Linear(final_state_dim, 64)
        self.a_v_fc = nn.Linear(64, 1)
        self.a_w_fc = nn.Linear(64, 1)


        self.w_v = mlp(2 * 32, [2 * 32, 1], last_relu=True)
        self.Wsv = torch.nn.ParameterList()
        for i in range(self.num_layer):
            if i == 0:
                self.Wsv.append(Parameter(torch.randn(self.X_dim, embedding_dim))) # 32,32
            elif i == self.num_layer - 1:
                self.Wsv.append(Parameter(torch.randn(embedding_dim, final_state_dim))) # 32, 32
            else:
                self.Wsv.append(Parameter(torch.randn(embedding_dim, embedding_dim))) # 32,32
                
        self.v_obs_fc = nn.Linear(final_state_dim, 64) # (32, 64)
        self.v_goal_fc = nn.Linear(64, 64) 
        self.critic = nn.Linear(128, 1)
                    

    def compute_similarity_matrix_embedded_gaussian(self, X): # TODO
        A = torch.matmul(torch.matmul(X, self.w_a), X.permute(0, 2, 1))
        normalized_A = softmax(A, dim=2)
        return normalized_A
    
    def compute_similarity_matrix_concatenation(self, X): # TODO
        indices = [pair for pair in itertools.product(list(range(X.size(1))), repeat=2)]  
        selected_features = torch.index_select(X, dim=1, index=torch.LongTensor(indices).reshape(-1).cuda())   
        
        pairwise_features = selected_features.reshape((-1, X.size(1) * X.size(1), X.size(2) * 2))    
        A = self.w_v(pairwise_features).reshape(-1, X.size(1), X.size(1))   #(none, 3, 3)  
        return A
        

    def forward(self, state, goal, obs):
        """
        Embed current state tensor pair (robot_state, human_states) into a latent space
        Each tensor is of shape (batch_size, # of agent, features)
        :param state:
        :return:
        """
        #encode
        #(batch, features, T, 1), (batch, 1, 2) , (batch, features, T, 3)
        robot_state = self.tcn_robot(state) # encoder1
        goal_state = self.goal_mlp(goal)    # encoder2
        other_state = self.tcn_other(obs)   # encoder3
        x1 = robot_state.permute(0, 3, 2, 1).contiguous()
        x1 = x1.view(-1, 1, 32)
        x2 = other_state.permute(0, 3, 2, 1).contiguous()
        x2 = x2.view(-1, NUM_ENV - 1, 32) # NUM_ENV: number of robots
        x3 = goal_state.view(-1, 1, 32)
        # print(x1.shape, x2.shape, x3.shape)
        #(batch, features, T, n) -> (batch, n, features)
        X = torch.cat([x1, x3, x2], dim=1)

        #actor module, GAN1
        next_H = H = X
        for i in range(self.num_layer):
            A = self.compute_similarity_matrix_embedded_gaussian(H) # attention
            next_H = relu(torch.matmul(torch.matmul(A, H), self.Wsa[i]))
            if self.skip_connection:
                next_H += H
            H = next_H
        a = relu(self.a_fc(H[:, 0, :]))
        mean1 = sigmoid(self.a_v_fc(a)) # linear velocity
        mean2 = tanh(self.a_w_fc(a)) # angular velocity
        mean = torch.cat((mean1, mean2), dim=-1)
        
        logstd = self.logstd.expand_as(mean)
        std = torch.exp(logstd)
        action = torch.normal(mean, std)
        logprob = log_normal_density(action, mean, std=std, log_std=logstd)
        
        #critic module, GAN2
        goal_v = torch.cat([x1[:, 0, :], x3[:, 0, :]], dim=1)  #(none, 64)
        X_v = torch.cat([x1, x2], dim=1)
        next_Hv = Hv = X_v
        for i in range(self.num_layer):
            A = self.compute_similarity_matrix_concatenation(Hv) # adjacent matrix
            next_H = relu(torch.matmul(torch.matmul(A, Hv), self.Wsv[i]))
            if self.skip_connection:
                next_Hv += Hv
            Hv = next_Hv
            
        v_obs = relu(self.v_obs_fc(Hv[:, 0, :]))
        v_goal = relu(self.v_goal_fc(goal_v))
        v = torch.cat([v_obs, v_goal], dim=1)
        v = self.critic(v)
        
        return v, action, logprob, mean
        
    def evaluate_actions(self, state, goal, obs, action):
        v, _, _, mean = self.forward(state, goal, obs)
        logstd = self.logstd.expand_as(mean)
        std = torch.exp(logstd)
        # evaluate
        logprob = log_normal_density(action, mean, log_std=logstd, std=std)
        dist_entropy = 0.5 + 0.5 * math.log(2 * math.pi) + logstd
        dist_entropy = dist_entropy.sum(-1).mean()
        return v, logprob, dist_entropy
        
        