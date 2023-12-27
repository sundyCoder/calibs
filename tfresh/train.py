from __future__ import print_function


import logging
import os
import sys
import socket
import numpy as np
import rospy
import torch
import torch.nn as nn
from mpi4py import MPI
import csv

from collections import deque
from torch.optim import Adam

from envGazebo_8r import GazeboWorld
from Network import TCN_GACN
from config import *
from utils import ppo_update, generate_train_data, generate_action, transform_buffer



def run(comm, env, policy, policy_path, action_bound, optimizer):
    # rate = rospy.Rate(5)
    buff = []
    global_update = 0
    global_step = 0


    if env.index == 0:
        env.ResetWorld()


    for id in range(MAX_EPISODES):
        env.reset_pose()
        terminal = False
        ep_reward = 0
        step = 1
        
        state_frame = env.GetSelfState() # [x,y,o,v,a,r]
        state_stack = deque([state_frame, state_frame, state_frame, state_frame, state_frame]) # [5, 6]
        state_stack1 = np.asarray(state_stack)#(5, 6)
        State = state_stack1.swapaxes(0, 1)[:, :, np.newaxis] #(6, 5, 1)

        goal = np.asarray(env.GetLocalGoal()).reshape(1, 2)
        
        obs = env.GetObstacle()
        obs_stack = deque([obs, obs, obs, obs, obs])
        obs_stack1 = np.asarray(obs_stack)   #(5, 3, 6)
        Obs = obs_stack1.swapaxes(0, 1).swapaxes(0, 2)  #(6, 5, 3)  
        
        state = [State, goal, Obs]

        while not terminal and not rospy.is_shutdown():
            state_list = comm.gather(state, root=0)


            # generate actions at rank==0
            v, a, logprob, scaled_action=generate_action(env=env, state_list=state_list,
                                                         policy=policy, action_bound=action_bound)

            # execute actions
            real_action = comm.scatter(scaled_action, root=0)
            env.Control(real_action[0], real_action[1])

            # rate.sleep()
            rospy.sleep(0.1)

            # get informtion
            r, terminal, result = env.GetRewardAndTerminate(step)
            ep_reward += r
            global_step += 1

            # get next state
            s_next = env.GetSelfState()
            left_state = state_stack.popleft()
            state_stack.append(s_next)
            state_stack1 = np.asarray(state_stack)
            State = state_stack1.swapaxes(0, 1)[:, :, np.newaxis] #(6, 5, 1)
            
            goal_next = np.asarray(env.GetLocalGoal()).reshape(1, 2)
            
            obs_next = env.GetObstacle()
            left_obs = obs_stack.popleft()
            obs_stack.append(obs_next)
            obs_stack1 = np.asarray(obs_stack)
            Obs = obs_stack1.swapaxes(0, 1).swapaxes(0, 2)
            
            state_next = [State, goal_next, Obs]

            if global_step % HORIZON == 0:
                state_next_list = comm.gather(state_next, root=0)
                last_v, _, _, _ = generate_action(env=env, state_list=state_next_list, policy=policy,
                                                               action_bound=action_bound)
            # add transitons in buff and update policy
            r_list = comm.gather(r, root=0)
            terminal_list = comm.gather(terminal, root=0)

            if env.index == 0:
                buff.append((state_list, a, r_list, terminal_list, logprob, v))
                if len(buff) > HORIZON - 1:
                    s_batch, goal_batch, speed_batch, a_batch, r_batch, d_batch, l_batch, v_batch = \
                        transform_buffer(buff=buff)
                    t_batch, advs_batch = generate_train_data(rewards=r_batch, gamma=GAMMA, values=v_batch,
                                                              last_value=last_v, dones=d_batch, lam=LAMDA)
                    memory = (s_batch, goal_batch, speed_batch, a_batch, l_batch, t_batch, v_batch, r_batch, advs_batch)
                    ppo_update(policy=policy, optimizer=optimizer, batch_size=BATCH_SIZE, memory=memory,
                                            epoch=EPOCH, coeff_entropy=COEFF_ENTROPY, clip_value=CLIP_VALUE, num_step=HORIZON,
                                            num_env=NUM_ENV, act_size=ACT_SIZE, T=TIME, robot_state_num=LEN_STATE, obs_state_num=LEN_STATE)

                    buff = []
                    global_update += 1

            step += 1
            state = state_next


        if env.index == 0:
            if global_update != 0 and global_update % 20 == 0:
                torch.save(policy.state_dict(), policy_path + '/Stage1_{}'.format(global_update))
                logger.info('########################## model saved when update {} times#########'
                            '################'.format(global_update))
            with open('train_stats.csv', 'a+') as f:
                wr = csv.writer(f)
                wr.writerow(['%.4f' % s if type(s) is float else s for s in [id+1, step, ep_reward, ep_reward/step]])
                

        logger.info('Env %02d, Goal (%05.1f, %05.1f), Episode %05d, setp %03d, Reward %-5.1f, %s' % \
                    (env.index, env.goal[0], env.goal[1], id + 1, step, ep_reward, result))
        logger_cal.info(ep_reward)


            
if __name__ == '__main__':

    # config log
    hostname = socket.gethostname()
    if not os.path.exists('./log/' + hostname):
        os.makedirs('./log/' + hostname)
    output_file = './log/' + hostname + '/output.log'
    cal_file = './log/' + hostname + '/cal.log'

    # config log
    logger = logging.getLogger('mylogger')
    logger.setLevel(logging.INFO)

    file_handler = logging.FileHandler(output_file, mode='a')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.addHandler(stdout_handler)

    logger_cal = logging.getLogger('loggercal')
    logger_cal.setLevel(logging.INFO)
    cal_f_handler = logging.FileHandler(cal_file, mode='a')
    file_handler.setLevel(logging.INFO)
    logger_cal.addHandler(cal_f_handler)

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    env = GazeboWorld(index=rank, num_env=NUM_ENV)
    reward = None
    action_bound = [[0, -1], [0.5, 1]] # v = [0, 0.5], a = [-1, 1]

    # torch.manual_seed(1)
    # np.random.seed(1)
    if rank == 0:
        policy_path = 'policy'
        # policy = MLPPolicy(obs_size, act_size)
        policy = TCN_GACN(robot_state_dim=6, obs_state_dim=6, action_space=2, goal_dim=2)
        policy.cuda()
        opt = Adam(policy.parameters(), lr=LEARNING_RATE)
        mse = nn.MSELoss()

        if not os.path.exists(policy_path):
            os.makedirs(policy_path)

        #file = 'policy_ttc2/Stage1_4960'
        file = 'policy_cross/Stage_4000'

        if os.path.exists(file):
            logger.info('####################################')
            logger.info('############Start Loading Model###########')
            logger.info('####################################')
            state_dict = torch.load(file)
            policy.load_state_dict(state_dict)
            logger.info('############Finish Loading Model###########')
        else:
            logger.info('#####################################')
            logger.info('############Start Training###########')
            logger.info('#####################################')
    else:
        policy = None
        policy_path = None
        opt = None

    try:
        run(comm=comm, env=env, policy=policy, policy_path=policy_path, action_bound=action_bound, optimizer=opt)
    except KeyboardInterrupt:
        pass
