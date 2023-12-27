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

from envGazebo_4r import GazeboWorld
from Network import RNN_AC
from config import *
from utils import ppo_update, generate_action, Buff



def run(comm, env, policy, policy_path, pi_optimizer, v_optimizer):
    # rate = rospy.Rate(5)
    global_update = 0
    buff = Buff(6, 2, STEPS_PER_EPOCH, gamma=0.99, lam=0.95)
    task_time = 0
    if env.index == 0:
        env.ResetWorld()
    for id in range(MAX_EPISODES):
        epoch_step = 0
        ep_reward = 0
        robot_step = 0
        env.reset_pose()
        observation = env.get_observation()
        observation = torch.as_tensor(observation, dtype=torch.float32)

        while epoch_step < STEPS_PER_EPOCH and not rospy.is_shutdown():
            state_list = comm.gather(observation, root=0)
                                                            
            action, value, logprob = generate_action(env=env, state_list=state_list, policy=policy)

            a_inc = comm.scatter(action, root=0)
            v = comm.scatter(value, root=0)
            logp = comm.scatter(logprob, root=0)
            
            a = np.round(a_inc, 2)
            a_inc = np.clip(a, a_min=[0, -1], a_max=[0.5, 1])

            real_action = a_inc
            env.Control(real_action[0], real_action[1])        
            rospy.sleep(0.1)
            r, terminal, result, info = env.GetRewardAndTerminate(robot_step)
            next_observation = env.get_observation()
            next_observation = torch.as_tensor(next_observation, dtype=torch.float32)
            
            buff.store(observation, a_inc, r, v, logp)
            ep_reward += r
            robot_step += 1

            observation = next_observation

            epoch_end = False
            if epoch_step == STEPS_PER_EPOCH - 1:
                epoch_end = True
            arrival = info['arrived'] == True
            terminal = info['crashed'] == True or robot_step > HORIZON
            epoch_step += 1

            if epoch_end or arrival:
                logger.info('Env %02d, Goal (%05.1f, %05.1f), Episode %05d, step %03d, Reward %-5.1f, %s' %(env.index, env.goal[0], env.goal[1], task_time + 1, robot_step, ep_reward, result))
                logger_cal.info(ep_reward)
                task_time += 1
                env.reset_pose()
                ep_reward = 0
                robot_step = 0
                buff.finish_path(0)
                observation = env.get_observation()
                observation = torch.as_tensor(observation, dtype=torch.float32)   
            elif terminal:
                if info['crashed'] or robot_step > HORIZON:
                    logger.info('Env %02d, Goal (%05.1f, %05.1f), Episode %05d, step %03d, Reward %-5.1f, %s' %(env.index, env.goal[0], env.goal[1], task_time + 1, robot_step, ep_reward, result))
                    logger_cal.info(ep_reward)
                    task_time += 1
                    env.reset_pose()
                    ep_reward = 0
                    robot_step = 0
                    buff.finish_path(0)
                    observation = env.get_observation()
                    observation = torch.as_tensor(observation, dtype=torch.float32)
        
        buff_data = buff.get()
        if buff_data is None:
            continue
        data_list_temp = comm.gather(buff_data, root = 0)
        if data_list_temp is None:
            continue
        # print(data_list_temp)
        data_list = []
        for item in data_list_temp:
            if item is not None:
                data_list.append(item)
            
        #if all(data_list):
        #    print( data_list)
        #    print(type(data_list))
                
        if env.index == 0 and len(data_list) != 0:
            ppo_update(policy, pi_optimizer, v_optimizer, data_list, clip_value=CLIP_VALUE, num_env=NUM_ENV, max_update_num=10, use_gpu=True)
            global_update += 1
            if global_update != 0 and global_update % 20 == 0:
                torch.save(policy.state_dict(), policy_path + '/Stage1_{}'.format(global_update))
                logger.info('########################## model saved when update {} times#########'
                                '################'.format(global_update))
            with open('train_stats.csv', 'a+') as f:
                wr = csv.writer(f)
                wr.writerow(['%.4f' % s if type(s) is float else s for s in [id+1, epoch_step, ep_reward, ep_reward/epoch_step]])

             

    


        




def run1(comm, env, policy, policy_path, pi_optimizer, v_optimizer):
    # rate = rospy.Rate(5)
    global_update = 0
    global_step = 0
    buff = Buff(6, 2, STEPS_PER_EPOCH, gamma=0.99, lam=0.95)
    if env.index == 0:
        env.ResetWorld()
    for id in range(MAX_EPISODES):
        env.reset_pose()
        terminal = False
        ep_reward = 0
        step = 0
        
        observation = env.get_observation()
        observation = torch.as_tensor(observation, dtype=torch.float32)
        state = observation

        while not terminal and not rospy.is_shutdown():

            state_list = comm.gather(state, root=0)
                                                         
            action, value, logprob = generate_action(env=env, state_list=state_list, policy=policy)

            a_inc = comm.scatter(action, root=0)
            v = comm.scatter(value, root=0)
            logp = comm.scatter(logprob, root=0)
            
            a = np.round(a_inc, 2)
            a_inc = np.clip(a, a_min=[0, -1], a_max=[0.5, 1])

            real_action = a_inc
            env.Control(real_action[0], real_action[1])
            
            rospy.sleep(0.1)

            # get informtion
            r, terminal, result = env.GetRewardAndTerminate(step)
            
            observation = env.get_observation()
            observation = torch.as_tensor(observation, dtype=torch.float32)
            state_next = observation
            buff.store(state, a_inc, r, v, logp)
            step += 1
            ep_reward += r
            global_step += 1
            state = state_next

            if terminal:
                buff.finish_path(0)
            
            buff_data = buff.get()
            if buff_data is None:
                continue
            data_list_temp = comm.gather(buff_data, root = 0)
            if data_list_temp is None:
                continue
            # print(data_list_temp)
            data_list = []
            for item in data_list_temp:
                if item is not None:
                    data_list.append(item)
            
            #if all(data_list):
            #    print( data_list)
            #    print(type(data_list))
                
            if env.index == 0 and len(data_list) != 0:
                ppo_update(policy, pi_optimizer, v_optimizer, data_list, clip_value=CLIP_VALUE, num_env=NUM_ENV, max_update_num=10, use_gpu=True)
                global_update += 1

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

    # torch.manual_seed(1)
    # np.random.seed(1)
    if rank == 0:
        policy_path = 'policy'
        policy = RNN_AC()
        pi_optimizer = Adam(policy.pi.parameters(), lr=LEARN_RATE_A) 
        v_optimizer = Adam(policy.v.parameters(), lr=LEARN_RATE_V)

        if not os.path.exists(policy_path):
            os.makedirs(policy_path)

        # file = 'policy/pre_train_check_point_1000.pt'
        file = "policy/Stage_100"

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
        pi_optimizer = None
        v_optimizer = None

    try:
        run(comm=comm, env=env, policy=policy, policy_path=policy_path, pi_optimizer=pi_optimizer, v_optimizer=v_optimizer)
    except KeyboardInterrupt:
        pass
