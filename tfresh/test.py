from __future__ import print_function

import logging
import os,sys
import socket
import numpy as np
import rospy
import torch
import csv
import time
import torch.nn as nn
from mpi4py import MPI
from collections import deque
from torch.optim import Adam
from envGazebo_8r import GazeboWorld
from Network import TCN_GACN
from config import *
from utils import generate_action_test 


def enjoy(rank, comm, env, policy, action_bound):
    env.reset_pose()
    step = 1
    terminal = False

    state_frame = env.GetSelfState() # [x,y,o,v,a,r]
    state_stack = deque([state_frame, state_frame, state_frame, state_frame, state_frame])
    state_stack1 = np.asarray(state_stack) #(5, 6)
    State = state_stack1.swapaxes(0, 1)[:, :, np.newaxis] #(6, 5, 1)
    goal = np.asarray(env.GetLocalGoal()).reshape(1, 2)
    
    obs = env.GetObstacle()
    obs_stack = deque([obs, obs, obs, obs, obs])
    obs_stack1 = np.asarray(obs_stack)   #(5, 3, 6)
    Obs = obs_stack1.swapaxes(0, 1).swapaxes(0, 2)  #(6, 5, 3)  
    state = [State, goal, Obs]

    #print state 
    time0 = time.time()
    travel_time, avg_speed, comp_cost = 0, [], []
    log_file = open("robot" + "_" + str(rank) +".log", "w+")
    while (not rospy.is_shutdown()):
        start_time = time.time()
        state_list = comm.gather(state, root=0)

        mean,scaled_action=generate_action_test(env=env, state_list=state_list, policy=policy, action_bound=action_bound)
        log_file.write("comp cost: {}\n".format(time.time() - start_time))

        # execute actions
        real_action = comm.scatter(scaled_action, root=0)
        if terminal:
            env.Control(0, 0)
        elif real_action is None:
            logger.info("action is none")
            env.Control(0.1, 0)
        else:
            env.Control(real_action[0], real_action[1])

        rospy.sleep(0.01)
        # get informtion
        r, terminal, result = env.GetRewardAndTerminate(step, is_training=False)
        if terminal:
            log_file.write("travel time: {}\n".format(step))
        step += 1

        # get next state
        s_next = env.GetSelfState()
        log_file.write("x, y, v, w: {}, {}, {}, {}\n".format(np.round(s_next[0],4), np.round(s_next[1], 4), np.round(real_action[0], 4), np.round(real_action[1], 4)))
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
        state = state_next

        terminate_list = comm.gather(terminal, root=0)
        #print(terminate_list)
        if terminate_list == [True]*NUM_ENV:
            logger.info(terminate_list)
            env.Control(0, 0)
            #break

    logger.info("total time=%3d" % (time.time() - time0))

            
if __name__ == '__main__':

    hostname = socket.gethostname()
    if not os.path.exists('./log/' + hostname):
        os.makedirs('./log/' + hostname)
    output_file = './log/' + hostname + '/test_log.log'

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

    # config log
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    env = GazeboWorld(index=rank, num_env=NUM_ENV)
    action_bound = [[0, -1], [0.5, 1]] # v = [0, 0.5], a = [-1, 1]

    if rank == 0:
        policy_path = 'policy_ttc2'
        policy_path = 'policy_random'
        policy_path = 'policy_ttc_swap'
        policy = TCN_GACN(robot_state_dim=6, obs_state_dim=6, action_space=2, goal_dim=2)
        policy.cuda()        

        #file = 'policy_ttc2/Stage1_4960' # circle_6
        #file = policy_path + '/Stage1_13000' # swap_6
        #file = "policy_cross/Stage1_4000" # cross_8
        file = "policy/Stage1_800"
        if os.path.exists(file):
            logger.info('####################################')
            logger.info('############Loading Model###########')
            logger.info('####################################')
            state_dict = torch.load(file)
            policy.load_state_dict(state_dict)
        else:
            logger.info('Model %s is not exists!' % (file)) 
    else:
        policy = None
        policy_path = None
        opt = None

    try:
        enjoy(rank, comm = comm, env=env, policy=policy, action_bound=action_bound)
    except KeyboardInterrupt:
        import traceback
        traceback.print_exc()
