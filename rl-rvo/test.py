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
from Network1 import RNN_AC
from config import *
from utils1 import generate_action_test 


def enjoy(rank, comm, env, policy):
    env.reset_pose()
    step = 0
    terminal = False

    observation = env.get_observation()
    observation = torch.as_tensor(observation, dtype=torch.float32)
    state = observation

    #print state 
    time0 = time.time()
    travel_time, avg_speed, comp_cost = 0, [], []
    log_file = open("tb" + "_" + str(rank) +".log", "w+")
    while (not rospy.is_shutdown()):
        start_time = time.time()
        state_list = comm.gather(state, root=0)

        action = generate_action_test(env=env, state_list=state_list, policy=policy)
        log_file.write("comp cost: {}\n".format(time.time() - start_time))

        # execute actions
        acc_vx_vy = comm.scatter(action, root=0)
        vel_cur = state.numpy()[0:2]
        action_vx_vy = vel_cur + env.acc * acc_vx_vy
        # action_vx_vy = acc_vx_vy
        
        real_action = env.omni2diff(action_vx_vy)
        
        if terminal:
            env.Control(0, 0)
        elif real_action is None:
            logger.info("action is none")
            env.Control(0.1, 0)
        else:
            env.Control(real_action[0], real_action[1])

        rospy.sleep(0.01)
        # get informtion
        _, terminal, _ = env.GetRewardAndTerminate(step, is_training=False)
        if terminal:
            log_file.write("travel time: {}\n".format(step))
        step += 1

        # get next state
        robot_state = env.GetSelfState()
        log_file.write("{}, {}, {}, {}\n".format(np.round(robot_state[0],4), np.round(robot_state[1], 4), np.round(real_action[0], 4), np.round(real_action[1], 4)))

        observation = env.get_observation(action_vx_vy)
        observation = torch.as_tensor(observation, dtype=torch.float32)
        state_next = observation
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

    if rank == 0:
        policy = RNN_AC()
        policy.cuda()        
        file = "new/policy.pth"
        if os.path.exists(file):
            logger.info('####################################')
            logger.info('############Loading Model###########')
            logger.info('####################################')
            state_dict = torch.load(file)
            policy.load_state_dict(state_dict['model_state'], strict=True)
        else:
            logger.info('Model %s is not exists!' % (file)) 
    else:
        policy = None
        policy_path = None
        opt = None

    try:
        enjoy(rank, comm = comm, env=env, policy=policy)
    except KeyboardInterrupt:
        import traceback
        traceback.print_exc()
