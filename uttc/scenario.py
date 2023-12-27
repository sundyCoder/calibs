
from config import *
import numpy as np
from math import sin, cos, atan2, pi, sqrt


def init_pose_goal(index):
    
    state_list, goal_list = [], []
    square, circular = [0, 0, 10, 10], [0, 0, 4]
    circle_point = np.array(circular)
    theta_step, theta = 2*pi / NUM_ENV, 0

    while theta < 2*pi:
        state = circle_point + np.array([cos(theta) * circular[2], sin(theta) * circular[2], theta + pi - circular[2] ])
        goal = circle_point[0:2] + np.array([cos(theta+pi), sin(theta+pi)]) * circular[2]
        theta = theta + theta_step
        state_list.append(list(np.round(state, 2)))
        goal_list.append(list(np.round(goal, 2)))
    return [state_list[index], goal_list[index]]


if __name__ == "__main__":
    for idx in range(NUM_ENV):
        [start, goal] = init_pose_goal(idx)
        print(start, goal)
