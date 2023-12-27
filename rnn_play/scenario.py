import rospy
from config import *
import itertools
import numpy as np
from collections import namedtuple
from math import sin, cos, atan2, pi, sqrt

#np.random.seed(2022)


def collision_cir_cir(circle1, circle2):

    dis = sqrt( (circle2.x - circle1.x)**2 + (circle2.y - circle1.y)**2 )

    if dis >0 and dis <= circle1.r + circle2.r:
        return True
    
    return False

def collision_cir_matrix(circle, matrix, reso, offset=np.zeros(2,)):

    if matrix is None:
        return False

    rad_step = 0.1
    cur_rad = 0

    while cur_rad <= 2*pi:
        crx = circle.x + circle.r * cos(cur_rad)
        cry = circle.y + circle.r * sin(cur_rad)
        cur_rad = cur_rad + rad_step
        index_x = int( (crx - offset[0]) / reso)
        index_y = int( (cry - offset[1]) / reso)
        if matrix[index_x, index_y]:
            return True

def collision_cir_seg(circle, segment):
    
    # reference: https://stackoverflow.com/questions/1073336/circle-line-segment-collision-detection-algorithm
    
    point = np.array([circle.x, circle.y])
    sp = np.array([segment[0].x, segment[0].y])
    ep = np.array([segment[1].x, segment[1].y])
    d_pos = (ep - sp)
    l2 = d_pos[0]**2 + d_pos[1]**2
    
    if (l2 == 0.0):
        distance = np.linalg.norm(point - sp)
        if distance < circle.r:
            return True

    #(point-sp) @ (ep-sp)
    ps = (point - sp) - (ep - sp)
    ps_l2 = ps[0]**2 + ps[1]**2
    t = max(0, min(1, (ps_l2) / l2 ))

    projection = sp + t * (ep-sp)
    relative = projection - point

    distance = np.linalg.norm(relative) 
    # angle = atan2( relative[1], relative[0] )
    if distance < circle.r:
        return True
    

def distance(point1, point2):
    diff = point2[0:2] - point1[0:2]
    return np.linalg.norm(diff)


def check_collision(check_point, point_list, components, range):

    circle = namedtuple('circle', 'x y r')
    point = namedtuple('point', 'x y')
    self_circle = circle(check_point[0, 0], check_point[1, 0], range)

    # for obs_cir in components['obs_circles'].obs_cir_list:
    #     temp_circle = circle(obs_cir.state[0, 0], obs_cir.state[1, 0], obs_cir.radius)
    #     if collision_cir_cir(self_circle, temp_circle):
    #         return True
    
    # # check collision with map
    # if collision_cir_matrix(self_circle, components['map_matrix'], components['xy_reso'], components['offset']):
    #     return True

    # # check collision with line obstacles
    # for line in components['obs_lines'].obs_line_states:
    #     segment = [point(line[0], line[1]), point(line[2], line[3])]
    #     if collision_cir_seg(self_circle, segment):
    #         return True

    for point in point_list:
        if distance(check_point, point) < range:
            return True
            
    return False


def random_start_goal(components=[], interval = 1.0):
    random_list = []
    goal_list = []
    while len(random_list) < 2 * NUM_ENV:

        new_point = np.random.uniform(low = SQUARE[0:2]+[-pi], high = SQUARE[2:4]+[pi], size = (1, 3)).T

        if not check_collision(new_point, random_list, components, interval):
            random_list.append(new_point)
            #print(len(random_list))

    start_list = random_list[0 : NUM_ENV]
    goal_temp_list = random_list[NUM_ENV: 2 * NUM_ENV]

    for goal in goal_temp_list:
        goal_list.append(np.delete(goal, 2, 0))

    return start_list, goal_list


def create_circle_init_goal(index):
    start_list, goal_list = [], []
    square, circular = [0, 0, 10, 10], [0, 0, 2]
    circle_point = np.array(circular)
    theta_step, theta = 2*pi / NUM_ENV, 0

    while theta < 2*pi:
        state = circle_point + np.array([cos(theta) * circular[2], sin(theta) * circular[2], theta + pi - circular[2] ])
        goal = circle_point[0:2] + np.array([cos(theta+pi), sin(theta+pi)]) * circular[2]
        theta = theta + theta_step
        start_list.append(list(np.round(state, 2)))
        goal_list.append(list(np.round(goal, 2)))
    return [start_list[index], goal_list[index]]


def create_random_init_goal(index):
    start_array, goal_array = random_start_goal()
    start_list, goal_list = [], []
    for idx in range(len(start_array)):
        list2d_s, list2d_g = start_array[idx], goal_array[idx]
        ups_list, upg_list = list(itertools.chain(*list2d_s)), list(itertools.chain(*list2d_g))
        start_list.append(ups_list)
        goal_list.append(upg_list)
    return [start_list[index], goal_list[index]]


if __name__ == "__main__":
    for index in range(NUM_ENV):
        #start_pos = create_circle_init_goal(index)
        start_pos = create_random_init_goal(index)[0]
        goal_pos = create_random_init_goal(index)[1]
        print(start_pos, goal_pos)
