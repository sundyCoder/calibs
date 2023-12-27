import math
import numpy as np
import cv2
import rospy
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist, Point, Quaternion
import tf
import time
from cv_bridge import CvBridge
from std_msgs.msg import String
from math import radians, copysign, sqrt, pow, pi, atan2, cos, sin, tan, asin
from tf.transformations import euler_from_quaternion
import numpy as np
from numpy import array, dot
import copy

def processObs(odometry, name):
    Quaternions = odometry.pose.pose.orientation
    Euler = tf.transformations.euler_from_quaternion([Quaternions.x, Quaternions.y, Quaternions.z, Quaternions.w])
    if name == 'burger':
        r = 0.2
    else:
        r = 0.4
    return [odometry.pose.pose.position.x, odometry.pose.pose.position.y, Euler[2], odometry.twist.twist.linear.x, r]


def processScan(data):
    currentLidar = [0] * 91
    for i in range(91):
        if i <= 45:
            currentLidar[45+i] = data[i]  
        else:
            currentLidar[i - 46] = data[i + 269]
    result = []
    obs = []
    #isAObs = False
    k = 0
    for j, item in enumerate(currentLidar):
        if item <= 0.9 and item >= 0.15:
            obs.append(j)
            #isAObs = True
            k = 0
        else:
            k += 1
            if k > 4:
                if len(obs) >= 3:
                    dis = 0
                    for zerg in obs:
                        dis += currentLidar[zerg]  
                    dis = dis / len(obs)
                    angle = ((obs[-1] + obs[0]) / 2 - 45) * pi / 180
                    r = dis * tan(((obs[-1] - obs[0])/2)*pi/180)
                    result.append([dis, angle, r])
                obs = []
    if len(obs) >= 3:
        dis = 0
        for zerg in obs:
            dis += currentLidar[zerg]  
        dis = dis / len(obs)
        angle = ((obs[-1] + obs[0]) / 2 - 45) * pi / 180
        r = dis * tan(((obs[-1] - obs[0])/2)*pi/180)
        result.append([dis, angle, r])
    return result









def modifyTheta(z):
    while z < -1 * pi or z > pi:
        if z < -1 * pi:
            z += 2*pi
        if z > pi:
            z -= 2*pi
    return z

def getClearObs(obsList):
    #the min-dis static obs
    result = [3., 0., 0.2]
    if len(obsList) >= 1:
        for item in obsList:
            if (item[0] - item[2]) <= (result[0] - result[2]):
                result = item
    if result[0] > 1.5:
        return None
    else:
        return result

def norm_sq(x):
    return dot(x, x)

def whetherToAvoid(state, obstacle):
    dt = 2
    x = array([obstacle[0]-state[0], obstacle[1]-state[1]])   
    Va = (state[3]*cos(state[2]), state[3]*sin(state[2]))
    Vobs = (obstacle[3]*cos(obstacle[2]), obstacle[3]*sin(obstacle[2]))
    v = array(Va) - array(Vobs)   
    r = state[4] + obstacle[4]
    if norm_sq(x) <= r*r:
        return True
    else:
        adjust_center = x/dt
        if norm_sq(v - adjust_center) <= r*r:
            return True
        
        x_len_sq = norm_sq(x)
        X = array(x)
        V = array(v)
        theta = atan2(V[1], V[0])
        theta = modifyTheta(theta)
        adj_theta = atan2(X[1], X[0])
        adj_theta = modifyTheta(adj_theta) 
        if abs(adj_theta - theta) <= asin(r/sqrt(x_len_sq)):
            return True
    return False

        




def calculateMovingObstacle(state, obstacle):
    OBS = [50, 4*pi, 0.4]
    rotation = modifyTheta(state[2])
    #print obstacle
    for item in obstacle:
        distance = sqrt((item[1]-state[1])**2 + (state[0] - item[0])**2)
        theta = atan2(item[1]-state[1], item[0] - state[0])
        o = modifyTheta(item[2])
        
        #FOV: theta and dinstance
        if abs(theta - rotation) <= math.pi/2 and distance <= 1.4:
            if whetherToAvoid(state, item):
                robot = [distance, theta-rotation, item[4]]
                if (theta - rotation) < OBS[1]:
                    OBS = robot
    if OBS[0] > 25:
        return None
    else:
        return OBS


def decisionForStatic(v, obstacle, r, theta):
    theta1 = obstacle[1]
    distance1 = obstacle[0]
    if abs(theta1 - pi/2 - theta) <= abs(theta1 + pi/2 - theta):
        Q = modifyTheta(theta1 - pi/2)
    else:
        Q = modifyTheta(theta1 + pi/2)
    dTheta1 = Q
    #a = (0.3 + obstacle[2] + r) * dTheta1 / distance1
    a = 0.32 * (0.3 + distance1 * tan(obstacle[2])) * dTheta1 / distance1
    av = v * (0.2 + distance1 * tan(obstacle[2])) / distance1
    return a, av

def decisionForMoving(v, otherRobot, r):
    theta1 = otherRobot[1]
    theta1 = modifyTheta(theta1)
    distance1 = otherRobot[0]
    Q = modifyTheta(theta1 - pi/2)
    dTheta1 = Q
    a = dTheta1
    #a = 2 * (r + otherRobot[2]) * dTheta1 / distance1  # 2(r_obs+r_self)
    #a = 2 * (r + otherRobot[2]) * dTheta1  # 2(r_obs+r_self)
    av = v * 0.2 / distance1  #r_self = 0.2
    return a, av

def OURS(state, obstacle, otherRobot, goal):
    rotation = modifyTheta(state[2])
    theta = modifyTheta(atan2(goal[1]-state[1], goal[0] - state[0]))
    distance = sqrt((goal[1]-state[1])**2 + (goal[0] - state[0])**2)
    dTheta = modifyTheta(theta - rotation)
    attr_v = 0.6 * min(distance / 6, 1)
    attr_a = dTheta
    push_v = 0
    push_a = 0
    
    staticObs = getClearObs(obstacle)
    #staticObs = None
    movingObs = calculateMovingObstacle(state, otherRobot)

    '''
    if staticObs == None and movingObs == None:
        v = min(max(attr_v, 0.3), 0.55)
        return v, attr_a
    elif staticObs == None and movingObs != None:
        push_v, push_a = decisionForMoving(attr_v, movingObs, state[4])
    elif staticObs != None and movingObs == None:
        push_v, push_a = decisionForStatic(attr_v, staticObs, state[4], theta)
    else:
        if (staticObs[0] - staticObs[2]) <= (movingObs[0] - movingObs[2]) and abs(staticObs[1] - movingObs[1]) >= pi/9:
            push_v, push_a = decisionForStatic(attr_v, staticObs, state[4], theta)
        else:
            push_v, push_a = decisionForMoving(attr_v, movingObs, state[4])
    '''
    if movingObs != None:
        theta1 = movingObs[1]
        theta1 = modifyTheta(theta1)
        distance1 = movingObs[0]
        Q = modifyTheta(theta1 - pi/2)
        dTheta1 = Q
        push_a +=  dTheta1
        #push_a +=  (0.2 + 0.2) * dTheta1 / distance1  # 2(r_obs+r_self)
        push_v += attr_v * (0.2) / distance1  #r_self = 0.2
        v = min(max(attr_v - push_v, 0.2), 0.5)
    elif staticObs != None:
        theta1 = staticObs[1]
        distance1 = staticObs[0]
        if abs(theta1 - pi/2) <= abs(theta1 + pi/2):
            Q = modifyTheta(theta1 - pi/2)
        else:
            Q = modifyTheta(theta1 + pi/2)
        dTheta1 = Q
        push_a += 0.8 * 0.4 * (0.3 + state[4] + distance1 * tan(staticObs[2])) * dTheta1 / distance1
        push_v += attr_v * (0.2 + distance1 * tan(staticObs[2])) / distance1
        v = min(max(attr_v - push_v, 0.2), 0.5)
    else:
        v = min(max(attr_v - push_v, 0.45), 0.5)



    '''
    #static avoid
    if len(obstacle) >= 1:
        theta1 = obstacle[0][1]
        distance1 = obstacle[0][0]
        if distance1 < 2:
            if abs(theta1 - pi/2) <= abs(theta1 + pi/2):
                Q = modifyTheta(theta1 - pi/2)
            else:
                Q = modifyTheta(theta1 + pi/2)
            dTheta1 = Q
            push_a = 0.8 * 0.4 * (0.3 + distance1 * tan(obstacle[0][2])) * dTheta1 / distance1
            push_v = attr_v * (0.2 + distance1 * tan(obstacle[0][2])) / distance1
    
    if otherRobot != None:
        theta1 = otherRobot[1]
        theta1 = modifyTheta(theta1)
        distance1 = otherRobot[0]
        Q = modifyTheta(theta1 - pi/2)
        dTheta1 = Q
        push_a =  dTheta1
        #push_a +=  (0.2 + 0.2) * dTheta1 / distance1  # 2(r_obs+r_self)
        push_v = attr_v * (0.2) / distance1  #r_self = 0.2
    '''
    
    
    if push_a != 0:
        a = push_a
    else:
        a = attr_a
    return v, a


def TestInSingle(state, obs, goal):
    rotation = modifyTheta(state[2])
    theta = modifyTheta(atan2(goal[1]-state[1], goal[0] - state[0]))
    distance = sqrt((goal[1]-state[1])**2 + (goal[0] - state[0])**2)
    dTheta = modifyTheta(theta - rotation)
    attr_v = 0.6 * min(distance / 6, 1)
    attr_a = dTheta
    push_v = 0
    push_a = 0
    
    staticObs = getClearObs(obs)
    if staticObs != None:
        theta1 = staticObs[1]
        distance1 = staticObs[0]
        if abs(theta1 - pi/2) <= abs(theta1 + pi/2):
            Q = modifyTheta(theta1 - pi/2)
        else:
            Q = modifyTheta(theta1 + pi/2)
        dTheta1 = Q
        push_a += 0.8 * 0.4 * (0.3 + distance1 * tan(staticObs[2])) * dTheta1 / distance1
        push_v += attr_v * (0.2 + distance1 * tan(staticObs[2])) / distance1

    v = min(max(attr_v - push_v, 0.45), 0.5)
    if push_a != 0:
        a = push_a
    else:
        a = attr_a
    return v, a


def TestOnlyRobot(state, otherRobot, goal):
    rotation = modifyTheta(state[2])
    theta = modifyTheta(atan2(goal[1]-state[1], goal[0] - state[0]))
    distance = sqrt((goal[1]-state[1])**2 + (goal[0] - state[0])**2)
    dTheta = modifyTheta(theta - rotation)
    attr_v = 0.6 * min(distance / 6, 1)
    attr_a = dTheta
    push_v = 0
    push_a = 0
    
    movingObs = calculateMovingObstacle(state, otherRobot)
    if movingObs != None:
        theta1 = movingObs[1]
        theta1 = modifyTheta(theta1)
        distance1 = movingObs[0]
        Q = modifyTheta(theta1 - pi/2)
        dTheta1 = Q
        push_a +=  dTheta1
        #push_a +=  (0.2 + 0.2) * dTheta1 / distance1  # 2(r_obs+r_self)
        push_v += attr_v * (0.2) / distance1  #r_self = 0.2
    
    v = min(max(attr_v - push_v, 0.2), 0.5)
    if push_a != 0:
        a = push_a
    else:
        a = attr_a
    return v, a
    



    

