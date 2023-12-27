import rospy
import math
import time
import numpy as np
import cv2
import copy
import tf
import random
from math import sin, cos, atan2, pi, sqrt

from geometry_msgs.msg import Twist
from geometry_msgs.msg import Quaternion, Pose
from gazebo_msgs.msg import ModelStates
from gazebo_msgs.msg import ModelState
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from std_srvs.srv import Empty
from config import *
from random import randint
from scipy.spatial import distance as dist
from scenario import create_random_init_goal, create_circle_init_goal

# 1, 2, 3, 4, 5, 0
swap_6r_start = [[-1.5, -0.9, 0],[1.5, 0.9, 3.14], [1.5, 0, 3.14], [1.5, -0.9, 3.14], [-1.5, 0.9, 0], [-1.5, 0, 0]]
swap_6r_goal =  [[1.5, -0.85], [-1.5, 0.85], [-1.5, 0], [-1.5, -0.85], [1.5, 0.85], [1.5, 0]]

cross_8r_start = [[2.9, 0.25, 3.06], [2.9, -1.26,3.06], [1.0,-3.25, 1.6], [-0.31,-3.24, 1.6], [-2.7,-1.25, 0], [-2.7, 0.21, 0], [-0.28, 2.47, -1.6], [1.0, 2.47, -1.6]]
cross_8r_goal = [[-2.7, 0.21], [-2.7, -1.25], [1.0, 2.47], [-0.28, 2.47], [2.9, -1.26],[2.9, 0.25], [-0.31, -3.24], [1.0, -3.25]]

random_8r_start = [[1.99, -2.63, -2.42], [1.57, 1.63, -1.72], [-1.56, 2.8, -3.08], [0.60, -1.11, 2.94], [-0.84, -0.03, 1.56], [0.13, -4.39, -1.71], [-2.73, -0.89, 2.11], [-2.55, -3.31, 2.22]]
random_8r_goal = [[-2.73, -0.89], [0.13, -4.39], [-2.55, -3.31], [-0.84, -0.03], [0.60, -1.11], [1.57, 1.63], [1.99, -2.63], [-1.56, 2.8]]

def wraptopi(theta):

    if theta > pi:
        theta = theta - 2*pi
    
    if theta < -pi:
        theta = theta + 2*pi

    return theta

def init_pose(index):
    start_pos = create_circle_init_goal(index)[0]
    #start_pos = swap_6r_start[index]
    #start_pos = create_random_init_goal(index)[0]
    return start_pos

def goal_point(index):
    goal_pose = create_circle_init_goal(index)[1]
    #goal_pose = swap_6r_goal[index]
    #goal_pose = create_random_init_goal(index)[1]
    return goal_pose

    
def processObs(odometry, name):
    Quaternions = odometry.pose.pose.orientation
    euler = tf.transformations.euler_from_quaternion([Quaternions.x, Quaternions.y, Quaternions.z, Quaternions.w])
    if name == 'burger':
        r = 0.15
    else:
        r = 0.3
    return [odometry.pose.pose.position.x, odometry.pose.pose.position.y, euler[2], odometry.twist.twist.linear.x, odometry.twist.twist.angular.z, r]

class GazeboWorld():
    def __init__(self, index, num_env):
        # initiliaze
        self.index = index
        self.num_env = num_env
        node_name = "ENV"+str(index)
        rospy.init_node(node_name, anonymous=False)    

        self.set_self_state = ModelState()
        self.set_self_state.model_name = 'robot'+str(index)
        self.set_self_state.pose.position.x = init_pose(self.index)[0]
        self.set_self_state.pose.position.y = init_pose(self.index)[1]
        self.set_self_state.pose.position.z = 0.
        quaternion = tf.transformations.quaternion_from_euler(0., 0., init_pose(self.index)[2])
        self.set_self_state.pose.orientation.x = quaternion[0]
        self.set_self_state.pose.orientation.y = quaternion[1]
        self.set_self_state.pose.orientation.z = quaternion[2]
        self.set_self_state.pose.orientation.w = quaternion[3]
        self.set_self_state.twist.linear.x = 0.
        self.set_self_state.twist.linear.y = 0.
        self.set_self_state.twist.linear.z = 0.
        self.set_self_state.twist.angular.x = 0.
        self.set_self_state.twist.angular.y = 0.
        self.set_self_state.twist.angular.z = 0.
        self.set_self_state.reference_frame = 'world'

        self.self_speed = [0.0, 0.0]  #v, w        
        self.state = None
        self.goal = goal_point(self.index)
        self.obstacle = None
        self.goal_threshold = 0.2 + 0.15

        self.cmd_vel = rospy.Publisher('robot'+str(index)+'/cmd_vel', Twist, queue_size=10)
        self.odom_sub = rospy.Subscriber('robot'+str(index)+'/odom', Odometry, self.OdometryCallBack)
        self.cmd_pose = rospy.Publisher('robot'+str(index)+'/cmd_pose', Pose, queue_size=2)
        #self.reset_stage = rospy.ServiceProxy('reset_positions', Empty)
        self.set_state = rospy.Publisher('gazebo/set_model_state', ModelState, queue_size=10)
        
        self.odom_sub0 = rospy.Subscriber('robot0/odom', Odometry, self.obstacleCallBack0)
        self.odom_sub1 = rospy.Subscriber('robot1/odom', Odometry, self.obstacleCallBack1)
        self.odom_sub2 = rospy.Subscriber('robot2/odom', Odometry, self.obstacleCallBack2)
        self.odom_sub3 = rospy.Subscriber('robot3/odom', Odometry, self.obstacleCallBack3)

        
        self.obstacle0 = Odometry()
        self.obstacle1 = Odometry()
        self.obstacle2 = Odometry()
        self.obstacle3 = Odometry()
        rospy.sleep(1.)
        rospy.on_shutdown(self.shutdown)

    def getObstaclePos(self):
        obstacle = []
        if self.index != 0:
            obstacle.append(processObs(self.obstacle0, 'burger'))
        if self.index != 1:
            obstacle.append(processObs(self.obstacle1, 'waffle'))
        if self.index != 2:
            obstacle.append(processObs(self.obstacle2, 'burger'))
        if self.index != 3:
            obstacle.append(processObs(self.obstacle3, 'waffle'))
        return obstacle
        
    def obstacleCallBack0(self, data):
        self.obstacle0 = data

    def obstacleCallBack1(self, data):
        self.obstacle1 = data

    def obstacleCallBack2(self, data):
        self.obstacle2 = data

    def obstacleCallBack3(self, data):
        self.obstacle3 = data


    def OdometryCallBack(self, odometry):
        quaternion = (odometry.pose.pose.orientation.x,
                    odometry.pose.pose.orientation.y,
                    odometry.pose.pose.orientation.z,
                    odometry.pose.pose.orientation.w)
        euler = tf.transformations.euler_from_quaternion(quaternion)
        if self.index in [0, 2]:
            r = 0.15  #buger
        else:
            r = 0.3      
        self.state = [odometry.pose.pose.position.x, odometry.pose.pose.position.y, euler[2], odometry.twist.twist.linear.x, odometry.twist.twist.angular.z, r, self.goal[0], self.goal[1]]

    def GetGoal(self):
        return self.goal

    def GetSelfState(self):
        return self.state
        
    def GetObstacle(self):
        temp = self.getObstaclePos()             # [ [x, y, vx, vy, r] ]
        obstacle = []
        for item in temp:
            if sqrt((self.state[0] - item[0])**2 + (self.state[1] - item[1])**2) < 3.0:
                obstacle.append(item)
        return obstacle

    def ResetWorld(self):
        #self.reset_stage()
        self.start_time = time.time()
        self.self_speed = [0.0, 0.0]
        rospy.sleep(0.5)


    def reset_pose(self):
        self.set_state.publish(self.set_self_state)
        getPose = init_pose(self.index)
        self.control_pose(getPose)
        self.goal = goal_point(self.index)
        self.pre_distance = np.sqrt((getPose[0] - self.goal[0]) ** 2 + (getPose[1] - self.goal[1]) ** 2)
        self.distance = copy.deepcopy(self.pre_distance)
        rospy.sleep(0.1)

    def control_pose(self, pose):
        pose_cmd = Pose()
        assert len(pose)==3
        pose_cmd.position.x = pose[0]
        pose_cmd.position.y = pose[1]
        pose_cmd.position.z = 0
        qtn = tf.transformations.quaternion_from_euler(0, 0, pose[1], 'rxyz')
        pose_cmd.orientation.x = qtn[0]
        pose_cmd.orientation.y = qtn[1]
        pose_cmd.orientation.z = qtn[2]
        pose_cmd.orientation.w = qtn[3]
        self.cmd_pose.publish(pose_cmd)

    def Control(self, v, a):
        move_cmd = Twist()
        move_cmd.linear.x = v
        move_cmd.linear.y = 0.
        move_cmd.linear.z = 0.
        move_cmd.angular.x = 0.
        move_cmd.angular.y = 0.
        move_cmd.angular.z = a
        self.cmd_vel.publish(move_cmd)

    def shutdown(self):
        rospy.loginfo("Stop Moving")
        self.cmd_vel.publish(Twist())
        rospy.sleep(1)


    def GetRewardAndTerminate(self, t, is_training = True):
        terminate, crash, result = False, False, "Run"
        robot_state = self.GetSelfState() # [x, y, vx, vy, r, vdx, vdy]
        goal_pos = self.GetGoal() # [x, y]
        nbr_state = self.GetObstacle()  # [ [x, y, vx, vy, r] ]

        for nbr in nbr_state:
            obsDis = np.linalg.norm(np.array([nbr[1] - robot_state[1], nbr[0] - robot_state[0]], dtype=np.float)) - robot_state[5] - nbr[5] + 0.12
            if obsDis <= 0:
                crash = True
                break

        self.distance = np.linalg.norm(np.array([goal_pos[1]- robot_state[1], goal_pos[0] - robot_state[0]], dtype=np.float))
        rd = 2*(self.pre_distance**2 - self.distance**2)
        rc = 0
        re = -0.01
        info = {'arrived':False, 'crashed':False}
        if self.distance <= self.goal_threshold: 
            terminate = True
            result = 'Reach'
            rd = 20
            info['arrived'] = True
        if t > HORIZON - 1 and is_training:
            terminate = True
            result = 'Time out'
        if crash and is_training:
            terminate = True
            result = 'Crashed'
            rc = -15
            info['crashed'] = True

        reward = rd + rc + re
        self.pre_distance = self.distance    
        return reward, terminate, result, info
    

    def get_observation(self):
        robot_omni_state = self.GetSelfState()
        nei_state_list = self.GetObstacle()

        propri_obs = np.array(robot_omni_state)     # (8, )
        
        if len(nei_state_list) == 0:
            exter_obs = np.zeros((6,))
        else:
            exter_obs = np.concatenate(nei_state_list) # vo list
  
        observation = np.round(np.concatenate([propri_obs, exter_obs]), 2)      # (8 + 6*a, )
        return observation
    

        
