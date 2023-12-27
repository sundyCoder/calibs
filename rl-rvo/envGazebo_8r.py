import rospy
import math
import time
import numpy as np
import cv2
import copy
import tf
import random
from math import sin, cos, atan2, pi, sqrt
from rvo import Rvo_inter

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
    D, L = 0.25, 0.46
    Quaternions = odometry.pose.pose.orientation
    euler = tf.transformations.euler_from_quaternion([Quaternions.x, Quaternions.y, Quaternions.z, Quaternions.w])
    if name == 'burger':
        r = 0.15
    else:
        r = 0.3
    vr = odometry.twist.twist.linear.x + 0.5*odometry.twist.twist.angular.z*L
    vl = 2*odometry.twist.twist.linear.x - vr
    v_x= (0.5*cos(euler[2])+D*sin(euler[2])/L)*vl +  (0.5*cos(euler[2])-D*sin(euler[2])/L)*vr
    v_y = (0.5*sin(euler[2])-D*cos(euler[2])/L)*vl +  (0.5*sin(euler[2])+D*cos(euler[2])/L)*vr
    return [odometry.pose.pose.position.x, odometry.pose.pose.position.y, v_x, v_y, r]

class GazeboWorld():
    def __init__(self, index, num_env):
        # initiliaze
        self.index = index
        self.num_env = num_env
        node_name = "ENV"+str(index)
        rospy.init_node(node_name, anonymous=False)
        self.D, self.L = 0.25, 0.46
        self.vel = np.zeros(2)
        self.acc = 1.0 # 1.0 in paper
        
        self.rvo = Rvo_inter(neighbor_region=4, neighbor_num=5, vxmax=0.5, vymax=0.5, acceler=self.acc)

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
        self.goal = None
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
        self.odom_sub4 = rospy.Subscriber('robot4/odom', Odometry, self.obstacleCallBack4)
        self.odom_sub5 = rospy.Subscriber('robot5/odom', Odometry, self.obstacleCallBack5)
        self.odom_sub6 = rospy.Subscriber('robot6/odom', Odometry, self.obstacleCallBack6)
        self.odom_sub7 = rospy.Subscriber('robot7/odom', Odometry, self.obstacleCallBack7)

        
        self.obstacle0 = Odometry()
        self.obstacle1 = Odometry()
        self.obstacle2 = Odometry()
        self.obstacle3 = Odometry()
        self.obstacle4 = Odometry()
        self.obstacle5 = Odometry()
        self.obstacle6 = Odometry()
        self.obstacle7 = Odometry()

        rospy.sleep(1.)
        rospy.on_shutdown(self.shutdown)
        
    def getObstaclePos(self):
        obstacle = []
        if self.index != 0:
            obstacle.append(processObs(self.obstacle0, 'polycar'))
        if self.index != 1:
            obstacle.append(processObs(self.obstacle1, 'polycar'))
        if self.index != 2:
            obstacle.append(processObs(self.obstacle2, 'waffle'))
        if self.index != 3:
            obstacle.append(processObs(self.obstacle3, 'burger'))
        if self.index != 4:
            obstacle.append(processObs(self.obstacle4, 'waffle'))
        if self.index != 5:
            obstacle.append(processObs(self.obstacle5, 'waffle'))
        if self.index != 6:
            obstacle.append(processObs(self.obstacle6, 'polycar'))
        if self.index != 7:
            obstacle.append(processObs(self.obstacle7, 'burger'))
        return obstacle
        
    def obstacleCallBack0(self, data):
        self.obstacle0 = data

    def obstacleCallBack1(self, data):
        self.obstacle1 = data

    def obstacleCallBack2(self, data):
        self.obstacle2 = data

    def obstacleCallBack3(self, data):
        self.obstacle3 = data

    def obstacleCallBack4(self, data):
        self.obstacle4 = data

    def obstacleCallBack5(self, data):
        self.obstacle5 = data
        
    def obstacleCallBack6(self, data):
        self.obstacle6 = data

    def obstacleCallBack7(self, data):
        self.obstacle7 = data

    def OdometryCallBack(self, odometry):
        quaternion = (odometry.pose.pose.orientation.x,
                      odometry.pose.pose.orientation.y,
                      odometry.pose.pose.orientation.z,
                      odometry.pose.pose.orientation.w)
        euler = tf.transformations.euler_from_quaternion(quaternion)
        if self.index in [3, 7]:
            r = 0.15  #buger
        else:
            r = 0.3
            
        vr = odometry.twist.twist.linear.x + 0.5*odometry.twist.twist.angular.z*self.L
        vl = 2*odometry.twist.twist.linear.x - vr
        v_x= (0.5*cos(euler[2])+self.D*sin(euler[2])/self.L)*vl +  (0.5*cos(euler[2])-self.D*sin(euler[2])/self.L)*vr
        v_y = (0.5*sin(euler[2])-self.D*cos(euler[2])/self.L)*vl +  (0.5*sin(euler[2])+self.D*cos(euler[2])/self.L)*vr
        self.state = [odometry.pose.pose.position.x, odometry.pose.pose.position.y, euler[2], v_x, v_y, r]
        self.orientation = euler[2]
        goal_pose = goal_point(self.index)
        self.gvel = np.array(goal_pose) - np.asarray([odometry.pose.pose.position.x, odometry.pose.pose.position.y]) # the goal velocity of the agent
        self.gvel = self.gvel/(sqrt(self.gvel.dot(self.gvel))) #*VEL_MAX
        
        self.STATE = [odometry.pose.pose.position.x, odometry.pose.pose.position.y, v_x, v_y, r, self.gvel[0], self.gvel[1]]

    def GetGoal(self):
        return self.goal

    def GetSelfState(self):
        return self.STATE
        
    def GetObstacle(self):
        return self.getObstaclePos()             # [ [x, y, vx, vy, r] ]

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
  
    def diff2omni(self, robot_state): # diff2omni
        v = robot_state[3]
        w = robot_state[4]
        theta = robot_state[2]
        vr = v + 0.5*w*self.L
        vl = 2*v - vr
        v_x= (0.5*cos(theta)+self.D*sin(theta)/self.L)*vl +  (0.5*cos(theta)-self.D*sin(theta)/self.L)*vr
        v_y = (0.5*sin(theta)-self.D*cos(theta)/self.L)*vl +  (0.5*sin(theta)+self.D*cos(theta)/self.L)*vr
        return [v_x, v_y]
    
    def omni2diff(self, pref_vel): # omni2diff
        theta = self.state[2]
        A = 0.5*cos(theta)+self.D*sin(theta)/self.L
        B = 0.5*cos(theta)-self.D*sin(theta)/self.L
        C = 0.5*sin(theta)-self.D*cos(theta)/self.L
        D = 0.5*sin(theta)+self.D*cos(theta)/self.L

        vx = pref_vel[0]
        vy = pref_vel[1]
        vr = (vy-C/A*vx)/(D-B*C/A)
        vl = (vx-B*vr)/A

        angular_z =  (vr-vl)/self.L
        angular_z= np.clip(angular_z, -ANGULAR_MAX, ANGULAR_MAX)

        linear_x =0.5*(vl+vr)
        linear_x = np.clip(linear_x, -VEL_MAX, VEL_MAX)
        return [linear_x, angular_z]


    def GetRewardAndTerminate(self, t, is_training = True):
        terminate, crash, result = False, False, "Run"
        robot_state = self.GetSelfState() # [x, y, vx, vy, r, vdx, vdy]
        goal_pos = self.GetGoal() # [x, y]
        nbr_state = self.GetObstacle()  # [ [x, y, vx, vy, r] ]

        for nbr in nbr_state:
            obsDis = np.linalg.norm(np.array([nbr[1] - robot_state[1], nbr[0] - robot_state[0]], dtype=np.float)) - robot_state[4] - nbr[4] + 0.17
            if obsDis <= 0:
                crash = True
                break

        re = 0

        self.distance = np.linalg.norm(np.array([goal_pos[1]- robot_state[1], goal_pos[0] - robot_state[0]], dtype=np.float))
        if self.distance <= self.goal_threshold: 
            terminate = True
            result = 'Reach'
            re = 15
        if t >= HORIZON - 1 and is_training:
            terminate = True
            result = 'Time out'
        if crash and is_training:
            terminate = True
            result = 'Crashed'
            re = -20
            
        return re, terminate, result
    

    def get_observation(self, action=np.zeros((2,))):
        robot_omni_state = self.GetSelfState()
        nei_state_list = self.GetObstacle()
        obs_vo_list, _, _, _ = self.rvo.config_vo_inf(robot_omni_state, nei_state_list, action)

        radian = wraptopi(self.orientation)
        cur_vel = robot_omni_state[2:4]
        radius = robot_omni_state[4]

        propri_obs = np.array([ robot_omni_state[2], robot_omni_state[3], robot_omni_state[5], robot_omni_state[6], radian, radius])     # (6, )
        
        if len(obs_vo_list) == 0:
            exter_obs = np.zeros((8,))
        else:
            exter_obs = np.concatenate(obs_vo_list) # vo list
  
        observation = np.round(np.concatenate([propri_obs, exter_obs]), 2)      # (6 + 8*a, )
        return observation
    
    def get_rvo_reward_v1(self, action=np.zeros((2,))):
        state = self.GetSelfState()
        nei_state_list = self.GetObstacle()
        vo_flag, min_exp_time, min_dis = self.rvo.config_vo_reward(state, nei_state_list, action)

        des_vel = np.round(np.squeeze(state[-2:]), 2)
        
        dis_des_reward = sqrt((action[0] - des_vel[0] )**2 + (action[1] - des_vel[1])**2)
        exp_time_reward = - 1./ (min_exp_time + 0.2) # (0-1)    
        rvo_reward = 0.
        # rvo reward    
        if vo_flag:
            rvo_reward = 0.3 + 1.2 * exp_time_reward        
            if min_exp_time <= 0.1:
                rvo_reward = 3.6 * exp_time_reward            
            if min_exp_time > 5:
                rvo_reward = 0.3 - dis_des_reward       
        else:
            rvo_reward = 0.3 - dis_des_reward

        rvo_reward = np.round(rvo_reward, 2)
        return rvo_reward
    
    def get_rvo_reward(self, action=np.zeros((2,))):
        state = self.GetSelfState()
        nei_state_list = self.GetObstacle()
        vo_flag, min_exp_time, min_dis = self.rvo.config_vo_reward(state, nei_state_list, action)

        des_vel = np.round(np.squeeze(state[-2:]), 2)
        dis_des = sqrt((action[0] - des_vel[0] )**2 + (action[1] - des_vel[1])**2)
        max_dis_des = 3
        dis_des_reward = - dis_des / max_dis_des #  (0-1)
        exp_time_reward = - 0.2/(min_exp_time+0.2) # (0-1)
        
        p1, p2, p3, p4, p5, p6, p7, p8 = (3.0, 0.3, 0.0, 6.0, 0.3, 3.0, 0, 0)

        # rvo reward    
        if vo_flag:
            rvo_reward = p2 + p3 * dis_des_reward + p4 * exp_time_reward
            
            if min_exp_time < 0.1:
                rvo_reward = p2 + p1 * p4 * exp_time_reward
        else:
            rvo_reward = p5 + p6 * dis_des_reward
        
        rvo_reward = np.round(rvo_reward, 2)
        return rvo_reward

        