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

random_6r_start = [[0.1, 4.9, -2.4], [7.01, 3.58, -2.14], [7.32, 1.06, 0.5], [0.57, 4.32, -1.52], [5.5, 4.5, -2.8], [4.56, 1.78, -2.06]]
random_6r_goal = [[0.08, 4.03], [3.65, 8.68], [0.39, 9.88], [8.49, 9.34], [5.58, 3.11], [7.92, 2.83]]



def init_pose(index):
    #start_pos = create_circle_init_goal(index)[0]
    #start_pos = swap_6r_start[index]
    start_pos = create_random_init_goal(index)[0]
    #start_pos = cross_8r_start[index]
    return start_pos

def goal_point(index):
    #goal_pose = create_circle_init_goal(index)[1]
    #goal_pose = swap_6r_goal[index]
    goal_pose = create_random_init_goal(index)[1]
    #goal_pose = cross_8r_goal[index]
    return goal_pose

    
def processObs(odometry, name):
    Quaternions = odometry.pose.pose.orientation
    Euler = tf.transformations.euler_from_quaternion([Quaternions.x, Quaternions.y, Quaternions.z, Quaternions.w])
    if name == 'burger':
        r = 0.15
    else:
        r = 0.3
    return [odometry.pose.pose.position.x, odometry.pose.pose.position.y, Euler[2], odometry.twist.twist.linear.x, odometry.twist.twist.angular.z, r]

class GazeboWorld():
    def __init__(self, index, num_env):
        # initiliaze
        self.index = index
        self.num_env = num_env
        node_name = "ENV"+str(index)
        rospy.init_node(node_name, anonymous=False)
        self.D, self.L = 0.25, 0.46
        self.vel = np.zeros(2)

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
        self.state = [odometry.pose.pose.position.x, odometry.pose.pose.position.y, euler[2], odometry.twist.twist.linear.x, odometry.twist.twist.angular.z, r]

        goal_pose = goal_point(self.index)
        self.gvel = np.array(goal_pose) - np.asarray([odometry.pose.pose.position.x, odometry.pose.pose.position.y]) # the goal velocity of the agent
        self.gvel = self.gvel/(sqrt(self.gvel.dot(self.gvel))) #*VEL_MAX

    def GetGoal(self):
        return self.goal
        
    def GetLocalGoal(self):
        [x, y, theta, v, w, r] = self.state
        [goal_x, goal_y] = self.goal
        local_x = (goal_x - x) * np.cos(theta) + (goal_y - y) * np.sin(theta)
        local_y = -(goal_x - x) * np.sin(theta) + (goal_y - y) * np.cos(theta)
        return [local_x, local_y]

    def GetSelfState(self):
        return self.state
        
    def GetObstacle(self):
        return self.getObstaclePos()

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

    
    def cal_des_vel(self):
        self.gvel = np.asarray(self.goal) - np.asarray(self.state[0:2])
        distGoalSq = self.gvel.dot(self.gvel)
        if distGoalSq > 0.2:
            self.gvel = np.round(self.gvel/sqrt(distGoalSq)*VEL_MAX, 2)
        return [self.gvel[0], self.gvel[1]]

    def relative(self, state1, state2):
        dif = np.array(state2[0:2]) - np.array(state1[0:2])
        dis = np.linalg.norm(dif)
        radian = atan2(dif[1], dif[0])
        return dis, radian

    
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
        angular_z= np.clip(angular_z[0], -ANGULAR_MAX, ANGULAR_MAX)

        linear_x =0.5*(vl+vr)
        linear_x = np.clip(linear_x[0], -VEL_MAX, VEL_MAX)
        return [linear_x, angular_z]

    
    def convert_nei_state(self, nbr_list):
        state_list = []
        for nbr in nbr_list:
            omni_vel = self.diff2omni(nbr)
            state = nbr[0:3] + omni_vel + [nbr[5]] # x, y, o, vx, vy, r
            state_list.append(state)
        return state_list

    def convert_self_state(self):
        omni_vel = self.diff2omni(self.state)
        state = self.state[0:3] + omni_vel + [self.state[5]]
        return state

    def ttc(self, rb, nb):
        vel_uncertain = 0.2 # The velocity uncertainty constant
        rad = rb[-1] + nb[-1]
        w = np.array(rb[0:2]) - np.array(nb[0:2])
        c = np.dot(w, w) - (rad*rad)
        if c < 0: # agents are colliding
            return 0
        v = np.array(rb[3:5]) - np.array(nb[3:5])
        a = np.dot(v,v) #- vel_uncertain**2 # vel_uncertainty^2
        b = np.dot(w,v) #- (rad*vel_uncertain)
        if b > 0:
            return np.inf
        discr = (b*b) - (a*c)
        if discr <= 0:
            return np.inf
        tau = c / (-b + np.sqrt(discr))
        if tau < 0:
            return np.inf
        return tau

    def ttc_reward_cal(self, nbr_list):
        # TODO: parameters
        ksi, time_horizon, max_force, ttc_fin = 0.5, 5, 10, np.inf # TODO: max_force

        des_vel = np.round(self.cal_des_vel(), 2)
        action =  np.round(self.diff2omni(self.state), 2)
        goal_force = (des_vel - action) / ksi

        avoid_force = np.zeros_like(goal_force)
        nbr_list = self.convert_nei_state(nbr_list)
        robot_state = self.convert_self_state()
        # print("len: {}".format(len(nbr_list)))
        for nb in nbr_list:
            obj_sense_dist = dist.euclidean(robot_state[0:2], nb[0:2]) - (robot_state[-1] + nb[-1]) # x, y, o, vx, vy, r
            if 0 < obj_sense_dist < 3: # TODO: bugs: 1 for swarp, 5 for circle and random
                ttc_val = self.ttc(robot_state, nb)
                if ttc_val != ttc_fin:
                    if ttc_val < ttc_fin: # This will compare all the ttc values within the neighbors
                        ttc_fin = ttc_val
                        rel_pos = np.array(robot_state[0:2]) - np.array(nb[0:2])
                        rel_vel = np.array(robot_state[3:5]) - np.array(nb[3:5])
                        n = (rel_pos + (ttc_fin*rel_vel)) / (np.linalg.norm(rel_pos + (ttc_fin*rel_vel))) # TODO: unit vector
                        f_avoid = (max((time_horizon - ttc_fin) , 0) / (ttc_fin + 1e-6))*n
                        avoid_force += f_avoid
        avoid_force = np.clip(avoid_force, -max_force, max_force) # Clipping the vector components to a magnitude of 10

        col_reward = -np.round(np.abs(avoid_force).sum(), 2)
        goal_reward = -np.round(np.abs(goal_force).sum(), 2)
        # [alpha1, alpha2] = softmax([col_reward, goal_reward])
        # print(f"alhpa1:{alpha1}, alpha2:{alpha2}")

        alpha1, alpha2 = 0.28, 0.1
        ttc_reward = col_reward*alpha1 + goal_reward*alpha2
        ttc_reward = np.round(ttc_reward, 2)
        # print(col_reward*alpha1, goal_reward*alpha2, ttc_reward)
        return ttc_reward

    def GetRewardAndTerminate(self, t, is_training = True):
        terminate, crash, result = False, False, "Run"
        robot_state = self.GetSelfState() # [x, y, o, v, w, r]
        goal_pos = self.GetGoal() # [x, y]
        nbr_state = self.GetObstacle()  # neighbouring state: 

        for nbr in nbr_state:
            obsDis = np.linalg.norm(np.array([nbr[1] - robot_state[1], nbr[0] - robot_state[0]], dtype=np.float)) - robot_state[5] - nbr[5] + 0.17
            if obsDis <= 0:
                crash = True
                break

        self.distance = np.linalg.norm(np.array([goal_pos[1]- robot_state[1], goal_pos[0] - robot_state[0]], dtype=np.float))
        if self.distance <= self.goal_threshold: 
            terminate = True
            result = 'Reach'
        if t > HORIZON and is_training:
            terminate = True
            result = 'Time out'
        if crash and is_training:
            terminate = True
            result = 'Crashed'
        re = -0.01
        ttc_reward = self.ttc_reward_cal(nbr_state) + re
        return ttc_reward, terminate, result
        
