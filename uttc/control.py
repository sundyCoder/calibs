#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
import csv
import tf
import time 
import math
import numpy as np
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge
from std_msgs.msg import String, Int8
from sensor_msgs.msg import Image, LaserScan
from geometry_msgs.msg import Twist, Point, Quaternion
from tf.transformations import euler_from_quaternion
from scipy.spatial import distance as dist
from math import radians, copysign, sqrt, pow, pi, atan2, cos, sin
from config import *


class TTC_Control():
    def __init__(self, turtlebot, end):
        self.turtlebot = "robot" + str(turtlebot)
        self.id = turtlebot
        self.file = 'tb' + str(turtlebot) + '.csv'
        self.goal = end
        self.ranges = []
        self.state = None
        rospy.init_node(self.turtlebot + '_control', anonymous=False)
        rospy.on_shutdown(self.shutdown)
        self.D, self.L = 0.25, 0.46
        self.vel = np.zeros(2)
        self.distance = 0
        
        self.position = Point()
        self.move_cmd = Twist()
        self.tf_listener = tf.TransformListener()
        self.odom_frame = self.turtlebot+'/odom'
        try:
            self.tf_listener.waitForTransform(self.odom_frame, self.turtlebot+'/base_footprint', rospy.Time(), rospy.Duration(1.0))
            self.base_frame = self.turtlebot+'/base_footprint'
        except (tf.Exception, tf.ConnectivityException, tf.LookupException):
            try:
                self.tf_listener.waitForTransform(self.odom_frame, self.turtlebot+'/base_link', rospy.Time(), rospy.Duration(1.0))
                self.base_frame = self.turtlebot+'/base_link'
            except (tf.Exception, tf.ConnectivityException, tf.LookupException):
                rospy.loginfo("Cannot find transform between odom and base_link or base_footprint")
                rospy.signal_shutdown("tf Exception")
        
        self.cmd_vel = rospy.Publisher(self.turtlebot+'/cmd_vel', Twist, queue_size=5)
        
        #self.ball_num.publish(self.ballNum)
        # self.scan_sub = rospy.Subscriber(self.turtlebot+'/scan', LaserScan, self.scanCallBack)
        self.odom_sub = rospy.Subscriber(self.turtlebot+'/odom', Odometry, self.odometry_callback)

        self.obstacle1 = Odometry()
        self.obstacle2 = Odometry()
        self.obstacle3 = Odometry()
        self.obstacle4 = Odometry()
        self.obstacle5 = Odometry()
        self.obstacle6 = Odometry()
        self.obstacle7 = Odometry()
        # self.obstacle8 = Odometry()
        # self.obstacle9 = Odometry()
        # self.obstacle10 = Odometry()
        # self.obstacle11 = Odometry()
        self.obstacle12 = Odometry()
        self.obstacle1_sub = rospy.Subscriber('robot1/odom', Odometry, self.obstacleCallBack1)
        self.obstacle2_sub = rospy.Subscriber('robot2/odom', Odometry, self.obstacleCallBack2)
        self.obstacle3_sub = rospy.Subscriber('robot3/odom', Odometry, self.obstacleCallBack4)
        self.obstacle5_sub = rospy.Subscriber('robot5/odom', Odometry, self.obstacleCallBack5)
        self.obstacle6_sub = rospy.Subscriber('robot6/odom', Odometry, self.obstacleCallBack6)
        self.obstacle7_sub = rospy.Subscriber('robot7/odom', Odometry, self.obstacleCallBack7)
        # self.obstacle8_sub = rospy.Subscriber('robot8/odom', Odometry, self.obstacleCallBack8)
        # self.obstacle9_sub = rospy.Subscriber('robot9/odom', Odometry, self.obstacleCallBack9)
        # self.obstacle10_sub = rospy.Subscriber('robot10/odom', Odometry, self.obstacleCallBack10)
        # self.obstacle11_sub = rospy.Subscriber('robot11/odom', Odometry, self.obstacleCallBack11)
        self.obstacle12_sub = rospy.Subscriber('robot0/odom', Odometry, self.obstacleCallBack12)


    def get_obstacle_pos(self):
        obstacle = []
        if self.turtlebot != 'robot1':
            obstacle.append(self.processObs(self.obstacle1, 'waffle'))
        if self.turtlebot != 'robot2':
            obstacle.append(self.processObs(self.obstacle2, 'burger'))
        if self.turtlebot != 'robot3':
            obstacle.append(self.processObs(self.obstacle3, 'polycar'))
        if self.turtlebot != 'robot4':
            obstacle.append(self.processObs(self.obstacle4, 'waffle'))
        if self.turtlebot != 'robot5':
            obstacle.append(self.processObs(self.obstacle5, 'burger'))
        if self.turtlebot != 'robot6':
            obstacle.append(self.processObs(self.obstacle6, 'polycar'))
        if self.turtlebot != 'robot7':
            obstacle.append(self.processObs(self.obstacle7, 'waffle'))
        # if self.turtlebot != 'robot8':
        #     obstacle.append(self.processObs(self.obstacle8, 'burger'))
        # if self.turtlebot != 'robot9':
        #     obstacle.append(self.processObs(self.obstacle9, 'polycar'))
        # if self.turtlebot != 'robot10':
        #     obstacle.append(self.processObs(self.obstacle10, 'waffle'))
        # if self.turtlebot != 'robot11':
        #     obstacle.append(self.processObs(self.obstacle11, 'burger'))
        if self.turtlebot != 'robot0':
            obstacle.append(self.processObs(self.obstacle12, 'polycar'))
        if len(obstacle) >= 1:
            return obstacle
        else:
            rospy.loginfo("obstacle gotten failed!")
            return None


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

    def obstacleCallBack8(self, data):
        self.obstacle8 = data

    def obstacleCallBack9(self, data):
        self.obstacle9 = data

    def obstacleCallBack10(self, data):
        self.obstacle10 = data

    def obstacleCallBack11(self, data):
        self.obstacle11 = data

    def obstacleCallBack12(self, data):
        self.obstacle12 = data

    def odometry_callback(self, odometry):
        Quaternions = odometry.pose.pose.orientation
        Euler = tf.transformations.euler_from_quaternion([Quaternions.x, Quaternions.y, Quaternions.z, Quaternions.w])
        r = 0.32
        # if self.id in [2, 5]: # burger
        #     r = 0.32
        # elif self.id in [1, 4]: # waffle
        #     r = 0.32
        # elif self.id in [0, 3]: # polycar
        #     r = 0.32
        self.state = [odometry.pose.pose.position.x, odometry.pose.pose.position.y, Euler[2], odometry.twist.twist.linear.x, odometry.twist.twist.angular.z, r]
        self.gvel = np.asarray(self.goal) - np.asarray(self.state[0:2]) # the goal velocity of the agent
        self.gvel = self.gvel/(sqrt(self.gvel.dot(self.gvel )))*VEL_MAX


    def scanCallBack(self, data):
        #ball's height is too small, cannot be detected
        self.ranges = data.ranges
    

    def processObs(self, odometry, name):
        Quaternions = odometry.pose.pose.orientation
        Euler = tf.transformations.euler_from_quaternion([Quaternions.x, Quaternions.y, Quaternions.z, Quaternions.w])
        r = 0.32
        # if name == 'burger':
        #     r = 0.32
        # elif name == 'waffle':
        #     r = 0.32
        # elif name == 'polycar':
        #     r = 0.32
        return [odometry.pose.pose.position.x, odometry.pose.pose.position.y, Euler[2], odometry.twist.twist.linear.x, odometry.twist.twist.angular.z, r]

    def get_neighbor_obstacle(self, obs, state):
        result = []
        for item in obs:
            dis = sqrt((item[0] - state[0])**2+(item[1] - state[1])**2)
            # theta = atan2(item[1]-state[1], item[0] - state[0])
            # if abs(theta - state[2]) <= math.pi/2.5 and dis <= 1.0:
            if dis <= 1.0:
                result.append(item)
        return result

    
    def moveVel(self, linear_vel, angular_vel):
        self.move_cmd.linear.x = linear_vel
        self.move_cmd.angular.z = angular_vel
        self.cmd_vel.publish(self.move_cmd)


    def move_front(self, distance):
        r = rospy.Rate(2)
        (pos, rot) = self.get_odom()
        x_start = pos.x
        y_start = pos.y
        pasex = x_start
        pasey = y_start
        while sqrt(pow((pasex - x_start), 2) + pow((pasey - y_start), 2)) <= (distance - 0.1):
            (position, rotation) = self.get_odom()
            linear_speed = VEL_MAX

            pasex = position.x
            pasey = position.y
            self.move_cmd.linear.x = linear_speed 
            self.move_cmd.angular.z = 0
            self.cmd_vel.publish(self.move_cmd)
            r.sleep()
        self.move_cmd.linear.x = 0.
        self.cmd_vel.publish(self.move_cmd)
    
    
    def get_odom(self):
        #get robot's pose and position
        try:
            (trans, rot) = self.tf_listener.lookupTransform(self.odom_frame, self.base_frame, rospy.Time(0))
            rotation = euler_from_quaternion(rot)

        except (tf.Exception, tf.ConnectivityException, tf.LookupException):
            rospy.loginfo("TF Exception")
            return

        return (Point(*trans), rotation[2])


    def shutdown(self):
        self.cmd_vel.publish(Twist())
        rospy.sleep(1)

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

    
    def cal_des_vel(self):
        self.gvel = np.asarray(self.goal) - np.asarray(self.state[0:2])
        distGoalSq = self.gvel.dot(self.gvel)
        if distGoalSq > 0.2:
            self.gvel = np.round(self.gvel/sqrt(distGoalSq)*VEL_MAX, 2)
        return [self.gvel[0], self.gvel[1]]

    
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

    def compute_ttc_force(self, nbr_list):
        # TODO: parameters
        ksi, time_horizon, max_force, ttc_fin = 0.5, 8, 10, np.inf # TODO: max_force

        des_vel = np.round(self.cal_des_vel(), 2)
        action =  np.round(self.diff2omni(self.state), 2)
        goal_force = (des_vel - action) / ksi

        avoid_force = np.zeros_like(goal_force)
        nbr_list = self.convert_nei_state(nbr_list)
        robot_state = self.convert_self_state()
        # print("len: {}".format(len(nbr_list)))
        for nb in nbr_list:
            obj_sense_dist = dist.euclidean(robot_state[0:2], nb[0:2]) - (robot_state[-1] + nb[-1]) # x, y, o, vx, vy, r
            if 0 < obj_sense_dist < 10: # TODO: bugs ?
                ttc_val = self.ttc(robot_state, nb)
                if ttc_val != ttc_fin:
                    if ttc_val < ttc_fin: # This will compare all the ttc values within the neighbors
                        ttc_fin = ttc_val
                        rel_pos = np.array(robot_state[0:2]) - np.array(nb[0:2])
                        rel_vel = np.array(robot_state[3:5]) - np.array(nb[3:5])
                        n = (rel_pos + (ttc_fin*rel_vel)) / (np.linalg.norm(rel_pos + (ttc_fin*rel_vel))) # TODO: unit vector
                        f_avoid = (max((time_horizon - ttc_fin) , 0) / (ttc_fin + 1e-6))*n
                        avoid_force += f_avoid
        total_force = goal_force + avoid_force # goal force + collision force
        #print("ID: {}, af: {}".format(self.id, avoid_force))
        F = np.clip(total_force, -max_force, max_force) # Clipping the vector components to a magnitude of 10
        return F
    
    def generate_action(self, obstacle):
        force = self.compute_ttc_force(obstacle)
        self.vel += force * dt
        self.vel = np.clip(self.vel, -VEL_MAX, VEL_MAX) # Clipping the maximum velocity components to maxspeed
        vel = np.array([[self.vel[0]], [self.vel[1]]])
        action = self.omni2diff(vel)
        return action
    

    def ttc_run(self, rank):
        self.time = rospy.get_rostime()
	rospy.sleep(1.0)
        (self.prePos, _) = self.get_odom()
        traj_file = open("tb_"+str(rank)+".csv","a+")
	self.move_front(0.1)
        print("after move front")


        while True and not rospy.is_shutdown():
            (pos, rot) = self.get_odom()
            self.distance += sqrt((self.state[0]-self.prePos.x)**2 + (self.state[1]-self.prePos.y)**2)
            if sqrt((pos.x - self.goal[0])**2 + (pos.y - self.goal[1])**2) <= 0.2:
                print "id=",  rank, "dis=", self.distance, "time=", rospy.get_rostime() - self.time
                with open('log.csv', "a+") as file:
                    csv_file = csv.writer(file)
                    data = [rank, round(self.distance, 3), rospy.get_rostime() - self.time]
                    csv_file.writerow(data)
                break
            obs = self.get_obstacle_pos()
            obstacle = self.get_neighbor_obstacle(obs, self.state)
            v, w = self.generate_action(obstacle)
            self.moveVel(v, w)
            self.prePos = pos

            csv_file = csv.writer(traj_file)
            data = [round(pos.x, 3), round(pos.y, 3)]
            csv_file.writerow(data)

        self.moveVel(0, 0)

if __name__ == "__main__":
    k = TTC_Control(3, [3.5, 0])
    k.ttc_run()
    rospy.spin()
