#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
import csv
import tf
import time 
import numpy as np
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist, Point, Quaternion
from tf.transformations import euler_from_quaternion
from scipy.spatial import distance as dist
from math import sqrt, pow, pi, atan2, cos, sin
from opt_QCQP import OptQCQP
from config import *


class QCQP_Control():
    def __init__(self, turtlebot, end):
        self.turtlebot = "robot" + str(turtlebot)
        self.id = turtlebot
        self.file = 'tb' + str(turtlebot) + '.csv'
        self.goal = end
        self.ranges = []
        self.state = None
        rospy.init_node(self.turtlebot + '_control', anonymous=False)
        rospy.on_shutdown(self.shutdown)
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
        self.obstacle8 = Odometry()
        self.obstacle9 = Odometry()
        self.obstacle10 = Odometry()
        self.obstacle11 = Odometry()
        self.obstacle12 = Odometry()

        self.obstacle1_sub = rospy.Subscriber('robot1/odom', Odometry, self.obstacleCallBack1)
        self.obstacle2_sub = rospy.Subscriber('robot2/odom', Odometry, self.obstacleCallBack2)
        self.obstacle3_sub = rospy.Subscriber('robot3/odom', Odometry, self.obstacleCallBack3)
        self.obstacle4_sub = rospy.Subscriber('robot4/odom', Odometry, self.obstacleCallBack4)
        self.obstacle5_sub = rospy.Subscriber('robot5/odom', Odometry, self.obstacleCallBack5)
        self.obstacle6_sub = rospy.Subscriber('robot6/odom', Odometry, self.obstacleCallBack6)
        self.obstacle7_sub = rospy.Subscriber('robot7/odom', Odometry, self.obstacleCallBack7)
        self.obstacle8_sub = rospy.Subscriber('robot8/odom', Odometry, self.obstacleCallBack8)
        self.obstacle9_sub = rospy.Subscriber('robot9/odom', Odometry, self.obstacleCallBack9)
        self.obstacle10_sub = rospy.Subscriber('robot10/odom', Odometry, self.obstacleCallBack10)
        self.obstacle11_sub = rospy.Subscriber('robot11/odom', Odometry, self.obstacleCallBack11)
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
        if self.turtlebot != 'robot8':
            obstacle.append(self.processObs(self.obstacle8, 'burger'))
        if self.turtlebot != 'robot9':
            obstacle.append(self.processObs(self.obstacle9, 'polycar'))
        if self.turtlebot != 'robot10':
            obstacle.append(self.processObs(self.obstacle10, 'waffle'))
        if self.turtlebot != 'robot11':
            obstacle.append(self.processObs(self.obstacle11, 'burger'))
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
        # return [odometry.pose.pose.position.x, odometry.pose.pose.position.y, Euler[2], odometry.twist.twist.linear.x, odometry.twist.twist.angular.z, r]
        return [odometry.pose.pose.position.x, odometry.pose.pose.position.y, r]

    def get_neighbor_obstacle(self, obs, state):
        result = []
        for item in obs:
            dis = sqrt((item[0] - state[0])**2+(item[1] - state[1])**2)
            # theta = atan2(item[1]-state[1], item[0] - state[0])
            # if abs(theta - state[2]) <= math.pi/2.5 and dis <= 1.0:
            # if dis <= 1.0:
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
        # get robot's pose and position
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
    
    def getPos(self):
        pos = [self.state[0], self.state[1]]
        return pos
    
    def done(self):
        if np.linalg.norm(np.array(self.getPos()) - np.array(self.goal)) <= 0.25:
            return True
        else:
            return False

    
    def omni2diff(self, pref_vel): # omni2diff
        self.D, self.L = 0.25, 0.46
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


    def generate_action(self, solver, dt, obstacle, state):
        solver.state_update(np.array([state[0], state[1], state[2]]), np.array(obstacle)[:,0:3])
        flag, vx, vy = solver.solve()
        if flag:
            [v, w] = self.omni2diff([vx, vy])
            return [v, w]
        else:
            return [0, 0] #[state[3], state[4]]
    

    def robot_run(self, dt):
        self.time = rospy.get_rostime()
        rospy.sleep(dt)
        (self.prePos, _) = self.get_odom()
        traj_file = open("tb_"+str(self.id)+".csv","a+")
        self.move_front(0.5)
        solver = OptQCQP(self.id, dt, len(start_state)) # id, dt, num_robots
        count = 0
        while True and not rospy.is_shutdown():
            count += 1
            (pos, rot) = self.get_odom()
            self.distance += sqrt((self.state[0]-self.prePos.x)**2 + (self.state[1]-self.prePos.y)**2)
            if self.done():
                travel_time = rospy.get_rostime().to_sec() - self.time.to_sec()
                print "count:", count, "ID = ", self.id, "dis=", self.distance, "time=", travel_time
                with open('log.csv', "a+") as file:
                    csv_file = csv.writer(file)
                    data = [self.id, round(self.distance, 3), travel_time]
                    csv_file.writerow(data)
                break
            obs = self.get_obstacle_pos()
            obstacle = self.get_neighbor_obstacle(obs, self.state)
            if len(obstacle) == 0:
                v, w = self.state[3], self.state[4]
            else:
                v, w = self.generate_action(solver, dt, obstacle, self.state)
            self.moveVel(v, w)
            self.prePos = pos

            if count % 100 == 0:
                csv_file = csv.writer(traj_file)
                data = [round(pos.x, 3), round(pos.y, 3)]
                csv_file.writerow(data)
            # rospy.sleep(dt)

        self.moveVel(0, 0)

if __name__ == "__main__":
    k = QCQP_Control(3, [3.5, 0])
    k.robot_run()
    rospy.spin()
