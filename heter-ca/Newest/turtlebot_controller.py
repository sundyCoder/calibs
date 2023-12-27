#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
import csv
import cv2
from sensor_msgs.msg import Image, LaserScan
from geometry_msgs.msg import Twist, Point, Quaternion
from nav_msgs.msg import Odometry
import tf
import time as TIME
from cv_bridge import CvBridge
from std_msgs.msg import String, Int8
from utils import OURS, processScan, processObs, TestInSingle, TestOnlyRobot
from math import radians, copysign, sqrt, pow, pi, atan2, cos, sin
import math
from tf.transformations import euler_from_quaternion
import numpy as np
import copy



#get robot's info and control it
#for every robot, we can create a Turtlebot()
class Turtlebot():
    def __init__(self, turtlebot, end):
        #turtlebot  type: str      name of robot
        self.turtlebot = "robot" + str(turtlebot)
        self.id = turtlebot
        self.file = 'tb'+str(turtlebot)+'.csv'
        self.goal = end
        rospy.init_node(self.turtlebot+'control', anonymous=False)
        rospy.on_shutdown(self.shutdown)
        self.ranges = []

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
        self.scan_sub = rospy.Subscriber(self.turtlebot+'/scan', LaserScan, self.scanCallBack)
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



    def getObstaclePos(self):
        obstacle = []
        if self.turtlebot != 'robot1':
            obstacle.append(processObs(self.obstacle1, 'waffle'))
        if self.turtlebot != 'robot2':
            obstacle.append(processObs(self.obstacle2, 'burger'))
        if self.turtlebot != 'robot3':
            obstacle.append(processObs(self.obstacle3, 'lizi'))
        if self.turtlebot != 'robot4':
            obstacle.append(processObs(self.obstacle4, 'waffle'))
        if self.turtlebot != 'robot5':
            obstacle.append(processObs(self.obstacle5, 'burger'))
        if self.turtlebot != 'robot6':
            obstacle.append(processObs(self.obstacle6, 'lizi'))
        if self.turtlebot != 'robot7':
            obstacle.append(processObs(self.obstacle7, 'waffle'))
        if self.turtlebot != 'robot8':
            obstacle.append(processObs(self.obstacle8, 'burger'))
        if self.turtlebot != 'robot9':
            obstacle.append(processObs(self.obstacle9, 'lizi'))
        if self.turtlebot != 'robot10':
            obstacle.append(processObs(self.obstacle10, 'waffle'))
        if self.turtlebot != 'robot11':
            obstacle.append(processObs(self.obstacle11, 'burger'))
        if self.turtlebot != 'robot0':
            obstacle.append(processObs(self.obstacle12, 'lizi'))
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

    def scanCallBack(self, data):
        #ball's height is too small, cannot be detected
        self.ranges = data.ranges
        
    def odometry_callback(self, odometry):
        
        Quaternions = odometry.pose.pose.orientation
        Euler = tf.transformations.euler_from_quaternion([Quaternions.x, Quaternions.y, Quaternions.z, Quaternions.w])
        if self.id % 3 == 2:
            r = 0.2
        elif self.id % 3 == 1:
            r = 0.4
        else:
            r = 0.4
        #self.state = Agent()
        self.state = [odometry.pose.pose.position.x, odometry.pose.pose.position.y, Euler[2], odometry.twist.twist.linear.x, r]
        



        
        
    
    
    def moveVel(self, linear_vel, angular_vel):
        self.move_cmd.linear.x = linear_vel
        self.move_cmd.angular.z = angular_vel
        self.cmd_vel.publish(self.move_cmd)


    def moveFront(self, distance):
        
        r = rospy.Rate(2)
        
        (pos, rot) = self.get_odom()
        x_start = pos.x
        y_start = pos.y
        pasex = x_start
        pasey = y_start
        goal_x = pos.x + distance*cos(rot)
        goal_y = pos.y + distance*sin(rot)
        while sqrt(pow((pasex - x_start), 2) + pow((pasey - y_start), 2)) <= (distance - 0.1):
            (position, rotation) = self.get_odom()
            #self.robot_pos.publish(self.position)
            
            linear_speed = 1

            pasex = position.x
            pasey = position.y
            self.move_cmd.linear.x = linear_speed * 0.1
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
    

    def runSingle(self, dis):
        #test in static obs
        self.moveFront(dis)
        with open(self.file, "w+") as file:
            csv_file = csv.writer(file)
            time0 = rospy.get_rostime()
            distance = 0
            (pos0, rot0) = self.get_odom()
            while True:
                timekkk = TIME.time()
                (pos, rot) = self.get_odom()
                #print "pos=", pos.x, " ", pos.y
                distance += sqrt((pos0.x-pos.x)**2 + (pos0.y-pos.y)**2)
                pos0 = pos
                data = [round(pos.x, 3), round(pos.y, 3)]
                csv_file.writerow(data)
                if sqrt((pos.x-self.goal[0])**2 + (pos.y-self.goal[1])**2) <= 0.2:
                    break
                o = processScan(self.ranges)
                v, a = TestInSingle(self.state, o, self.goal)
                self.moveVel(v, a)
            time = rospy.get_rostime() - time0
            print "distance=", distance, "time=", time
            self.moveVel(0, 0)

    def runOnlyRobot(self, dis):
        #test in Inter-robot
        self.moveFront(dis)
        with open(self.file, "w+") as file:
            csv_file = csv.writer(file)
            time0 = rospy.get_rostime()
            distance = 0
            (pos0, rot0) = self.get_odom()
            while True:
                timekkk = TIME.time()
                (pos, rot) = self.get_odom()
                #print "pos=", pos.x, " ", pos.y
                distance += sqrt((pos0.x-pos.x)**2 + (pos0.y-pos.y)**2)
                pos0 = pos
                data = [round(pos.x, 3), round(pos.y, 3)]
                csv_file.writerow(data)
                if sqrt((pos.x-self.goal[0])**2 + (pos.y-self.goal[1])**2) <= 0.2:
                    break
                #o = processScan(self.ranges)
                o = self.getObstaclePos()
                v, a = TestOnlyRobot(self.state, o, self.goal)
                #print TIME.time() - timekkk
                self.moveVel(v, a)
            time = rospy.get_rostime() - time0
            print "distance=", distance, "time=", time
            self.moveVel(0, 0)


    
    def run(self, dis):
        #test in random
        self.moveFront(dis)
        with open(self.file, "w+") as file:
            csv_file = csv.writer(file)
            time0 = rospy.get_rostime()
            distance = 0
            (pos0, rot0) = self.get_odom()
            while True:
                timekkk = TIME.time()
                (pos, rot) = self.get_odom()
                #print "pos=", pos.x, " ", pos.y
                distance += sqrt((pos0.x-pos.x)**2 + (pos0.y-pos.y)**2)
                pos0 = pos
                data = [round(pos.x, 3), round(pos.y, 3)]
                csv_file.writerow(data)
                if sqrt((pos.x-self.goal[0])**2 + (pos.y-self.goal[1])**2) <= 0.2:
                    break
                o = processScan(self.ranges)
                OtherRobot = self.getObstaclePos()
                #Obs = calculateObstacle(pos, rot, OtherRobot)
                v, a = OURS(self.state, o, OtherRobot, self.goal)
                #print TIME.time() - timekkk
                self.moveVel(v, a)
            time = rospy.get_rostime() - time0
            print "distance=", distance, "time=", time
            self.moveVel(0, 0)
            #kkkk = float("+")

    def recordLidar(self):
        with open("lidar2.csv", "w+") as file:
            csv_file = csv.writer(file)
            i = 0
            
            while i <= 100:
                #print len(self.ranges)
                if len(self.ranges) >= 10:
                    csv_file.writerow(self.ranges)
                    i += 1

    @staticmethod
    def main():
        rospy.spin()





if __name__ == '__main__':
    #k2 = Turtlebot(2, [1.4, 1.5], 'blue', [0., 0.])
    #k1 = Turtlebot(1, [1.8, 0.7], 'red', [0., 0.])
    k = Turtlebot(0, [4., 4.])
    k.Start(0.1)
    rospy.spin()


