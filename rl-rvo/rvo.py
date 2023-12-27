import numpy as np
from math import sin, cos, atan2, asin, pi, sqrt
from time import time

# state: [x, y, vx, vy, radius, vx_des, vy_des]
# moving_state_list: [[x, y, vx, vy, radius]]
# obstacle_state_list: [[x, y, radius]]
# rvo_vel: [vx, vy]

class Reciprocal_vel_obs(object):

    def __init__(self, neighbor_region=5, vxmax = 0.5, vymax = 0.5, acceler = 0.5):

        self.vxmax = vxmax
        self.vymax = vymax
        self.acceler = acceler
        self.nr = neighbor_region

    def cal_vel(self, robot_state, nei_state_list=[], obs_cir_list=[], obs_line_list=[], mode = 'rvo'):
        
        robot_state, ns_list, oc_list, ol_list = self.preprocess(robot_state, nei_state_list, obs_cir_list, obs_line_list)

        # configure the vo or rvo or hrvo
        vo_list = self.config_vo(robot_state, ns_list, oc_list, ol_list, mode)
        vo_outside, vo_inside = self.vel_candidate(robot_state, vo_list)
        rvo_vel = self.vel_select(robot_state, vo_outside, vo_inside, ns_list, oc_list, ol_list, vo_list)

        return rvo_vel

    def preprocess(self, robot_state, nei_state_list, obs_cir_list, obs_line_list):
        # components in the region 
        robot_state = np.squeeze(robot_state)
        ns_list = list(filter(lambda x: 0 < Reciprocal_vel_obs.distance(robot_state, x) <= self.nr, nei_state_list))
        # oc_list = list(filter(lambda y: 0 < Reciprocal_vel_obs.distance(robot_state, y) <= self.nr, obs_cir_list))
        # ol_list = list(map(lambda z: Reciprocal_vel_obs.segment_in_circle(robot_state[0], robot_state[1], self.nr, z), obs_line_list))
        # ol_list = [x for x in ol_list if x is not None]                 

        return robot_state, ns_list, [], []

    def config_vo(self, robot_state, nei_state_list, obs_cir_list, obs_line_list, mode):
        # mode: vo, rvo, hrvo
        vo_list1 = list(map(lambda x: self.config_vo_circle(robot_state, x, mode), nei_state_list))
        vo_list2 = list(map(lambda y: self.config_vo_circle(robot_state, y, 'vo'), obs_cir_list))
        vo_list3 = list(map(lambda z: self.config_vo_line(robot_state, z), obs_line_list))
       
        return vo_list1 + vo_list2 + vo_list3

    def config_vo_circle(self, state, circular, mode='vo'):
        
        x, y, vx, vy, r = state[0:5]
        mx, my, mvx, mvy, mr = circular[0:5]

        dis_mr = sqrt((my - y)**2 + (mx - x)**2)
        angle_mr = atan2(my - y, mx - x)
        
        if dis_mr < r + mr:
           dis_mr = r + mr
        
        ratio = (r + mr)/dis_mr
        half_angle = asin( ratio ) 
        line_left_ori = Reciprocal_vel_obs.wraptopi(angle_mr + half_angle) 
        line_right_ori = Reciprocal_vel_obs.wraptopi(angle_mr - half_angle) 

        if mode == 'vo':
            apex = [mvx, mvy]
        
        elif mode == 'rvo':
            apex = [(vx + mvx)/2, (vy + mvy)/2]

        elif mode == 'hrvo':

            rvo_apex = [(vx + mvx)/2, (vy + mvy)/2]
            vo_apex = [mvx, mvy]

            cl_vector = [mx - x, my - y]

            cur_v = [vx - rvo_apex[0], vy - rvo_apex[1]]

            dis_rv = Reciprocal_vel_obs.distance(rvo_apex, vo_apex)
            radians_rv = atan2(rvo_apex[1] - vo_apex[1], rvo_apex[0] - vo_apex[0])

            diff = line_left_ori - radians_rv

            temp =  pi - 2 * half_angle

            if temp == 0:
                temp = temp + 0.01

            dis_diff = dis_rv * sin(diff) / sin(temp)

            if Reciprocal_vel_obs.cross_product(cl_vector, cur_v) <= 0: 
                apex = [ rvo_apex[0] - dis_diff * cos(line_right_ori), rvo_apex[1] - dis_diff * sin(line_right_ori) ]          
            else:
                apex = [ vo_apex[0] + dis_diff * cos(line_right_ori), vo_apex[1] + dis_diff * sin(line_right_ori) ]       
        
        return apex+[line_left_ori, line_right_ori]  

    def config_vo_line(self, robot_state, line):

        x, y, vx, vy, r = robot_state[0:5]

        apex = [0, 0]

        theta1 = atan2(line[0][1] - y, line[0][0] - x)
        theta2 = atan2(line[1][1] - y, line[1][0] - x)
        
        dis_mr1 = sqrt( (line[0][1] - y)**2 + (line[0][0] - x)**2 )
        dis_mr2 = sqrt( (line[1][1] - y)**2 + (line[1][0] - x)**2 )

        half_angle1 = asin(Reciprocal_vel_obs.clamp(r/dis_mr1, 0, 1))
        half_angle2 = asin(Reciprocal_vel_obs.clamp(r/dis_mr2, 0, 1))

        if Reciprocal_vel_obs.wraptopi(theta2-theta1) > 0:
            line_left_ori = Reciprocal_vel_obs.wraptopi(theta2 + half_angle2)
            line_right_ori = Reciprocal_vel_obs.wraptopi(theta1 - half_angle1)
        else:
            line_left_ori = Reciprocal_vel_obs.wraptopi(theta1 + half_angle1)
            line_right_ori = Reciprocal_vel_obs.wraptopi(theta2 - half_angle2)

        return  apex+[line_left_ori, line_right_ori]  

    def vel_candidate(self, robot_state, vo_list):
        
        vo_outside, vo_inside = [], []
        
        cur_vx, cur_vy = robot_state[2:4]
        cur_vx_range = np.clip([cur_vx-self.acceler, cur_vx+self.acceler], -self.vxmax, self.vxmax)
        cur_vy_range = np.clip([cur_vy-self.acceler, cur_vy+self.acceler], -self.vymax, self.vymax)
        
        for new_vx in np.arange(cur_vx_range[0], cur_vx_range[1], 0.05):
            for new_vy in np.arange(cur_vy_range[0], cur_vy_range[1], 0.05):
                
                if sqrt(new_vx**2 + new_vy**2) < 0.3:
                    continue

                if self.vo_out2(new_vx, new_vy, vo_list):
                    vo_outside.append([new_vx, new_vy])
                else:
                    vo_inside.append([new_vx, new_vy])

        return vo_outside, vo_inside

    def vo_out(self, vx, vy, vo_list):
    
        for rvo in vo_list:
            theta = atan2(vy - rvo[1], vx - rvo[0])
            if Reciprocal_vel_obs.between_angle(rvo[2], rvo[3], theta):
                return False
    
        return True
    
    def vo_out2(self, vx, vy, vo_list):
    
        for rvo in vo_list:
            line_left_vector = [cos(rvo[2]), sin(rvo[2])]
            line_right_vector = [cos(rvo[3]), sin(rvo[3])]
            line_vector = [vx - rvo[0], vy - rvo[1]]
            if Reciprocal_vel_obs.between_vector(line_left_vector, line_right_vector, line_vector):
                return False
    
        return True
    
    def vel_select(self, robot_state, vo_outside, vo_inside, nei_state_list, obs_cir_list, obs_line_list, vo_list):

        vel_des = [robot_state[5], robot_state[6]]
        
        if (len(vo_outside) != 0):
            temp= min(vo_outside, key = lambda v: Reciprocal_vel_obs.distance(v, vel_des))
            return temp
        else:
            temp = min(vo_inside, key = lambda v: self.penalty(v, vel_des, robot_state, nei_state_list, obs_cir_list, obs_line_list, 1) )
            return temp
    """        
    def penalty(self, vel, vel_des, robot_state, nei_state_list, obs_cir_list, obs_line_list, factor):
        
        tc_list = []

        for moving in nei_state_list:
            
            rel_x, rel_y = robot_state[0:2] - moving[0:2]
            rel_vx = 2*vel[0] - moving[2] - robot_state[2]
            rel_vy = 2*vel[1] - moving[3] - robot_state[3]

            tc = self.cal_exp_tim(rel_x, rel_y, rel_vx, rel_vy, robot_state[4]+moving[4])

            tc_list.append(tc)

        for obs_cir in obs_cir_list:
            
            rel_x, rel_y = robot_state[0:2] - obs_cir[0:2]
            rel_vx = vel[0] - moving[2] 
            rel_vy = vel[1] - moving[3]

            tc = self.cal_exp_tim(rel_x, rel_y, rel_vx, rel_vy, robot_state[4]+moving[4])

            tc_list.append(tc)

        for obs_seg in obs_line_list:
            tc = Reciprocal_vel_obs.exp_collision_segment(obs_seg, robot_state[0], robot_state[1], vel[0], vel[1], robot_state[4])
            tc_list.append(tc)
   
        tc_min = min(tc_list)

        if tc_min == 0:
            tc_inv = float('inf')
        else:
            tc_inv = 1/tc_min

        penalty_vel = factor * tc_inv + Reciprocal_vel_obs.distance(vel_des, vel)

        return penalty_vel
    """
    
    # judge the direction by vector
    @staticmethod
    def between_vector(line_left_vector, line_right_vector, line_vector):

        if Reciprocal_vel_obs.cross_product(line_left_vector, line_vector) <= 0 and Reciprocal_vel_obs.cross_product(line_right_vector, line_vector) >= 0:
            return True
        else:
            return False

    @staticmethod
    def between_angle(line_left_ori, line_right_ori, line_ori):
        
        if Reciprocal_vel_obs.wraptopi(line_ori - line_left_ori) <= 0 and Reciprocal_vel_obs.wraptopi(line_ori - line_right_ori) >= 0:
            return True
        else:
            return False

    @staticmethod
    def distance(point1, point2):
        return sqrt( (point2[0] - point1[0]) ** 2 + (point2[1] - point1[1]) ** 2 )
    
    @staticmethod
    def cross_product(vector1, vector2): 
        return float(vector1[0] * vector2[1] - vector2[0] * vector1[1])

    # calculate the expect collision time
    @staticmethod
    def cal_exp_tim(rel_x, rel_y, rel_vx, rel_vy, r):
        # rel_x: xa - xb
        # rel_y: ya - yb

        # (vx2 + vy2)*t2 + (2x*vx + 2*y*vy)*t+x2+y2-(r+mr)2 = 0

        a = rel_vx ** 2 + rel_vy ** 2
        b = 2* rel_x * rel_vx + 2* rel_y * rel_vy
        c = rel_x ** 2 + rel_y ** 2 - r ** 2

        if c <= 0:
            return 0

        temp = b ** 2 - 4 * a * c

        if temp <= 0:
            t = float('inf')
        else:
            t1 = ( -b + sqrt(temp) ) / (2 * a)
            t2 = ( -b - sqrt(temp) ) / (2 * a)

            t3 = t1 if t1 >= 0 else float('inf')
            t4 = t2 if t2 >= 0 else float('inf')
        
            t = min(t3, t4)

        return t
    
    """
    @staticmethod
    def segment_in_circle(x, y, r, line):
        # 
        # center: x, y, center point of the circle
        # r, radius of the circle
        # line: two point  
        # reference: https://stackoverflow.com/questions/1073336/circle-line-segment-collision-detection-algorithm
        start_point = np.array(line[0:2])

        d = np.array([line[2] - line[0], line[3] - line[1] ])
        f = np.array([line[0] - x, line[1] - y])
        a = d @ d
        b = 2*f @ d
        c = f @ f - r**2

        discriminant = b**2 - 4*a*c

        if discriminant < 0:
            return None
        else:
            t1 = (-b - sqrt(discriminant)) / (2*a) 
            t2 = (-b + sqrt(discriminant)) / (2*a) 

            if t1>=0 and t1<=1 and t2>=0 and t2<=1:
                segment_point1 = start_point + t1 * d
                segment_point2 = start_point + t2 * d

            elif t1>=0 and t1<=1 and t2 > 1:
                segment_point1 = start_point + t1 * d
                segment_point2 = np.array(line[2:4])
            
            elif t1<0 and t2>=0 and t2<=1:
                segment_point1 = np.array(line[0:2]) 
                segment_point2 = start_point + t2 * d
            
            elif t1<0 and t2>1:
                segment_point1 = np.array(line[0:2]) 
                segment_point2 = np.array(line[2:4]) 
            else:
                return None
        
        diff_norm = np.linalg.norm(segment_point1 - segment_point2)

        if diff_norm == 0:
            return None

        return [segment_point1, segment_point2]
    """
    
    @staticmethod
    def wraptopi(theta):

        if theta > pi:
            theta = theta - 2*pi
        
        if theta < -pi:
            theta = theta + 2*pi

        return theta

    @staticmethod
    def clamp(n, minn, maxn):
        return max(min(maxn, n), minn)    

    """
    @staticmethod
    def exp_collision_segment(obs_seg, x, y, vx, vy, r):

        point1 = obs_seg[0]
        point2 = obs_seg[1]
        
        t1 = Reciprocal_vel_obs.cal_exp_tim(x - point1[0], y - point1[1], vx, vy, r)
        t2 = Reciprocal_vel_obs.cal_exp_tim(x - point2[0], y - point2[1], vx, vy, r)

        c_point = np.array([x, y])   

        l0 = (point2 - point1 ) @ (point2 - point1 )
        t = (c_point - point1) @ (point2 - point1) / l0
        project = point1 + t * (point2 - point1)
        distance = sqrt( (project - c_point) @ (project - c_point) ) 
        theta1 = atan2( (project - c_point)[1], (project - c_point)[0] )
        theta2 = atan2( vy, vx)
        theta3 = Reciprocal_vel_obs.wraptopi(theta2 - theta1)

        real_distance = (distance - r) / cos(theta3)
        speed = sqrt(vy**2 + vx**2)

        if speed == 0:
            t3 = float('inf')
        else:
            t3 = real_distance / sqrt(vy**2 + vx**2)

        if t3 < 0:
            t3 = float('inf')

        return min([t1, t2, t3])
    """   
        
class Rvo_inter(Reciprocal_vel_obs):
    def __init__(self, neighbor_region=5, neighbor_num=10, vxmax=0.5, vymax=0.5, acceler=0.5, env_train=True, exp_radius=0.2, ctime_threshold=5, ctime_line_threshold=1):
        super(Rvo_inter, self).__init__(neighbor_region, vxmax, vymax, acceler)

        self.env_train = env_train
        self.exp_radius = exp_radius
        self.nm = neighbor_num
        self.ctime_threshold = ctime_threshold
        self.ctime_line_threshold = ctime_line_threshold

    def config_vo_inf(self, robot_state, nei_state_list, obs_cir_list=[], obs_line_list=[], action=np.zeros((2,)), **kwargs):
        # mode: vo, rvo, hrvo
        robot_state, ns_list, _, _ = self.preprocess(robot_state, nei_state_list, obs_cir_list, obs_line_list)

        action = np.squeeze(action)

        vo_list1 = list(map(lambda x: self.config_vo_circle2(robot_state, x, action, 'rvo', **kwargs), ns_list))

        obs_vo_list = []

        collision_flag = False
        vo_flag = False
        min_exp_time = float('inf')

        for vo_inf in vo_list1:
            obs_vo_list.append(vo_inf[0])
            if vo_inf[1] is True:
                vo_flag = True
                if vo_inf[2] < min_exp_time:
                    min_exp_time = vo_inf[2]            
            if vo_inf[3] is True:
                collision_flag = True

        obs_vo_list.sort(reverse=True, key=lambda x: (-x[-1], x[-2]))

        if len(obs_vo_list) > self.nm:
            obs_vo_list_nm = obs_vo_list[-self.nm:]
        else:
            obs_vo_list_nm = obs_vo_list

        if self.nm == 0:
            obs_vo_list_nm = []

        return obs_vo_list_nm, vo_flag, min_exp_time, collision_flag
        
    def config_vo_reward(self, robot_state, nei_state_list, obs_cir_list=[], obs_line_list=[], action=np.zeros((2,)), **kwargs):

        robot_state, ns_list, oc_list, ol_list = self.preprocess(robot_state, nei_state_list, obs_cir_list, obs_line_list)

        vo_list1 = list(map(lambda x: self.config_vo_circle2(robot_state, x, action, 'rvo', **kwargs), ns_list))
        
        vo_flag = False
        min_exp_time = float('inf')
        min_dis = float('inf')

        for vo_inf in vo_list1:

            if vo_inf[4] < min_dis:
                min_dis = vo_inf[4]

            if vo_inf[1] is True:
                vo_flag = True
                if vo_inf[2] < min_exp_time:
                    min_exp_time = vo_inf[2]
        
        return vo_flag, min_exp_time, min_dis


    def config_vo_circle2(self, state, circular, action, mode='rvo', **kwargs):
        
        x, y, vx, vy, r = state[0:5]
        mx, my, mvx, mvy, mr = circular[0:5]

        if mvx == 0 and mvy == 0:
            mode = 'vo'

        vo_flag = False
        collision_flag = False

        rel_x = x - mx
        rel_y = y - my

        dis_mr = sqrt((rel_y)**2 + (rel_x)**2)
        angle_mr = atan2(my - y, mx - x)

        real_dis_mr = sqrt((rel_y)**2 + (rel_x)**2)
        
        env_train = kwargs.get('env_train', self.env_train)

        if env_train:
            if dis_mr <= r + mr:
                dis_mr = r + mr
                collision_flag = True
        else:
            if dis_mr <= r - self.exp_radius + mr:
                collision_flag = True

            if dis_mr <= r + mr:
                dis_mr = r + mr
                
        ratio = (r + mr)/dis_mr
        half_angle = asin( ratio ) 
        line_left_ori = Reciprocal_vel_obs.wraptopi(angle_mr + half_angle) 
        line_right_ori = Reciprocal_vel_obs.wraptopi(angle_mr - half_angle) 

        if mode == 'vo':
            vo = [mvx, mvy, line_left_ori, line_right_ori]
            rel_vx = action[0] - mvx 
            rel_vy = action[1] - mvy

        elif mode == 'rvo':
            vo = [(vx + mvx)/2, (vy + mvy)/2, line_left_ori, line_right_ori] 
            rel_vx = 2*action[0] - mvx - vx
            rel_vy = 2*action[1] - mvy - vy

        exp_time = float('inf')

        if self.vo_out_jud_vector(action[0], action[1], vo):
            vo_flag = False
            exp_time = float('inf')
        else:
            exp_time = self.cal_exp_tim(rel_x, rel_y, rel_vx, rel_vy, r + mr)
            if exp_time < self.ctime_threshold:
                vo_flag = True
            else:
                vo_flag = False
                exp_time = float('inf')
            
        input_exp_time = 1 / (exp_time+0.2)
        min_dis = real_dis_mr-mr

        observation_vo = [vo[0], vo[1], cos(vo[2]), sin(vo[2]), cos(vo[3]), sin(vo[3]), min_dis, input_exp_time]
        
        return [observation_vo, vo_flag, exp_time, collision_flag, min_dis]


    def vo_out_jud_vector(self, vx, vy, vo):
        
        rel_vector = [vx - vo[0], vy - vo[1]]
        line_left_vector = [cos(vo[2]), sin(vo[2]) ]
        line_right_vector = [cos(vo[3]), sin(vo[3]) ]
        
        if Reciprocal_vel_obs.between_vector(line_left_vector, line_right_vector, rel_vector):
            return False
        else:
            return True




