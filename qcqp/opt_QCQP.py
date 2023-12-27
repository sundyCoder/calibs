import time
import numpy as np
import gurobipy as gp
from gurobipy import GRB
from config import *

class OptQCQP():
    def __init__(self, rb_id, dt, num_robots):
        self.robot_state = start_state[rb_id]
        self.goal = end_state[rb_id]
        self.robot_radius = radius[rb_id]
        self.num_robots = num_robots
        self.vx_range = [-VEL_MAX, VEL_MAX]
        self.vy_range = [-ANGULAR_MAX, ANGULAR_MAX]
        self.dt = dt
        self.nbr_state = self.get_nbr_state(rb_id)
        self.create_opt()


    def get_nbr_state(self, rb_id):
        import copy
        aa = copy.deepcopy(nbr_state)
        aa.pop(rb_id)
        return np.array(aa)
        
    def create_opt(self):
        self.m = gp.Model("QCQP")

        # 1. define the variables: velocity: vx, vy
        self.vx = self.m.addVar(lb=self.vx_range[0], ub=self.vx_range[1], name="vx")
        self.vy = self.m.addVar(lb=self.vy_range[0], ub=self.vy_range[1], name="vy")
        # position of t+1: x_pred, y_pred
        self.x_pred = self.m.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="xp")
        self.y_pred = self.m.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="yp")
        # position change between t and t+1: x_delta, y_delta
        self.x_delta = self.m.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="xd")
        self.y_delta = self.m.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="yd")

        
        # state of the robot: x, y, orientation
        self.robot_s = [] 
        self.robot_s += [self.m.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="s1")] # x
        self.robot_s += [self.m.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="s2")] # y
        self.robot_s += [self.m.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="s3")] # orientation
        # self.robot_s += [self.m.addVar(lb=-np.pi, ub=np.pi, name="s3")] # orientation
        self.robot_s = np.array(self.robot_s)
        
        # self.neighbor_states: x, y, r
        self.neighbor_states = [] # state of neighbour robot
        for i in range(self.num_robots - 1): # neighbouring robots
            for var_e in ["x_", "y_", "r_"]:
                coeff_name = var_e + str(i)
                self.neighbor_states += [self.m.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name=coeff_name)]
        self.neighbor_states = np.array(self.neighbor_states).reshape((self.num_robots-1, 3))

        # self.ep = [] # position uncertainty
        # for i in range(self.num_robots-1):
        #     self.ep = self.ep + [self.m.addVar(lb=0, ub=GRB.INFINITY, name="ep"+str(i))]
        # self.ep=np.array(self.ep).reshape((1, len(start_state)-1))  # [1, 5]

        # 2. constraints: inter-robot collision
        for i in range(len(self.nbr_state)-1): 
            # self.m.addConstr((self.x_pred - self.neighbor_states[i][0])**2 + (self.y_pred - self.neighbor_states[i][1])**2 - (self.robot_radius + self.neighbor_states[i][2])**2 - self.ep[0][i] >= 0, "c"+str(i))
            self.m.addConstr((self.x_pred - self.neighbor_states[i][0])**2 + (self.y_pred - self.neighbor_states[i][1])**2 - (self.robot_radius + self.neighbor_states[i][2])**2 >= 0, "c1"+str(i))
            # self.m.addConstr((self.robot_s[0] - self.neighbor_states[i][0])**2 + (self.robot_s[1] - self.neighbor_states[i][1])**2 - (self.robot_radius + self.neighbor_states[i][2])**2 >= 0, "c2"+str(i))
        
        # state transformation constraint
        self.m.addConstr(self.x_pred - self.robot_s[0] == self.vx * self.dt)
        self.m.addConstr(self.y_pred - self.robot_s[1] == self.vy * self.dt)

        self.m.addConstrs((self.robot_s[j] == self.robot_state[j] for j in range(3)), "current_state") # robot's initial state
        self.m.addConstrs((self.neighbor_states[i][j] == self.nbr_state[i][j] for i in range(self.num_robots-1) for j in range(3)), "node") # neighbours' state

        # 3. objective function
        obj_func = np.sqrt((self.x_pred - self.goal[0])**2 + (self.y_pred - self.goal[1]**2))
        self.m.setObjective(obj_func, GRB.MINIMIZE)
        # self.m.setObjective(0.5*(self.x_pred - self.goal[0])**2 + 0.5*(self.y_pred - self.goal[1])**2, GRB.MINIMIZE)
        # self.m.setObjective((self.x_pred - self.goal[0])**2 + (self.y_pred - self.goal[1])**2 - ((self.robot_s[0] - self.goal[0])**2 + (self.robot_s[1] - self.goal[1])**2), GRB.MINIMIZE)
        
        
        self.m.setParam(GRB.Param.NonConvex, 2)
        self.m.setParam("OutputFlag", 0)
        # self.m.setParam("DualReductions", 0)
        
        self.m.write("tmc_coll.lp")
        self.m.update()
        self.m.optimize()

        
    def state_update(self, new_robot_state, new_nbr_s):
        """update the status of robots"""
        self.robot_state = new_robot_state
        self.nbr_state = new_nbr_s
        for i in range(3):
            self.m.getConstrByName("current_state["+str(i)+"]").rhs = self.robot_state[i]
        for i in range(self.num_robots-1):
            for j in range(3):
                self.m.getConstrByName("node[" + str(i) + "," + str(j) + "]").rhs = self.nbr_state[i][j]
        self.m.update()
        
    def solve(self):
        """solve the optimization problem"""
        self.m.optimize()
        # print("*********************************************************")
        # print("Time to solve (ms)=",self.m.runtime*1000)
        # print("*********************************************************")
        if self.m.status != 2:
            print(self.m.status)
        if self.m.status == GRB.Status.OPTIMAL:
            # print('Optimal Solution found: [ {}, {}]'.format(self.vx.X, self.vy.X))
            self.vx_output = self.vx.X
            self.vy_output = self.vy.X
            return True, self.vx.X, self.vy.X

        elif self.m.status == GRB.Status.INF_OR_UNBD: # 4
            print('Model is infeasible or unbounded')
            return False, 0, 0
        elif self.m.status == GRB.Status.INFEASIBLE: # 3
            print('Model is infeasible')
            return False, 0, 0
        elif self.m.status == GRB.Status.UNBOUNDED:
            print('Model is unbounded')
            return False, self.vx_output, self.vy_output
        else:
            print('Optimization ended with status %d' % self.m.status)
            return False, 0, 0
            
if __name__ == "__main__":
    solver = OptQCQP(0, 0.2, 4)
    solver.update(np.array([-2., 0., np.pi/2]), np.array([[0., -4., 0.3], [4., 0., 0.2], [0., 4., 0.3]]))
    solver.solve()