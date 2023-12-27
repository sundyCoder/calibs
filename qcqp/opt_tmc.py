import numpy as np
import time
import gurobipy as gp
from gurobipy import GRB

start_state = np.array([[-4., 0., 0.], [0., -4., np.pi/2], [4., 0., np.pi], [0., 4., -np.pi/2]])
end_state = np.array([[4., 0.], [0., 4.], [-4., 0.], [0., -4.]])
all_radius = [0.2, 0.3, 0.2, 0.3]


class Opt():
    def __init__(self, i, dt, num_robots):
        self.robot_state = start_state[i]
        self.target_position = end_state[i]
        self.robot_radius = all_radius[i]
        self.neighbor_robots = np.array([[0., -4., 0.3], [4., 0., 0.2], [0., 4., 0.3]])
        self.num_robots = num_robots
        self.v_round = [0., 0.5]
        self.w_round = [-1., 1.]
        self.dt = dt
        
        self.create_opt()

        
    def create_opt(self):
        self.m = gp.Model("local_solve")
        self.v = self.m.addVar(lb=self.v_round[0], ub=self.v_round[1], name="v")
        self.w = self.m.addVar(lb=self.w_round[0], ub=self.w_round[1], name="w")
        self.x_pred = self.m.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="xp")
        self.y_pred = self.m.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="yp")
        self.x_delta = self.m.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="xd")
        self.y_delta = self.m.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="yd")
        self.sin_cos_ref = self.m.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="sin_cos_ref")
        
        
        self.robot_sta = []
        for i in range(3):
            self.robot_sta=self.robot_sta+[self.m.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="s"+str(i))]
        self.robot_sta=np.array(self.robot_sta)
        
        self.neighbor_rob = []
        coeff=["ax","ay","ar","bx","by","br","cx","cy","cr","dx","dy","dr","ex","ey","er","fx","fy","fr","gx","gy","gr","hx","hy","hr"] 
        for i in range(self.num_robots - 1):
            for j in range(3):
                self.neighbor_rob=self.neighbor_rob+[self.m.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name=coeff[j+3*i])]
        self.neighbor_rob=np.array(self.neighbor_rob).reshape((self.num_robots-1, 3))

        self.ep = []
        for i in range(self.num_robots-1):
            self.ep = self.ep + [self.m.addVar(lb=0, ub=GRB.INFINITY, name="ep"+str(i))]
        self.ep=np.array(self.ep).reshape((1, len(start_state)-1))   

        
        for i in range(len(self.neighbor_robots)-1):
            self.m.addConstr((self.x_pred - self.neighbor_rob[i][0])**2 + (self.y_pred - self.neighbor_rob[i][1])**2 - (self.robot_radius + self.neighbor_rob[i][2])**2 - self.ep[0][i] >= 0, "c"+str(i))
        
        self.m.addConstr(2 * self.robot_sta[2] + self.w * self.dt == 2 * self.sin_cos_ref, "d1")
        self.m.addConstr(self.x_pred - self.robot_sta[0] == self.v * self.x_delta * self.dt, "d2")
        self.m.addConstr(self.y_pred - self.robot_sta[1] == self.v * self.y_delta * self.dt, "d3")
        self.m.addGenConstrSin(self.sin_cos_ref, self.y_delta, "d4")
        self.m.addGenConstrCos(self.sin_cos_ref, self.x_delta, "d5")
        
        self.m.addConstrs((self.robot_sta[j] == self.robot_state[j] for j in range(3)), "current_state")
        self.m.addConstrs((self.neighbor_rob[i][j] == self.neighbor_robots[i][j] for i in range(self.num_robots-1) for j in range(3)), "node")
        
        self.m.setObjective((self.x_pred - self.target_position[0])**2 + (self.y_pred - self.target_position[1])**2, GRB.MINIMIZE)
        
        self.m.setParam(GRB.Param.NonConvex, 2)
        
        self.m.write("tmc_coll.lp")
        self.m.update()
        self.m.optimize()


        print("*********************************************************")
        print("Time to solve (ms)=",self.m.runtime*1000)
        print("*********************************************************")

        if self.m.status == GRB.Status.OPTIMAL:
            print('Optimal Solution found')
            
            print(self.v.X, self.w.X)
            print("linear_velocity:", self.v.X, " angular_velocity:", self.w.X)
            self.v_output = self.v.X
            self.w_output = self.w.X
            
            return True
        elif self.m.status == GRB.Status.INF_OR_UNBD:
            print('Model is infeasible or unbounded')
            return False
        elif self.m.status == GRB.Status.INFEASIBLE:
            print('Model is infeasible')
            return False
        elif self.m.status == GRB.Status.UNBOUNDED:
            print('Model is unbounded')
            return False
        else:
            print('Optimization ended with status %d' % self.m.status)
            return False

        
    def update(self, robot_state, neighbor_robots):
        self.robot_state = robot_state
        self.neighbor_robots = neighbor_robots
        # print(self.m.getConstrByName("node1[0]").rhs)
        # print(self.m.getConstrByName("node2[0,0]").rhs)
        for i in range(3):
            self.m.getConstrByName("current_state["+str(i)+"]").rhs = self.robot_state[i]
        for i in range(self.num_robots-1):
            for j in range(3):
                self.m.getConstrByName("node[" + str(i) + "," + str(j) + "]").rhs = self.neighbor_robots[i][j]
        self.m.update()
        
    def solve(self):
        self.m.optimize()
        print("*********************************************************")
        print("Time to solve (ms)=",self.m.runtime*1000)
        print("*********************************************************")

        if self.m.status == GRB.Status.OPTIMAL:
            print('Optimal Solution found')
            
            print("linear_velocity:", self.v.X, " angular_velocity:", self.w.X)
            self.v_output = self.v.X
            self.w_output = self.w.X
            
            return True
        elif self.m.status == GRB.Status.INF_OR_UNBD:
            print('Model is infeasible or unbounded')
            return False
        elif self.m.status == GRB.Status.INFEASIBLE:
            print('Model is infeasible')
            return False
        elif self.m.status == GRB.Status.UNBOUNDED:
            print('Model is unbounded')
            return False
        else:
            print('Optimization ended with status %d' % self.m.status)
            return False
            
if __name__ == "__main__":
    solver = Opt(0, 0.2, 4)
    solver.update(np.array([-2., 0., np.pi/2]), np.array([[0., -4., 0.3], [4., 0., 0.2], [0., 4., 0.3]]))
    solver.solve()
