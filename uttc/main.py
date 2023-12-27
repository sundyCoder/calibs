
import rospy
from scenario import *
from mpi4py import MPI
from control import TTC_Control


if __name__ == "__main__":
    circle4 = [[0., 2.5], [-2.5, 0.], [0., -2.5], [2.5, 0.]]
    circle6 = [[-1.5, 2.6], [-3, 0], [-1.5, -2.6], [1.5, -2.6], [3., 0], [1.5, 2.6]]
    
    groupSwarping4 = [[1.5, -0.4], [-1.5, 0.4], [1.5, 0.4], [-1.5, -0.4]]
    groupCrossing4 = [[0.4, -1.5], [-1.5, 0.4], [-0.4, -1.5], [-1.5, -0.4]]
    groupSwarping6 = [[1.5, -0.8], [-1.5, 0.8], [1.5, 0.8], [-1.5, 0], [1.5, 0.], [-1.5, -0.8]]
    groupCrossing6 = [[0.8, -1.5], [-1.5, 0.8], [-0.8, -1.5], [-1.5, 0], [0., -1.5], [-1.5, -0.8]]


    circle6 = [[-4.0, 0.0], [-2.0, -3.46], [2.0, -3.46], [4.0, -0.0], [2.0, 3.46], [-2.0, 3.46]]
    circle8 = [[-4, 0], [-2.8, -2.8], [0, -4], [2.8, -2.8], [4, 0], [2.8, 2.8], [0, 4], [-2.8, 2.8]]
    circle12 = [[-4, 0], [-3.44, -2.0], [-2.0, -3.44], [0., -4], [2.0, -3.44], [3.44, -2.0], [4., 0], [3.44, 2.0], [2.0, 3.44], [0., 4.], [-2.0, 3.44], [-3.44, 2.0]]
    cross8_start = [[2.9, 0.25], [2.9, -1.26], [1.0,-3.25], [-0.31,-3.24], [-2.7,-1.25], [-2.7, 0.21], [-0.28, 2.47], [1.0, 2.47]]
    cross8_goal = [[-2.7, 0.21], [-2.7, -1.25], [1.0, 2.47], [-0.28, 2.47], [2.9, -1.26],[2.9, 0.25], [-0.31, -3.24], [1.0, -3.25]]

    #random_start = [[0.075, 3.992], [6.582, 7.358], [0.65, 5.564], [4.67,3.731], [0.264, 1.802], [2.778, 3.064], [6.495, 4.714], [7.104, 6.541]]
    #random_goal = [[6.732, 0.167], [5.016, 6.259], [0.556, 0.801], [3.807, 6.408], [7.352, 3.063], [6.38, 4.432], [4.663, 7.602], [1.647, 0.408]]
    random_start = [[1.99, -2.63], [1.57, 1.63], [-1.56, 2.8], [0.60, -1.11], [-0.84, -0.03], [0.13, -4.39], [-2.73, -0.89], [-2.55, -3.31]]
    random_goal = [[-2.73, -0.89], [0.13, -4.39], [-2.55, -3.31], [-0.84, -0.03], [0.60, -1.11], [1.57, 1.63], [1.99, -2.63], [-1.56, 2.8]]

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    k = TTC_Control(rank, circle6[rank])
    k.ttc_run(rank)
    rospy.spin()
