
import rospy
from config import *
from mpi4py import MPI
from control import QCQP_Control


if __name__ == "__main__":

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # dt = 0.8 # 60/62.50  FPS of Gazebo simulator
    k = QCQP_Control(rank, circle6_goal[rank])
    k.robot_run(dt)
    rospy.spin()
