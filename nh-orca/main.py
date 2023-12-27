from multiTurtleSim import *
sim  = MultiTurtleSim(
                timestep= 0.25,
                botRaidus=0.4, 
                effective_distance=0.25,
                wheelDist=0.46,
                neighborDist=15,
                maxNeighbors=10,
                timeHorizon= 1,
                timeHorizonObst=10,
                maxSpeed=0.5,
                maxLinearSpeed = 9999,
                maxRotSpeed =99999,
                botStartVel = np.array([0,0])
)
# rospy.set_param("use_sim_time",True)


'''

circle4 = [[0., 2.0], [-2.0, 0.], [0., -2.0], [2.0, 0.]]
sim.addBot(1,np.array([0., -2.0]),0.0,circle4[0])
sim.addBot(2,np.array([2.0, 0.]),0.0,circle4[1])
sim.addBot(3,np.array([0, 2.0]),0.0,circle4[2])
sim.addBot(4,np.array([-2.0, 0.]),0.0,circle4[3])

groupSwarping4 = [[1.5, -0.4], [-1.5, 0.4], [1.5, 0.4], [-1.5, -0.4]]
sim.addBot(1,np.array([-1.5, -0.4]),0.0,groupSwarping4[0])
sim.addBot(2,np.array([1.5, 0.4]),0.0,groupSwarping4[1])
sim.addBot(3,np.array([-1.5, 0.4]),0.0,groupSwarping4[2])
sim.addBot(4,np.array([1.5, -0.4]),0.0,groupSwarping4[3])


groupSwarping6 = [[1.5, -0.8], [-1.5, 0.8], [1.5, 0.8], [-1.5, 0], [1.5, 0.], [-1.5, -0.8]]
sim.addBot(1,np.array([-1.5, -0.8]),0.0,groupSwarping6[0])
sim.addBot(2,np.array([1.5, 0.8]),0.0,groupSwarping6[1])
sim.addBot(3,np.array([-1.5, 0.8]),0.0,groupSwarping6[2])
sim.addBot(4,np.array([1.5, 0]),0.0,groupSwarping6[3])
sim.addBot(5,np.array([-1.5, 0]),0.0,groupSwarping6[4])
sim.addBot(6,np.array([1.5, -0.8]),0.0,groupSwarping6[5])


groupCrossing6 = [[0.8, -1.5], [-1.5, 0.8], [-0.8, -1.5], [-1.5, 0], [0., -1.5], [-1.5, -0.8]]
sim.addBot(1,np.array([0.8, 1.5]),0.0,groupCrossing6[0])
sim.addBot(2,np.array([1.5, 0.8]),0.0,groupCrossing6[1])
sim.addBot(3,np.array([-0.8, 1.5]),0.0,groupCrossing6[2])
sim.addBot(4,np.array([1.5, 0]),0.0,groupCrossing6[3])
sim.addBot(5,np.array([0, 1.5]),0.0,groupCrossing6[4])
sim.addBot(6,np.array([1.5, -0.8]),0.0,groupCrossing6[5])


circle8 = [[-2.8, 2.8], [-4, 0], [-2.8, -2.8], [0, -4], [2.8, -2.8], [4, 0], [2.8, 2.8], [0, 4]]
sim.addBot(1,np.array([2.8, -2.8]),0.0,circle8[0])
sim.addBot(2,np.array([4, 0.]),0.0,circle8[1])
sim.addBot(3,np.array([2.8, 2.8]),0.0,circle8[2])
sim.addBot(4,np.array([0, 4]),0.0,circle8[3])
sim.addBot(5,np.array([-2.8, 2.8]),0.0,circle8[4])
sim.addBot(6,np.array([-4, 0.]),0.0,circle8[5])
sim.addBot(7,np.array([-2.8, -2.8]),0.0,circle8[6])
sim.addBot(8,np.array([0, -4]),0.0,circle8[7])

circle12 = [[-3.44, 2.0], [-4, 0], [-3.44, -2.0], [-2.0, -3.44], [0., -4.], [2.0, -3.44], [3.44, -2.0], [4., 0], [3.44, 2.0], [2.0, 3.44], [0., 4.], [-2.0, 3.44]]
sim.addBot(1,np.array([3.44, -2.0]),0.0,circle12[0])
sim.addBot(2,np.array([4., 0.]),0.0,circle12[1])
sim.addBot(3,np.array([3.44, 2.0]),0.0,circle12[2])
sim.addBot(4,np.array([2.0, 3.44]),0.0,circle12[3])
sim.addBot(5,np.array([0, 4.]),0.0,circle12[4])
sim.addBot(6,np.array([-2.0, 3.44]),0.0,circle12[5])
sim.addBot(7,np.array([-3.44, 2.0]),0.0,circle12[6])
sim.addBot(8,np.array([-4., 0]),0.0,circle12[7])
sim.addBot(9,np.array([-3.44, -2.0]),0.0,circle12[8])
sim.addBot(10,np.array([-2.0, -3.44]),0.0,circle12[9])
sim.addBot(11,np.array([0, -4.]),0.0,circle12[10])
sim.addBot(12,np.array([2.0, -3.44]),0.0,circle12[11])

Z2 = [[4.2, 4], [0., 0.]]
sim.addBot(1,np.array([0., 0.]),0.0,Z2[0])
sim.addBot(2,np.array([4., 4.]),0.0,Z2[1])




Z2 = [[2, 5], [5, 5], [5, 2], [2, 2]]
sim.addBot(1,np.array([7., 0.]),0.0,Z2[0])
sim.addBot(2,np.array([0., 0.]),0.0,Z2[1])
sim.addBot(3,np.array([0., 7.]),0.0,Z2[2])
sim.addBot(4,np.array([7., 7.]),0.0,Z2[3])
Z2 = [[4.2, 4], [0., 0.]]
sim.addBot(1,np.array([0., 0.]),0.0,Z2[0])
sim.addBot(2,np.array([4., 4.]),0.0,Z2[1])
circle8 = [[-2.8, 2.8], [-4, 0], [-2.8, -2.8], [0, -4], [2.8, -2.8], [4, 0], [2.8, 2.8], [0, 4]]
sim.addBot(1,np.array([2.8, -2.8]),0.0,circle8[0])
sim.addBot(2,np.array([4, 0.]),0.0,circle8[1])
sim.addBot(3,np.array([2.8, 2.8]),0.0,circle8[2])
sim.addBot(4,np.array([0, 4]),0.0,circle8[3])
sim.addBot(5,np.array([-2.8, 2.8]),0.0,circle8[4])
sim.addBot(6,np.array([-4, 0.]),0.0,circle8[5])
sim.addBot(7,np.array([-2.8, -2.8]),0.0,circle8[6])
sim.addBot(8,np.array([0, -4]),0.0,circle8[7])
'''

circle6 = [[-1.5, 2.6], [-3, 0], [-1.5, -2.6], [1.5, -2.6], [3., 0], [1.5, 2.6]]
sim.addBot(1,np.array([1.5, -2.6]),0.0,circle6[0])
sim.addBot(2,np.array([3, 0]),0.0,circle6[1])
sim.addBot(3,np.array([1.5, 2.6]),0.0,circle6[2])
sim.addBot(4,np.array([-1.5, 2.6]),0.0,circle6[3])
sim.addBot(5,np.array([-3, 0]),0.0,circle6[4])
sim.addBot(6,np.array([-1.5, -2.6]),0.0,circle6[5])

while not (sim.done() or rospy.is_shutdown()):
    sim.step()
    rospy.sleep(0.5)
