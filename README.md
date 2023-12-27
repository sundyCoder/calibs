# Tfresh


## How to launch the environment

    export PYTHONPATH=/home/jlchen/catkin_ws/devel/lib/python2.7/dist-packages:/opt/ros/melodic/lib/python2.7/dist-packages
    
    roslaunch scenaro/circle_6r_3t.launch
    
## How to run the training script

    export PYTHONPATH=/home/jlchen/catkin_ws/devel/lib/python2.7/dist-packages:/opt/ros/melodic/lib/python2.7/dist-packages
    
    mpiexec -n 8 python2 train.py 


