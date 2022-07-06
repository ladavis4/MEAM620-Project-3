# MEAM620 Autonomous Navigation Project
This repository holds my submission to the final project for MEAM620 "Advanced Robotics" at the University of Pennsylvania in spring of 2022. Thank you to our instructors Dr. Ani Hsieh and Dr. Camillo Taylor, as well as the entire TA staff. All software that I wrote for this project is in Python.

## Project Description and Results
This class involved learning the planning and control to fly a quadcopter through an environment using visual inertial odometry. 
This involved some hands on testing with the Crazyflie platform, but most of our work was in simulation.
We were graded on how quickly our simulated quadcopter could fly through a set of "maps" without hitting any walls. 

My full software solution first involves converting the map into an occupancy grid and using A* to find a path from the starting position to the goal point. 
We prune points from the path using the Ramer–Douglas–Peucker algorithm then fit a minimum-jerk trajectory through all the waypoints. We then iteratively generate more
minimum-jerk trajectories, attempting to balance the tradeoff between speed through the course and control. Example A* paths (in red) and minimum-jerk trajectories (in black) are shown below. 


<img src=images/path1.png height="180"> <img src=images/path2.png height="250">

In the execution phase, we implement an Extended Kalman Filter for localization using IMU data and feature correspondences between stereo cameras. Completion on the "maze" course was one of by best efforts with a time of 6.4 seconds, shown below. 

<img src=images/flying.gif height="450">




