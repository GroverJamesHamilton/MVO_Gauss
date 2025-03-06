
Visual odometry with scale recovery howto:
After making a catkin workspace in the repo,
1. In the terminal, build with catkin_make in abb_ws
2. cd into /src/abb (The essential files here are abb_node.cpp, function.cpp and function.h header file, EKF in filter.cpp and filter.h)
3. Find you image topic by typing: rosbag info <bag file>
4. Replace topic to subscribe to under //Replace this with your topic below// in abb_node.cpp
5. Build in abb_ws/src/abb with cmake. and then make
6. Init ROS by typing roscore in a separate terminal
7. In previous terminal, run program with: rosrun abb abb_node
The program will now wait for either the image topic or ground truth topic /tf,
if /tf is not available the ground truth will not be printed on the trajectory
same if the subscribed image topic is not found
8. Run your bagfile with: rosbag play --clock <bagfile>

2 modes are available:
Unscaled and scaled, in abb_node.cpp change bool "scaleRecoveryMode" to true for scale recovery and false
for unscaled.

The program is adapted for the KITTI dataset, in which most data has the same intrinsics parameters.
But you can define your own as well

Offline mode: The algorithm is most likely to be slower than the frame sample size
therefore the int parameter "queueSize" can be changed can be altered. Can be changed to 1 for online-mode
but the result may be very inaccurate depending on the ORB parameters

The function also publishes unscaled or scaled visual odometry to /odom, the the timestamp and velocity will be inaccurate in offline-mode
