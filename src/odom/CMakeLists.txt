cmake_minimum_required(VERSION 3.0.2)
project(odom)

find_package(catkin REQUIRED COMPONENTS
  nav_msgs
  roscpp
  rospy
)

catkin_package()

include_directories(
  ${catkin_INCLUDE_DIRS}
)

add_executable(odom_node src/odom_node.cpp)
add_dependencies(odom_node nav_msgs_generate_messages_cpp)
 target_link_libraries(odom_node
   ${catkin_LIBRARIES}
 )
