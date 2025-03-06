cmake_minimum_required(VERSION 3.0.2)
project(nav)

find_package(catkin REQUIRED COMPONENTS
  sensor_msgs
  roscpp
  rospy
)

catkin_package()

include_directories(
  ${catkin_INCLUDE_DIRS}
)

add_executable(nav_node src/nav_node.cpp)
add_dependencies(nav_node sensor_msgs_generate_messages_cpp)
 target_link_libraries(nav_node
   ${catkin_LIBRARIES}
 )
