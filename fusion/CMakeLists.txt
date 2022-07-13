cmake_minimum_required(VERSION 3.0.2)
project(fusion)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  sensor_msgs
)

catkin_package()

include_directories(
  ${catkin_INCLUDE_DIRS}
)

add_executable(fusion_node src/fusion_node.cpp)
add_dependencies(fusion_node sensor_msgs_generate_messages_cpp)
 target_link_libraries(fusion
   ${catkin_LIBRARIES}
 )
