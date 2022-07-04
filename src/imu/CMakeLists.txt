cmake_minimum_required(VERSION 3.0.2)
project(imu)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  sensor_msgs
)

catkin_package()

include_directories(
  ${catkin_INCLUDE_DIRS}
)

add_executable(imu_node src/imu_node.cpp)
add_dependencies(imu_node sensor_msgs_generate_messages_cpp)
 target_link_libraries(imu_node
   ${catkin_LIBRARIES}
 )
