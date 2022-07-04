cmake_minimum_required(VERSION 3.0.2)
project(abb)

find_package(catkin REQUIRED COMPONENTS
  cv_bridge
  image_transport
  roscpp
  sensor_msgs
  std_msgs
  nav_msgs
)

include_directories(
  include/abb
  ${catkin_INCLUDE_DIRS}
)

find_package(OpenGL REQUIRED)
find_package(OpenCV REQUIRED)

include_directories(${OPENGL_INCLUDE_DIR})
include_directories(${OpenCV_INCLUDE_DIR})

add_executable(${PROJECT_NAME}_node abb_node.cpp function.h function.cpp )

add_dependencies(${PROJECT_NAME}_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

## Specify libraries to link a library or executable target against
target_link_libraries(${PROJECT_NAME}_node
   ${catkin_LIBRARIES} ${OpenCV_LIBRARIES} ${OPENGL_LIBRARIES}
)

