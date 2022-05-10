cmake_minimum_required(VERSION 3.4...3.18)
project(ros_unet)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_EXPORT_COMPILE_COMMANDS On)

## Compile as C++11, supported in ROS Kinetic and newer
#add_compile_options(-std=c++11)

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)

find_package(OpenCV 3 REQUIRED )
find_package(G2O REQUIRED) # Find /opt/ros/melodic/lib/g2o

find_package(catkin REQUIRED
  cv_bridge
  sensor_msgs
  std_msgs
  pcl_conversions
  pcl_ros
  unloader_msgs
  visualization_msgs
)

add_service_files(DIRECTORY srv FILES
  ComputeEdge.srv ComputeObb.srv SetCamera.srv ClearCamera.srv ComputeFloor.srv)
catkin_install_python(PROGRAMS
    scripts/ros_unet_server.py
    scripts/ros_client.py
    DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})
generate_messages(DEPENDENCIES sensor_msgs std_msgs geometry_msgs visualization_msgs ros_unet)

catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES ros_unet
CATKIN_DEPENDS geometry_msgs roscpp rospy sensor_msgs std_msgs pcl_conversions pcl_ros visualization_msgs unloader_msgs message_generation
)

###########
## Build ##
###########
include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${G2O_INCLUDE_DIR}
)

add_executable(obb_server
  src/obb_server.cpp
  src/segment2d.cpp
  src/ros_util.cpp
  src/mask2obb.cpp)
target_link_libraries(obb_server ${catkin_LIBRARIES} ${OpenCV_LIBS})

add_executable(floor_detector src/floor_detector.cpp)
target_link_libraries(floor_detector ${catkin_LIBRARIES} ${OpenCV_LIBS})
