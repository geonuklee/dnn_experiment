cmake_minimum_required(VERSION 3.4...3.18)
project(ros_unet)
set(CMAKE_CXX_STANDARD 14)

## Compile as C++11, supported in ROS Kinetic and newer
# add_compile_options(-std=c++11)

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)

find_package(OpenCV 3 REQUIRED )
find_package(G2O REQUIRED) # Find /opt/ros/melodic/lib/g2o

find_package(catkin REQUIRED
  cv_bridge
  sensor_msgs
  std_msgs
  pcl_conversions
  pcl_ros
)

catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES ros_unet
    CATKIN_DEPENDS geometry_msgs roscpp rospy sensor_msgs std_msgs pcl_conversions pcl_ros visualization_msgs
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
add_executable(pipeline src/pipeline_node.cpp src/segment2d.cpp src/segment2d_ml.cpp src/ros_util.cpp)
target_link_libraries(pipeline ${catkin_LIBRARIES} ${OpenCV_LIBS})
