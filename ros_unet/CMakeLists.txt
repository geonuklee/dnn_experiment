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

#catkin_python_setup() # Call setup.py for catkin. ref for setup.py of catkin : http://docs.ros.org/en/jade/api/catkin/html/howto/format2/installing_python.html
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

# Above Depends on pcl::segmentation > pcl::io > vtk > python2 which cause import failure at python3
# To exclude libpython2.7, use hard coded library path. (I can't remove the libpython2.7 from the string with CMake)
set(PCL_LIBRARIES
  /usr/lib/x86_64-linux-gnu/libpcl_surface.so.1.8
  /usr/lib/x86_64-linux-gnu/libpcl_common.so.1.8
  /usr/lib/x86_64-linux-gnu/libpcl_filters.so.1.8
  /usr/lib/x86_64-linux-gnu/libpcl_segmentation.so.1.8
  /usr/lib/x86_64-linux-gnu/libpcl_sample_consensus.so.1.8
  /usr/lib/x86_64-linux-gnu/libpcl_search.so.1.8
  /usr/lib/x86_64-linux-gnu/libpcl_kdtree.so.1.8
  /usr/lib/x86_64-linux-gnu/libpcl_octree.so.1.8
  /usr/lib/x86_64-linux-gnu/libpcl_ml.so.1.8
  /usr/lib/x86_64-linux-gnu/libpcl_features.so.1.8
  /lib/x86_64-linux-gnu/libutil.so.1
)

add_library(unetsegment SHARED
  src/watershed.cpp
  src/segment2d.cpp
  src/utils.cpp
)
target_link_libraries(unetsegment ${OpenCV_LIBS} ${PCL_LIBRARIES})

add_executable(obb_server
  src/mask2obb.cpp
  src/obb_server.cpp
  src/ros_util.cpp
)

target_link_libraries(obb_server unetsegment ${catkin_LIBRARIES} ${OpenCV_LIBS})

add_executable(floor_detector src/floor_detector.cpp)
target_link_libraries(floor_detector ${catkin_LIBRARIES} ${OpenCV_LIBS})
