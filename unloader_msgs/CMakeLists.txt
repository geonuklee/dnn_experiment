cmake_minimum_required(VERSION 3.0.2)
project(unloader_msgs)

find_package(catkin REQUIRED
  COMPONENTS
    roscpp rospy
    message_generation
    std_msgs sensor_msgs
    geometry_msgs trajectory_msgs
    fiducial_msgs
)

add_message_files(
  FILES
    RobotState.msg
    VisionState.msg
    Object.msg
    ObjectArray.msg
    ObjectPairs.msg
    BoolArray.msg
    BoolStamped.msg
    CaliInfo.msg
    DateTime.msg
    JointTrajectoryArray.msg
    SyncronizedSeq.msg
    UnloaderTime.msg
    YolactSegm.msg
    YoloDetections.msg
    MultiarmPose.msg
    TimeArray.msg
)

add_service_files(
  FILES
    MarkerPose.srv
    CharucoPose.srv
    Capture.srv
    Optimization.srv
    RobotarmPose.srv
)

generate_messages(
  DEPENDENCIES
    std_msgs
    sensor_msgs
    geometry_msgs
    trajectory_msgs
    fiducial_msgs
)

catkin_package(
  CATKIN_DEPENDS
    message_runtime
    std_msgs
    sensor_msgs
    geometry_msgs
    trajectory_msgs
    fiducial_msgs
)
