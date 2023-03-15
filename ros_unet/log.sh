#mkdir rosbag

#rosbag record -o rosbag/k4a_scale --duration=5s \
#/cam0/k4a/rgb/image_raw \
#/cam0/k4a/depth_to_rgb/image_raw \
#/cam0/aligned/depth_to_rgb/image_rect \
#/cam0/k4a/rgb/camera_info \

#mkdir rosbag_230303
#roslaunch ros_unet label.launch target:=230303
rosbag record -o rosbag_230318/helios --duration=5s \
  /cam0/helios2/depth_rect \
  /cam0/helios2/rgb_rect \
  /cam0/helios2/intensity_rect \
  /cam0/helios2/camera_info_rect \
  /tf /tf_static \
  /cam0/k4a/imu
#roslaunch ros_unet bg.launch filename:=rosbag_230318/helios_2023-03-16-02-21-23.bag
