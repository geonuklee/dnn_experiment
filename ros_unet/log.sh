#mkdir rosbag

#rosbag record -o rosbag/k4a_scale --duration=5s \
#/cam0/k4a/rgb/image_raw \
#/cam0/k4a/depth_to_rgb/image_raw \
#/cam0/aligned/depth_to_rgb/image_rect \
#/cam0/k4a/rgb/camera_info \

rosbag record -o rosbag_0223/aruco --duration=5s \
  /cam0/aligned/rgb_to_depth/image_raw \
  /cam0/helios2/depth/image_raw \
  /cam0/helios2/camera_info
  #\ | egrep "Recording" | sed -e 's/.*\(rosbag_0523\/helios.*\.bag\).*/\1/'| xargs rosbag info

