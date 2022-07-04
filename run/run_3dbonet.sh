#docker build -t $USER:3dbonet -f docker/3DBoNet.Dockerfile .

# Change 3D-BoNet directory name while linking, due to syntax error for module name.
docker run \
  -p 5901:5900 \
  --gpus all \
  -v $PWD/docker/init_3dbonet.sh:/home/docker/init_3dbonet.sh \
  -v $PWD/ros_bonet:/home/docker/catkin_ws/src/ros_bonet \
  -v $PWD/pymodules:/home/docker/pymodules \
  -v $PWD/thirdparty:/home/docker/thirdparty \
  -v $PWD/ros_unet/rosbag_train:/home/docker/rosbag_train \
  -v $HOME/dataset/Data_S3DIS:/home/docker/catkin_ws/src/ros_bonet/scripts/bonet/data_s3dis \
  -it $USER:3dbonet

# For first execution,
# source ~/init_3dbonet.sh

# After then,
# roslaunch ros_bonet bonet.launch

