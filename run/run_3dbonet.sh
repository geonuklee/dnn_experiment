#docker build -t 3dbonet -f docker/3DBoNet.Dockerfile .

# Change 3D-BoNet directory name while linking, due to syntax error for module name.
docker run \
  --network host \
  --gpus all \
  -v $PWD/ros_bonet:/home/docker/catkin_ws/src/ros_bonet \
  -v $HOME/dataset/Data_S3DIS:/home/docker/catkin_ws/src/ros_bonet/scripts/bonet/data_s3dis \
  -it 3dbonet 

# For first execution,
# cd catkin_ws/
# catkin build dnn_experiment && source ~/.bashrc
# docker commit **** 3dbonet

# After then,
# roslaunch dnn_experiment bonet.launch

# To run original 3D-Bonet,
# cd bonet; python2 main_eval.py
