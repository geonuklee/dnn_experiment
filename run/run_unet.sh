#docker build -t unet -f docker/unet.Dockerfile .

# Change 3D-BoNet directory name while linking, due to syntax error for module name.
docker run \
  --network host \
  --gpus all \
  -v $PWD:/home/docker/catkin_ws/src/dnn_experiment \
  -v $HOME/dataset/unloading/stc_k4a_2021-11-12-18-20-58.bag:/home/docker/catkin_ws/src/dnn_experiment/rosbag.bag \
  -v ~/.ssh:/home/docker/.ssh \
  -it unet

# ssh geo@localhost -p 8006 "/usr/bin/zsh -ci \"msg The alarm from the docker\""
