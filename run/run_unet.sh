#docker build -t unet -f docker/unet.Dockerfile .

# Change 3D-BoNet directory name while linking, due to syntax error for module name.
docker run \
  --network host \
  --gpus all \
  -v $PWD:/home/docker/catkin_ws/src/dnn_experiment \
  -v /home/geo/dataset/unloading/stc2021/stc_2021-08-19-11-48-10.bag:/home/docker/catkin_ws/src/dnn_experiment/rosbag.bag \
  -v ~/.ssh:/home/docker/.ssh \
  -it unet

# ssh geo@localhost -p 8006 "/usr/bin/zsh -ci \"msg The alarm from the docker\""
