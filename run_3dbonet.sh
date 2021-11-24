#docker build -t 3dbonet -f docker/3DBoNet.Dockerfile .

docker run \
  --network host \
  --gpus all \
  -v $PWD:/home/docker/catkin_ws/src/dnn_experiment \
  -v $PWD/3D-BoNet:/home/docker/catkin_ws/src/dnn_experiment/bonet \
  -v $HOME/dataset/Data_S3DIS:/home/docker/3D-BoNet/data_s3dis\
  -it 3dbonet 

# For first execution,
# cd catkin_ws/
# catkin build dnn_experiment && source ~/.bashrc
# docker commit **** 3dbonet

# After then,
# roslaunch dnn_experiment bonet.launch
