#docker build -t 3dbonet -f docker/3DBoNet.Dockerfile .

# Change 3D-BoNet directory name while linking, due to syntax error for module name.
docker run \
  --network host \
  --gpus all \
  -v $PWD/ros_bonet:/home/docker/catkin_ws/src/ros_bonet \
  -v $PWD/pymodules:/home/docker/pymodules \
  -v $PWD/thirdparty:/home/docker/thirdparty \
  -v $PWD/ros_unet/obb_dataset_train:/home/docker/obb_dataset_train \
  -v $PWD/ros_unet/rosbag_train:/home/docker/rosbag_train \
  -v $HOME/dataset/Data_S3DIS:/home/docker/catkin_ws/src/ros_bonet/scripts/bonet/data_s3dis \
  -it 3dbonet 

  #-v $PWD/ros_unet/obb_dataset_train:/home/docker/catkin_ws/src/ros_bonet/obb_dataset_train \
# For first execution,
# # Denote that call cmake 2 times.
# cd ~/thirdparty/g2o; mkdir build4docker; cd build4docker; cmake ..; cmake ..; make -j8
# cd ~/pymodules; python2 setup.py install --user
# cd ~/pymodules/exts; mkdir build4docker; cd build4docker;
# cmake ../py2ext -DCMAKE_BUILD_TYPE=Release; make -j8;
# python2 -c 'import unet_ext; print(unet_ext)'
# cd ~/catkin_ws/
# catkin build ros_bonet && source ~/.bashrc
# docker commit **** 3dbonet

# After then,
# roslaunch ros_bonet bonet.launch

# To run original 3D-Bonet,
# cd bonet; python2 main_eval.py
