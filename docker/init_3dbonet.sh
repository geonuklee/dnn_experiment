CWD=$PWD
cd ~/thirdparty/g2o
sudo rm -rf build4docker; sudo mkdir build4docker
cd build4docker
sudo cmake -DCMAKE_BUILD_TYPE=Release -DG2O_USE_CSPARSE=OFF -DG2O_BUILD_APPS=OFF -DG2O_BUILD_EXAMPLES=OFF ..
sudo cmake -DCMAKE_BUILD_TYPE=Release -DG2O_USE_CSPARSE=OFF -DG2O_BUILD_APPS=OFF -DG2O_BUILD_EXAMPLES=OFF ..
sudo make -j8 && sudo make install

cd ~/thirdparty/opencv
sudo rm -rf build4docker; sudo mkdir build4docker
cd build4docker
sudo cmake -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTS=OFF -DBUILD_PERF_TESTS=OFF ..
sudo make -j8 && sudo make install

cd ~/pymodules/exts/
sudo rm -rf build4docker; sudo mkdir build4docker
cd build4docker
sudo cmake -DCMAKE_BUILD_TYPE=Release ../py2ext
sudo make -j8

cd ~/pymodules
sudo rm -rf build
sudo python2 setup.py install

TFOPS=~/catkin_ws/src/ros_bonet/scripts/bonet/tf_ops
cd $TFOPS/grouping/ && sudo ./tf_grouping_compile.sh
cd $TFOPS/interpolation/ && sudo ./tf_interpolate_compile.sh
cd $TFOPS/sampling/ && sudo ./tf_sampling_compile.sh

cd $CWD
