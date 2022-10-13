rm -rf ~/.local/lib/python2.7/site-packages/unet_*
rm -rf ~/.local/lib/python3.6/site-packages/unet_*

CWD=$PWD
rm -rf build2; mkdir build2 && cd build2
cmake ../py2ext -DCMAKE_BUILD_TYPE=Release && make -j8
cd ..
rm -rf build3; mkdir build3 && cd build3
cmake ../py3ext -DCMAKE_BUILD_TYPE=Release && make -j8
cd ..

cd $CWD

cd ~/catkin_ws
#catkin clean -y ros_unet
catkin build ros_unet --cmake-args -DCMAKE_BUILD_TYPE=Release && \
          cp ~/catkin_ws/build/ros_unet/compile_commands.json ~/.vim
if [ $? -ne 0 ] ; then
  echo "Build failure"
  cd $PKG
  return 0
fi
cd $CWD

python2 -c "import unet_ext; print(unet_ext.__file__)"
python3 -c "import unet_ext; print(unet_ext.__file__)"
python3 -c "import unetsegment; print(unetsegment.__file__)"

# Reference below for debug
# CWD=$PWD
# rm -rf build2; mkdir build2 && cd build2
# cmake ../py2ext -DCMAKE_BUILD_TYPE=Debug && make -j4
# cd $CWD
# gdb -q python2 -r tmp.py
