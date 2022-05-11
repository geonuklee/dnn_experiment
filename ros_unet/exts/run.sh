rm ~/.local/lib/python2.7/site-packages/unet*
rm ~/.local/lib/python3.6/site-packages/unet*

CWD=$PWD
rm -rf build2; mkdir build2 && cd build2
cmake ../py2ext -DCMAKE_BUILD_TYPE=Release && make -j8
cd $CWD

rm -rf build3; mkdir build3 && cd build3
cmake ../py3ext -DCMAKE_BUILD_TYPE=Release && make -j8
cd $CWD

echo "Check installation of unet"
python2 -c "import unet_ext; print(unet_ext.__file__)"
python3 -c "import unet_ext; print(unet_ext.__file__)"
