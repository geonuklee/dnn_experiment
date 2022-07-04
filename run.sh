PKG=$PWD

cd ~/pymodules/exts/build4docker
make -j8
cd ~/pymodules
python2 setup.py install --user

cd $PKG
rm -rf /home/docker/obb_dataset_train/cached_block/
#python2 scripts/dataset_train.py
#python2 scripts/dataset_test.py
