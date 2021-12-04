#filename=~/dataset/unloading/stc2021/stc_2021-08-19-11-48-10.bag
#filename=~/dataset/unloading/stc2021/stc_2021-08-19-14-51-21.bag
#filename=~/dataset/unloading/stc2021/stc_2021-08-19-14-52-07.bag
#filename=~/dataset/unloading/stc2021/stc_2021-08-19-17-05-03.bag
#filename=~/dataset/unloading/stc2021/stc_2021-08-19-17-07-40.bag
#filename=~/dataset/unloading/stc2021/stc_2021-08-19-17-08-57.bag
#filename=~/dataset/unloading/stc2021/stc_2021-08-19-17-10-55.bag
#filename=~/dataset/unloading/stc2021/stc_2021-08-19-17-11-34.bag
#filename=~/dataset/unloading/stc2021/stc_2021-09-02-10-28-32.bag
#filename=~/dataset/unloading/stc2021/stc_2021-09-02-12-15-02.bag  # << Poor for train
#filename=~/dataset/unloading/stc2021/stc_2021-09-02-13-02-02.bag
#filename=~/dataset/unloading/stc2021/stc_2021-09-02-18-50-18.bag

files=(
~/dataset/unloading/stc2021/stc_2021-08-19-11-48-10.bag \
~/dataset/unloading/stc2021/stc_2021-08-19-14-51-21.bag \
~/dataset/unloading/stc2021/stc_2021-08-19-14-52-07.bag \
~/dataset/unloading/stc2021/stc_2021-08-19-17-05-03.bag \
~/dataset/unloading/stc2021/stc_2021-08-19-17-07-40.bag \
~/dataset/unloading/stc2021/stc_2021-08-19-17-08-57.bag \
~/dataset/unloading/stc2021/stc_2021-08-19-17-10-55.bag \
~/dataset/unloading/stc2021/stc_2021-08-19-17-11-34.bag \
~/dataset/unloading/stc2021/stc_2021-09-02-10-28-32.bag \
~/dataset/unloading/stc2021/stc_2021-09-02-12-15-02.bag \
~/dataset/unloading/stc2021/stc_2021-09-02-13-02-02.bag \
~/dataset/unloading/stc2021/stc_2021-09-02-18-50-18.bag 
)

for filename in ${files[@]}
do
    echo $filename
    rosbag info $filename | grep "/cam.*/helios2/depth/image_raw"
    rosbag info $filename | grep "/cam.*/aligned/rgb_to_depth/image_raw"
    roslaunch dnn_experiment unet.launch filename:=$filename
done


