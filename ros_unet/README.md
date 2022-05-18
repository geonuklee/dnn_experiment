# Prebuilt

```bash
sudo apt install python3-ros # You might be need to reinstall ros-melodic-desktop again.
pip2 install --user glob2
sudo apt install ros-melodic-ros-numpy
```

* [ ] TODO : helios2 arena, kinect installation.

# Dataset
## Architecture
* pkg\_root
  * vtk\_dataset
    * src
    * cache
  * segment\_dataset
    * src
    * cache
  * obb\_dataset
    * ~.png, ~.pick
  * rosbag
  * rviz
##

# DUNet
## Generate dataset
```bash
python2 scripts/unet/gen_labeling.py
```

## Train
```bash
python3 scripts/unet/train_dunet.py
```

# OBB
## Generate dataset - for evaluation
```bash
roslaunch ros_unet label.launch
```
* Whilte line for outline
* Yellow line for convex edge
* red-blue dot for one axis.
## Evaluation
```bash
roslaunch ros_unet eval.launch
```
