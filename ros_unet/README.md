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
python2 scripts/unet/gen_obblabeling.py
```
* Whilte line for outline
* Yellow line for convex edge
* red-green dot for y axis of OBB orientation
  * Or, red-lue dot for z axis of OBB orientation
## Evaluation
```bash
roslaunch ros_unet eval.launch
```
