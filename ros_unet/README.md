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

# Generate Dataset

```bash
python2 scripts/unet/gen_labeling.py
```

# Train UNet

```bash
python3 scripts/unet/train_dunet.py
```

