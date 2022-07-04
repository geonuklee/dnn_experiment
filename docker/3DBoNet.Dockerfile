FROM nvidia/cuda:8.0-cudnn6-devel-ubuntu16.04
RUN rm /etc/apt/sources.list.d/cuda.list && rm /etc/apt/sources.list.d/nvidia-ml.list

RUN sed -i 's/archive.ubuntu.com/ftp.daumkakao.com/g' /etc/apt/sources.list && \
      sed -i 's/security.ubuntu.com/ftp.daumkakao.com/g' /etc/apt/sources.list && \
      apt-get update && apt-get install -y lsb-release 

# https://stackoverflow.com/questions/65427262/add-sudo-permission-without-password-to-user-by-command-line
ENV USERNAME docker
RUN apt-get install -y sudo && \
      useradd -m $USERNAME && \
      echo "$USERNAME:$USERNAME" | chpasswd && \
      usermod --shell /bin/bash $USERNAME && \
      usermod -aG sudo $USERNAME && \
      echo "$USERNAME ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers.d/$USERNAME && \
      chmod 0440 /etc/sudoers.d/$USERNAME && \
      usermod  --uid 1000 $USERNAME && \  
      groupmod --gid 1000 $USERNAME
USER docker
WORKDIR /home/$USERNAME

## Install ros-kinect, the latest ros for '16.04' 
RUN sudo sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
RUN sudo apt install -y curl && \
      curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | sudo apt-key add -

RUN sudo apt-get update && sudo apt-get install -y \
  ros-kinetic-desktop \
  ros-kinetic-pcl-ros \
  ros-kinetic-voxel-grid \
  ros-kinetic-nodelet

RUN sudo apt install -y \
      python-catkin-tools \
      python-pip python3-pip \
      gcc g++ curl
RUN pip install --upgrade "pip < 21.0" # Limit it due to depricated python 2.7
RUN echo "export PATH=/usr/local/cuda/bin:$PATH"  >> /home/$USERNAME/.bashrc
RUN echo "source /opt/ros/kinetic/setup.bash" >> /home/$USERNAME/.bashrc  # Must be after expanding PATH

# Install latest cmake
RUN sudo apt install -y software-properties-common lsb-release wget && sudo apt clean all 
RUN sudo apt install -y cmake cmake-curses-gui

#RUN sudo apt install -y software-properties-common && \
#    sudo add-apt-repository -y ppa:ubuntu-toolchain-r/test && \
#    sudo apt-get update && \
#    sudo apt install -y gcc-8 g++-8 && \
#    sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-8 1 && \
#    sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-8 1
#RUN wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | gpg --dearmor - | sudo tee /etc/apt/trusted.gpg.d/kitware.gpg >/dev/null
## Install latest cmake
#RUN sudo apt-add-repository "deb https://apt.kitware.com/ubuntu/ $(lsb_release -cs) main" && \
#      sudo apt update && \
#      sudo apt install -y kitware-archive-keyring && \
#      sudo rm /etc/apt/trusted.gpg.d/kitware.gpg && \
#      sudo apt install -y cmake cmake-curses-gui

RUN mkdir -p catkin_ws/src/ && cd catkin_ws && catkin init
RUN echo "source ~/catkin_ws/devel/setup.bash" >> /home/$USERNAME/.bashrc  # Must be after expanding PATH

RUN sudo apt-get install -y git
RUN git clone https://github.com/eric-wieser/ros_numpy.git
RUN cd ros_numpy && python2 setup.py install --user

RUN pip install tensorflow-gpu==1.4.0 scipy==1.2.3 h5py==2.9 open3d-python==0.3.0
RUN pip install pybind11

# pip must be 20.0.1 : https://github.com/pypa/pip/issues/7620
RUN python2 -m pip install --user --force-reinstall pip==20.0.1 &&\
    python2 -m pip install numexpr==2.6.2 tables==3.5.2

# torch to comparing with 'unet' with ObbDataset which depend on torch
RUN python2 -m pip install future torch torchvision glob2

RUN sudo apt install libgtk2.0-dev -y

RUN sudo apt install -y x11vnc xvfb fluxbox htop && \
  echo "if ps -al | grep "x11vnc"; then" >> /home/$USERNAME/.bashrc && \
  echo "  echo \"\" " >> /home/$USERNAME/.bashrc && \
  echo "else" >> /home/$USERNAME/.bashrc && \
  echo "  export DISPLAY=:20 # The default display of Xvfb" >> /home/$USERNAME/.bashrc && \
  echo "  # with docker run -p 59xx:5902 -it TAG" >> /home/$USERNAME/.bashrc && \
  echo "  x11vnc -allow 172.17.0.1 -geometry 1920x1024 -nopw -q -bg -xkb -display \$DISPLAY -forever -create -rfbport 5900 &" >> /home/$USERNAME/.bashrc && \
  echo "  #with docker run --network host -it TAG" >> /home/$USERNAME/.bashrc && \
  echo "  #x11vnc -localhost -geometry 1920x1024 -nopw -q -bg -xkb -display \$DISPLAY -forever -create -rfbport 5900 &" >> /home/$USERNAME/.bashrc && \
  echo "fi" >> /home/$USERNAME/.bashrc

