FROM nvidia/cuda:8.0-cudnn6-devel-ubuntu16.04

# TODO : 우분투 16.04에서는 뭔짓을 해도 rviz 안됨. 여기선 visualization 없이 rostopic 처리만 수행
#ref : https://roomedia.tistory.com/entry/4%EC%9D%BC%EC%B0%A8-Ubuntu-18042-LTS%EC%97%90-Docker-%EC%9D%B4%EC%9A%A9%ED%95%98%EC%97%AC-ROS-%EC%84%A4%EC%B9%98%ED%95%98%EA%B8%B0 
# https://stackoverflow.com/questions/59388345/rviz-in-nvidia-docker-container
ENV NVIDIA_VISIBLE_DEVICES \
    ${NVIDIA_VISIBLE_DEVICES:-all}
ENV NVIDIA_DRIVER_CAPABILITIES \
    ${NVIDIA_DRIVER_CAPABILITIES:+$NVIDIA_DRIVER_CAPABILITIES,}graphics

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

## Install ros-kinect, the latest ros for '16.04' 
RUN sudo sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
RUN sudo apt install -y curl && \
      curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | sudo apt-key add -

RUN sudo apt-get update && sudo apt-get install -y ros-kinetic-desktop

RUN sudo apt install -y \
      python-catkin-tools \
      python-pip python3-pip \
      gcc g++ curl
RUN pip install --upgrade "pip < 21.0" # Limit it due to depricated python 2.7
RUN echo "export PATH=/usr/local/cuda/bin:$PATH"  >> /home/$USERNAME/.bashrc
RUN echo "source /opt/ros/kinetic/setup.bash" >> /home/$USERNAME/.bashrc  # Must be after expanding PATH

# Install latest cmake
RUN sudo apt install -y software-properties-common lsb-release wget && sudo apt clean all 
RUN wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | gpg --dearmor - | sudo tee /etc/apt/trusted.gpg.d/kitware.gpg >/dev/null
# Install latest cmake
RUN sudo apt-add-repository "deb https://apt.kitware.com/ubuntu/ $(lsb_release -cs) main" && \
      sudo apt update && \
      sudo apt install -y kitware-archive-keyring && \
      sudo rm /etc/apt/trusted.gpg.d/kitware.gpg && \
      sudo apt install -y cmake
WORKDIR /home/$USERNAME

RUN mkdir -p catkin_ws/src/ && cd catkin_ws && catkin init
RUN echo "source ~/catkin_ws/devel/setup.bash" >> /home/$USERNAME/.bashrc  # Must be after expanding PATH

RUN sudo apt-get install -y git
RUN git clone https://github.com/eric-wieser/ros_numpy.git
RUN cd ros_numpy && python2 setup.py install --user

RUN pip install tensorflow-gpu==1.4.0 scipy==1.2.3 h5py==2.9 open3d-python==0.3.0


# << Useage - bash script >>
##docker build -t 3dbonet .
#XSOCK=/tmp/.X11-unix
#XAUTH=/tmp/.docker.xauth
#touch $XAUTH
#xauth nlist $DISPLAY | sed -e 's/^..../ffff/' | xauth -f $XAUTH nmerge -

#docker run -it --gpus all\
#  -e DISPLAY=unix$DISPLAY \
#  -e QT_X11_NO_MITSHM=1 \
#  -e XAUTHORITY=$XAUTH \
#  -v $XSOCK:$XSOCK \
#  -v $XAUTH:$XAUTH:rw \
#  -v $PWD/3D-BoNet:/home/docker/3D-BoNet \
#  3dbonet
