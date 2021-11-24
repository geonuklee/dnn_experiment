# Ubuntu 16.04.7 LTS
FROM nvidia/cuda:9.0-cudnn7-devel

# TODO : 우분투 16.04에서는 뭔짓을 해도 rviz 안됨. 여기선 visualization 없이 rostopic 처리만 수행
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

# Install ros-kinect, the latest ros for '16.04' 
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
RUN sudo apt-add-repository "deb https://apt.kitware.com/ubuntu/ $(lsb_release -cs) main" && \
      sudo apt update && \
      sudo apt install -y kitware-archive-keyring && \
      sudo rm /etc/apt/trusted.gpg.d/kitware.gpg && \
      sudo apt install -y cmake

# Download INSTALL_SCRIPT 
#   : https://unix.stackexchange.com/questions/379816/install-anaconda-in-ubuntu-docker
WORKDIR /home/$USERNAME
ARG INSTALL_SCRIPT=Anaconda3-2021.05-Linux-x86_64.sh
COPY $INSTALL_SCRIPT $INSTALL_SCRIPT
RUN yes "yes" | bash $INSTALL_SCRIPT # Benefit : automatic input naming, Disadventage : naming dir as yes

# https://stackoverflow.com/questions/57292146/problems-running-conda-update-in-a-dockerfile
ENV PATH /home/$USERNAME/yes/bin:$PATH
RUN conda update --all

RUN conda create -n pointgroup -y python==3.7
RUN conda install -y -c bioconda google-sparsehash
RUN conda install -y libboost
RUN conda install -y -c daleydeng gcc-5
RUN echo "conda activate pointgroup"  >> /home/$USERNAME/.bashrc

# ref: https://pythonspeed.com/articles/activate-conda-dockerfile/
# Make RUN commands use the new environment:
RUN echo "conda activate pointgroup" >> ~/.bashrc
SHELL ["/bin/bash", "--login", "-c"]

COPY PointGroup/requirements.txt requirements.txt 
RUN pip install -r requirements.txt | xargs echo

# << Useage - bash script >>
#docker build -t 3dbonet .
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
