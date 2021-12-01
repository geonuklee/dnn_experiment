FROM nvidia/cuda:11.4.1-cudnn8-runtime-ubuntu18.04

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

# Assign port 5901 to avoid collision with vnc of host
RUN sudo DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends tzdata &&\
  sudo apt install -y x11vnc xvfb fluxbox htop
RUN echo "if ps -al | grep "x11vnc"; then" >> /home/$USERNAME/.bashrc && \
  echo "echo \"\" " >> /home/$USERNAME/.bashrc && \
  echo "else" >> /home/$USERNAME/.bashrc && \
  echo "export DISPLAY=:1" >> /home/$USERNAME/.bashrc && \
  echo "Xvfb \$DISPLAY -screen 0 1920x1680x24 & fluxbox & x11vnc -display \$DISPLAY -forever -localhost -rfbport 5901&" >> /home/$USERNAME/.bashrc && \
  echo "fi" >> /home/$USERNAME/.bashrc

RUN sudo apt-get install -y python3 python3-pip libgl1 # libgl1 for opencv
RUN pip3 install pip --upgrade && pip3 install future --upgrade --user && pip3 install --user torch torchvision numpy opencv-python

RUN sudo apt-get install -y ssh

RUN sudo sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
RUN sudo apt install -y curl && \
      curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | sudo apt-key add -

# Install python3-rospkg to avoid install ros-melodic-desktop --fix-missing again.
# ref) https://medium.com/@inderpreetsinghchhabra/using-python3-with-ros-kinetic-2488354efece
RUN sudo apt-get update && sudo apt-get install -y python3-rospkg && \
  sudo apt-get install -y ros-melodic-desktop 

RUN sudo apt install -y \
      python-catkin-tools \
      python-pip python3-pip \
      gcc g++ curl
RUN pip install --upgrade "pip < 21.0" # Limit it due to depricated python 2.7
RUN echo "export PATH=/usr/local/cuda/bin:$PATH"  >> /home/$USERNAME/.bashrc
RUN echo "source /opt/ros/melodic/setup.bash" >> /home/$USERNAME/.bashrc  # Must be after expanding PATH

# Make catkin workspace
WORKDIR /home/$USERNAME
RUN mkdir -p catkin_ws/src/ && cd catkin_ws && catkin init
RUN echo "source ~/catkin_ws/devel/setup.bash" >> /home/$USERNAME/.bashrc  # Must be after expanding PATH


WORKDIR /home/$USERNAME
