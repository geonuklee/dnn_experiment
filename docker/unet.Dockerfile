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
#RUN sudo apt install -y x11vnc xvfb fluxbox htop
#RUN echo "if ps -al | grep "x11vnc"; then" >> /home/$USERNAME/.bashrc && \
#  echo "echo \"\" " >> /home/$USERNAME/.bashrc && \
#  echo "else" >> /home/$USERNAME/.bashrc && \
#  echo "export DISPLAY=:1" >> /home/$USERNAME/.bashrc && \
#  echo "Xvfb \$DISPLAY -screen 0 1920x1680x24 & fluxbox & x11vnc -display \$DISPLAY -forever -localhost -rfbport 5901&" >> /home/$USERNAME/.bashrc && \
#  echo "fi" >> /home/$USERNAME/.bashrc

RUN sudo apt-get install -y python3 python3-pip libgl1 # libgl1 for opencv
RUN pip3 install pip --upgrade && pip3 install future --upgrade --user && pip3 install --user torch torchvision numpy opencv-python

RUN sudo apt-get install -y ssh

WORKDIR /home/$USERNAME
