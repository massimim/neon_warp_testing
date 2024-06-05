FROM nvcr.io/nvidia/cuda:12.2.0-devel-ubuntu22.04

LABEL maintainer='Massimiliano Meneghin'

# fix for keys
# from https://github.com/NVIDIA/nvidia-docker/issues/1631#issuecomment-1113105899
RUN \
  # Update nvidia GPG key
  rm -f /etc/apt/sources.list.d/cuda.list && \
  rm -f /etc/apt/sources.list.d/nvidia-ml.list && \
  apt-key del 7fa2af80 && \
  apt-get update && apt-get install -y --no-install-recommends wget && \
  wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb && \
  dpkg -i cuda-keyring_1.0-1_all.deb && \
  apt-get update

ADD setTimeZone.sh /setTimeZone.sh
RUN apt-get update
RUN apt-get install -y sudo gpg wget libtbb-dev
RUN /setTimeZone.sh; rm /setTimeZone.sh
RUN apt-get install -y emacs cmake clang-12 clang-13 clang-14 git gcc-10 gcc-12 


#--- [BEGIN] SSHD

RUN apt-get update
RUN apt-get -y install openssh-server
RUN systemctl enable ssh
RUN mkdir /var/run/sshd
# Add host keys
RUN cd /etc/ssh/ && ssh-keygen -A -N ''

RUN  sed -i "s/#PermitRootLogin.*/PermitRootLogin no/g" /etc/ssh/sshd_config \
  && sed -i "s/#AuthorizedKeysFile/AuthorizedKeysFile/g" /etc/ssh/sshd_config

# Default ssh config file that skips (yes/no) question when first login to the host
RUN echo "StrictHostKeyChecking no" > ${SSHDIR}/config
# This file can be overwritten by the following  step if ssh/ directory has config file

RUN apt-get install iproute2 -y

# RUN apt-get update && apt-get install -y openssh-server
# RUN sed -i 's/PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config

EXPOSE 22

#--- [END] OpenMPI
RUN apt-get install -y openmpi-bin openmpi-common libopenmpi-dev libgtk2.0-dev
#--- [BEGIN] OpenMPI

RUN apt-get install -y xorg-dev libglu1-mesa-dev freeglut3-dev mesa-common-dev

#--- [END] Python
RUN apt-get install -y python3
RUN apt-get install -y python3-pip
#--- [BEGIN] OpenMPI

RUN echo "export PATH=\"/usr/local/cuda/bin:$PATH\"" >> /etc/profile
RUN echo "LD_LIBRARY_PATH=\"/usr/local/cuda/lib64:$LD_LIBRARY_PATH\"">> /etc/profile

ARG USERNAME=ubuntu
ARG USER_UID=1000
ARG USER_GID=$USER_UID



# Create the user
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    #
    # [Optional] Add sudo support. Omit if you don't need to install software after connecting.
    && apt-get update \
    && apt-get install -y sudo \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

#RUN echo "ubuntu ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers
#RUN useradd -rm -d /home/ubuntu -s /bin/bash ubuntu
ADD test.sh /home/ubuntu
RUN chown ubuntu:root /home/ubuntu/test.sh
RUN chmod u+x  /home/ubuntu/test.sh

USER ubuntu
WORKDIR /home/ubuntu

RUN pip install numpy
CMD /bin/bash