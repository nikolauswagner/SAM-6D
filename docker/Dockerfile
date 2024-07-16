FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04

# Config
ENV TZ=Europe/London
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
ENV ROS_DISTRO noetic

# Install ROS
RUN rm /etc/apt/sources.list.d/* && \
    DEBIAN_FRONTEND=noninteractive apt update --no-install-recommends && \
    DEBIAN_FRONTEND=noninteractive apt install -y software-properties-common lsb-release curl wget apt-transport-https git --no-install-recommends
RUN rm -rf /var/lib/apt/lists/* && \
    echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list && \
    apt-key adv --keyserver 'hkp://keyserver.ubuntu.com:80' --recv-key C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654
RUN DEBIAN_FRONTEND=noninteractive apt update && \
    DEBIAN_FRONTEND=noninteractive apt install -y ros-${ROS_DISTRO}-catkin ros-${ROS_DISTRO}-ros-base ros-${ROS_DISTRO}-vision-msgs ros-${ROS_DISTRO}-cv-bridge python3-catkin-tools python3-venv python3-pip python3-rosdep --no-install-recommends && \
    rosdep init && rosdep update

# Install dependencies
RUN pip3 install --pre torch==2.0.0 \
                 torchvision==0.15.1 \
                 fvcore \
                 xformers==0.0.18 \
                 torchmetrics==0.10.3 \
                 blenderproc==2.6.1 \
                 opencv-python  \
                 omegaconf \
                 ruamel.yaml \
                 hydra-colorlog \
                 hydra-core \
                 gdown \
                 pandas \
                 imageio \
                 pyrender \
                 pytorch-lightning==1.8.1 \
                 pycocotools  \
                 distinctipy \
                 git+https://github.com/facebookresearch/segment-anything.git \
                 ultralytics==8.0.135 \
                 timm \
                 gorilla-core==0.2.7.8 \
                 trimesh==4.0.8 \
                 gpustat==1.0.0 \
                 imgaug \
                 einops
RUN pip3 install scipy --upgrade
RUN pip3 uninstall -y torch torchvision
RUN pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu118
RUN DEBIAN_FRONTEND=noninteractive apt install -y ros-${ROS_DISTRO}-tf
RUN pip3 install ninja

RUN git clone https://github.com/nikolauswagner/SAM-6D.git
WORKDIR /SAM-6D/Pose_Estimation_Model/model/pointnet2
RUN python3 setup.py install
RUN ls /

COPY entrypoint.sh /entrypoint.sh
ENTRYPOINT ["/bin/bash", "/entrypoint.sh"]