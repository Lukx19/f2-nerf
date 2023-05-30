FROM nvidia/cuda:11.7.0-cudnn8-devel-ubuntu22.04

RUN apt-get update && apt-get install -y \
    git zlib1g-dev wget unzip cmake 7zip tmux \
    python3-pip gdb ccache colmap ffmpeg

RUN apt-get update && apt-get install -y htop nvtop mc

RUN apt-get update && apt-get install -y python-is-python3

RUN pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu117
RUN pip3 install click numpy omegaconf hydra-core opencv-python tqdm scipy matplotlib pandas


RUN useradd -ms /bin/bash user
USER user
WORKDIR /home/user
COPY .bashrc /home/user/
COPY .tmux.conf /home/user/

RUN mkdir -p /home/user/workspace
RUN mkdir -p /home/user/data

CMD ["bash"]