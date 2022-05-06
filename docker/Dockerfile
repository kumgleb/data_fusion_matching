FROM nvidia/cuda:11.3.0-cudnn8-devel-ubuntu20.04

#install required libs and packages
RUN apt-get update -y && \
    apt-get autoremove -y && \ 
    apt-get install -y --no-install-recommends \
    wget curl python3 python3-pip \
    ca-certificates libjpeg-dev libpng-dev lsb-release ninja-build build-essential git && \
    rm -rf /var/lib/apt/lists/*


ENV FORCE_CUDA="1"

ENV PATH=/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

RUN	pip install numpy scipy pandas tqdm 

RUN pip install torch==1.7.0+cu110 torchvision==0.8.1+cu110 -f https://download.pytorch.org/whl/torch_stable.html

WORKDIR /workspace