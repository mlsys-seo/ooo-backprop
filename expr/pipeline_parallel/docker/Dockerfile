FROM nvidia/cuda:11.1.1-cudnn8-devel-ubuntu18.04

# install packages
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    wget \
    vim \
    cmake \
    net-tools \
    python3 \
    python3-dev \
    python3-pip \
    python3-setuptools \
    libcublas-dev-11-1 \
    libcudnn8-dev=8.0.5.39-1+cuda11.1 \
    libnccl2=2.8.4-1+cuda11.1 \
    libnccl-dev=2.8.4-1+cuda11.1 \
    lsb-release \
    libnuma-dev \
    ibverbs-providers \
    librdmacm-dev \
    ibverbs-utils \
    rdmacm-utils \
    libibverbs-dev

# install python packages
RUN pip3 install --upgrade pip &&\
    pip3 install \
        tensorflow==2.4.1

COPY . /workspace/

# patch tensorflow
RUN cd /usr/local/lib/python3.6/dist-packages/tensorflow &&\
    git patch -0 < ./modules/tensorflow/patchfile`

ENV LD_LIBRARY_PATH /usr/local/cuda/lib64/stubs:$LD_LIBRARY_PATH

WORKDIR /workspace
CMD /bin/bash