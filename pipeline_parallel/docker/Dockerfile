FROM nvidia/cuda:11.1.1-cudnn8-devel-ubuntu18.04

ARG https_proxy
ARG http_proxy

ARG BYTEPS_BASE_PATH=/usr/local

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
        libibverbs-dev \ # bazel

COPY . /workspace/
WORKDIR /workspace/

## FOR DATA PARALLEL
byteps

ARG FRAMEWORK=single-gpu
RUN if [ "$FRAMEWORK" = "single-gpu" ]; then \
        pip3 install --upgrade pip; \
        pip3 install -U tensorflow-gpu==1.15.0; \
    elif [ "$FRAMEWORK" = "pipeline-parallel" ]; then \
        pip3 install -U numpy==1.18.1 torchvision==0.5.0 torch==1.4.0; \
    elif [ "$FRAMEWORK" = "data-parallel" ]; then \
        pip3 install -U mxnet-cu100==1.5.0; \
    else \
        echo "unknown option: $FRAMEWORK"; \
        exit 1; \
    fi

ENV LD_LIBRARY_PATH /usr/local/cuda/lib64/stubs:$LD_LIBRARY_PATH


WORKDIR /workspace
RUN cd $BYTEPS_BASE_PATH &&\
    git clone --recursive -b $BYTEPS_BRANCH $BYTEPS_GIT_LINK &&\
    cd $BYTEPS_PATH &&\
    python3 setup.py install