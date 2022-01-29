#!/bin/bash

# set up Model
export MODEL_SIZE=50
export BATCH_SIZE=128
export REVERSE_FIRST_K=30
export NUM_TRAINING_STEP=50

# set up cluster
export MASTER_HOST=127.0.0.1
# export MASTER_HOST=172.31.2.54
export MASTER_PORT=1234

export NODE_HOST_LIST=( )
NODE_HOST_LIST[0]=127.0.0.1
# NODE_HOST_LIST[0]=172.31.2.54
# NODE_HOST_LIST[1]=172.31.2.252
# NODE_HOST_LIST[2]=172.31.2.34
# NODE_HOST_LIST[3]=172.31.2.48

export NUM_NODE=${#NODE_HOST_LIST[@]}

# set up cluster-setting
export NUM_WORKER=4
export NUM_SERVER=1

export NUM_WORKER_PER_NODE=`expr $NUM_WORKER / $NUM_NODE`
export NUM_SERVER_PER_NODE=`expr $NUM_SERVER / $NUM_NODE`


# export INDEX=0
# export GPU_IDX=0
export DEBUG_PRINT=1
export DEBUG_C_PRINT=1

# set up network
# export DOCKER_IMAGE="195164261969.dkr.ecr.us-west-2.amazonaws.com/ooo_byteps:latest"
# export SSH_KEY_PATH="~/.ssh/hanyang_bdsl_oregon.pem"
# export SSH_ID="ubuntu"
# export DMLC_INTERFACE=ens3

export DOCKER_IMAGE="mlsys.duckdns.org:9999/ooo-backprop-byteps:latest"
export SSH_KEY_PATH="~/.ssh/bdsl_rsa"
export SSH_ID="cheezestick"
export DMLC_INTERFACE=enp2s0f1


export OUTPUT_DIR=$(dirname $(dirname $(dirname $(realpath $0))))