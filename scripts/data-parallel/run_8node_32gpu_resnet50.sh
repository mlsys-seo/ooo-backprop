#!/bin/bash

# set up Model
export MODEL_SIZE=50
export BATCH_SIZE=128
export REVERSE_FIRST_K=30
export NUM_TRAINING_STEP=50

# set up cluster
export MASTER_HOST=127.0.0.1
export MASTER_PORT=1234

export NODE_HOST_LIST=( )
NODE_HOST_LIST[0]=127.0.0.1
NODE_HOST_LIST[1]=127.0.0.1
NODE_HOST_LIST[2]=127.0.0.1
NODE_HOST_LIST[3]=127.0.0.1
NODE_HOST_LIST[4]=127.0.0.1
NODE_HOST_LIST[5]=127.0.0.1
NODE_HOST_LIST[6]=127.0.0.1
NODE_HOST_LIST[7]=127.0.0.1

export NUM_NODE=${#NODE_HOST_LIST[@]}

# set up cluster-setting
export NUM_WORKER=32
export NUM_SERVER=8

export NUM_WORKER_PER_NODE=`expr $NUM_WORKER / $NUM_NODE`
export NUM_SERVER_PER_NODE=`expr $NUM_SERVER / $NUM_NODE`

export DEBUG_PRINT=0
export DEBUG_C_PRINT=0

# set up network
export DOCKER_IMAGE="mlsys.duckdns.org:9999/ooo-backprop-byteps:latest"
export SSH_KEY_PATH="~/.ssh/bdsl_rsa"
export SSH_ID="cheezestick"
export DMLC_INTERFACE=enp2s0f1

export OUTPUT_DIR=$(dirname $(realpath $0))/outputs


$(dirname $(dirname $(dirname $(realpath $0))))/expr/data-parallel/scripts/remote/run.sh preset