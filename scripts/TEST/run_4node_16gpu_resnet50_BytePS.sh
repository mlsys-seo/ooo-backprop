#!/bin/bash

# set up Model
export MODEL_SIZE=50
export BATCH_SIZE=128
export REVERSE_FIRST_K=45
export NUM_TRAINING_STEP=50

# set up cluster
export MASTER_HOST=172.31.2.164
export MASTER_PORT=1234

NODE_HOST_LIST=( )
NODE_HOST_LIST[0]=172.31.2.164
NODE_HOST_LIST[1]=172.31.2.212
NODE_HOST_LIST[2]=172.31.2.125
NODE_HOST_LIST[3]=172.31.2.232

export NODE_HOSTS_STRING=${NODE_HOST_LIST[@]}

export NUM_NODE=${#NODE_HOST_LIST[@]}

# set up cluster-setting
export NUM_WORKER=16
export NUM_SERVER=4

export NUM_WORKER_PER_NODE=`expr $NUM_WORKER / $NUM_NODE`
export NUM_SERVER_PER_NODE=`expr $NUM_SERVER / $NUM_NODE`

export DEBUG_PRINT=0
export DEBUG_C_PRINT=0

# set up network
export DOCKER_IMAGE="public.ecr.aws/bdsldocker/ooo_backprop_data_parallel:latest"
export SSH_KEY_PATH="~/.ssh/hanyang_bdsl_oregon.pem"
export SSH_ID="ubuntu"

../../expr/data-parallel/scripts/multi_node/run.sh preset