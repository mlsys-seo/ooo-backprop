#!/bin/bash

# set up Model
export MODEL_SIZE=50
export BATCH_SIZE=128
export REVERSE_FIRST_K=45
export NUM_TRAINING_STEP=50

# set up cluster
export MASTER_HOST=127.0.0.1
export MASTER_PORT=1234

NODE_HOST_LIST=( )
NODE_HOST_LIST[0]="NODE IP"
NODE_HOST_LIST[1]="NODE IP"
NODE_HOST_LIST[2]="NODE IP"
NODE_HOST_LIST[3]="NODE IP"
NODE_HOST_LIST[4]="NODE IP"
NODE_HOST_LIST[5]="NODE IP"
NODE_HOST_LIST[6]="NODE IP"
NODE_HOST_LIST[7]="NODE IP"

export NODE_HOSTS_STRING=${NODE_HOST_LIST[@]}

export NUM_NODE=${#NODE_HOST_LIST[@]}

# set up cluster-setting
export NUM_WORKER=32
export NUM_SERVER=8

export NUM_WORKER_PER_NODE=`expr $NUM_WORKER / $NUM_NODE`
export NUM_SERVER_PER_NODE=`expr $NUM_SERVER / $NUM_NODE`

export DEBUG_PRINT=0
export DEBUG_C_PRINT=0

# set up network
export DOCKER_IMAGE="public.ecr.aws/bdsldocker/ooo_backprop_data_parallel:latest"
export SSH_KEY_PATH="SSH_KEY_PATH"
export SSH_ID="ACCOUNT_ID"
export DMLC_INTERFACE="NETWORK INTERFACE"

../../expr/data-parallel/scripts/multi_node/run.sh preset