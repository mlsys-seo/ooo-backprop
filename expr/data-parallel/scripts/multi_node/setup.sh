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
# NODE_HOST_LIST[1]=172.31.2.252
# NODE_HOST_LIST[2]=172.31.2.34
# NODE_HOST_LIST[3]=172.31.2.48

export NUM_NODE=${#NODE_HOST_LIST[@]}

# set up cluster-setting
export NUM_WORKER=4
export NUM_SERVER=1

export NUM_WORKER_PER_NODE=`expr $NUM_WORKER / $NUM_NODE`
export NUM_SERVER_PER_NODE=`expr $NUM_SERVER / $NUM_NODE`

export DEBUG_PRINT=1
export DEBUG_C_PRINT=1

export DOCKER_IMAGE="mlsys.duckdns.org:9999/ooo_backprop_data_parallel:latest"
export SSH_KEY_PATH="SSH_KEY_PATH"
export SSH_ID="ACCOUNT_ID"
export DMLC_INTERFACE="NETWORK INTERFACE"