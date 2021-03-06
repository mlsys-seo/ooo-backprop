#!/bin/bash

# set up Model
export MODEL="bert_36"
export GLOBAL_BATCH_SIZE=380
export MICRO_BATCH_SIZE=38
export MODULO_BATCH_SIZE=1
export NUM_TRAINING_STEP=30
export PIPELINE_STYLE="fastforward_push"
export TASK="pretrain"
export MASTER_PORT=2232

# set up cluster (assign IP of MASTER_HOST)
export MASTER_HOST=              # Write your Cluster IP

# NODE_HOST_LIST[0] should be the IP of MASTER_HOST
NODE_HOST_LIST=( )
NODE_HOST_LIST[0]=               # Write your Cluster IP
NODE_HOST_LIST[1]=               # Write your Cluster IP
NODE_HOST_LIST[2]=               # Write your Cluster IP

export NODE_HOSTS_STRING=${NODE_HOST_LIST[@]}

export NUM_NODE=${#NODE_HOST_LIST[@]}

export NUM_WORKER_PER_NODE=8

# set up network
export DOCKER_IMAGE="hanyangbdsl/ooo_backprop_pipeline_parallel:latest"
export SSH_KEY_PATH="SSH_KEY_PATH"
export SSH_ID="ACCOUNT_ID"

../../../expr/pipe_par/scripts/run.sh preset