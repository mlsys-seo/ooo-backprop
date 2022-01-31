#!/bin/bash

# set up Model
export MODEL="bert_12"
export GLOBAL_BATCH_SIZE=480
export MICRO_BATCH_SIZE=40
export MODULO_BATCH_SIZE=1
export NUM_TRAINING_STEP=50
export PIPELINE_STYLE="gpipe"
export TASK="pretrain"

# set up cluster (assign IP of MASTER_HOST)
export MASTER_HOST=
export MASTER_PORT=2232

# NODE_HOST_LIST[0] should be the IP of MASTER_HOST
NODE_HOST_LIST=( )
NODE_HOST_LIST[0]=

export NODE_HOSTS_STRING=${NODE_HOST_LIST[@]}

export NUM_NODE=${#NODE_HOST_LIST[@]}

export NUM_WORKER_PER_NODE=4

export DEBUG_PRINT=0
export DEBUG_C_PRINT=0

# set up network
export DOCKER_IMAGE="public.ecr.aws/bdsldocker/ooo_backprop_pipeline_parallel:latest"
export SSH_KEY_PATH="~/.ssh/hanyang_bdsl_oregon.pem"
export SSH_ID="ubuntu"

../../../expr/pipe_par/scripts/run.sh preset