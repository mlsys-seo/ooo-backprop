#!/bin/bash

# set up Model
export MODEL="bert_24"
export GLOBAL_BATCH_SIZE=96
export MICRO_BATCH_SIZE=4
export MODULO_BATCH_SIZE=1
export NUM_TRAINING_STEP=30
export PIPELINE_STYLE="fastforward"
export TASK="finetune"
export MASTER_PORT=2232

# set up cluster (assign IP of MASTER_HOST)
export MASTER_HOST=               # Write your Cluster IP

# NODE_HOST_LIST[0] should be the IP of MASTER_HOST
NODE_HOST_LIST=( )
NODE_HOST_LIST[0]=                # Write your Cluster IP

export NODE_HOSTS_STRING=${NODE_HOST_LIST[@]}

export NUM_NODE=${#NODE_HOST_LIST[@]}

export NUM_WORKER_PER_NODE=4

# set up network
export DOCKER_IMAGE="public.ecr.aws/bdsldocker/ooo_backprop_pipeline_parallel:latest"
export SSH_KEY_PATH=              # Write your SSH Key
export SSH_ID=                    # Write your SSH ID

../../../expr/pipe_par/scripts/run.sh preset