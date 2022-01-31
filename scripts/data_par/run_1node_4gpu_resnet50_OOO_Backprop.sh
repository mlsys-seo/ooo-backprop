#!/bin/bash

# set up Model
export MODEL_SIZE=50
export BATCH_SIZE=128
export REVERSE_FIRST_K=45
export NUM_TRAINING_STEP=50

# set up cluster (assign IP of MASTER_HOST)
export MASTER_HOST=127.0.0.1
export MASTER_PORT=1234
export NODE_HOST=127.0.0.1
export NUM_NODE=1

# set up cluster (assign IP of MASTER_HOST)-setting
export NUM_WORKER=4
export NUM_SERVER=1

export NUM_WORKER_PER_NODE=`expr $NUM_WORKER / $NUM_NODE`
export NUM_SERVER_PER_NODE=`expr $NUM_SERVER / $NUM_NODE`

export DEBUG_PRINT=0
export DEBUG_C_PRINT=0

export DOCKER_IMAGE="public.ecr.aws/bdsldocker/ooo_backprop_data_parallel:latest"

../../expr/data_par/scripts/single_node/run.sh preset
