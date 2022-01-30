#!/bin/bash

# set up Model
export MODEL="bert_24" # bert_12, bert_24, bert_36, bert_48
export GLOBAL_BATCH_SIZE=96
export MICRO_BATCH_SIZE=8
export MODULO_BATCH_SIZE=2
export NUM_TRAINING_STEP=50
export PIPELINE_STYLE="modulo" # modulo, fastforward,, fastforward-push, gpipe
export TASK="finetune" # pretrain, finetune

# set up cluster
export MASTER_HOST=172.0.0.1
export MASTER_PORT=1234

export NODE_HOST_LIST=( )
NODE_HOST_LIST[0]=172.0.0.1

export NUM_NODE=${#NODE_HOST_LIST[@]}

export NUM_WORKER_PER_NODE=4

export DEBUG_PRINT=0
export DEBUG_C_PRINT=0

# set up network
export DOCKER_IMAGE="public.ecr.aws/bdsldocker/ooo_backprop_pipeline_parallel:latest"
export SSH_KEY_PATH="SSH_KEY_PATH"
export SSH_ID="SSH_ID"