#!/bin/bash

# set up Model
export MODEL="bert_24" # bert_12, bert_24, bert_36, bert_48
export GLOBAL_BATCH_SIZE=96
export MICRO_BATCH_SIZE=8
export MODULO_BATCH_SIZE=2
export NUM_TRAINING_STEP=50
export PIPELINE_STYLE="fast-forward" # modulo, fastforward,, fastforward-push, gpipe
export TASK="pretrain" # pretrain, finetune

# set up cluster
export MASTER_HOST=172.31.2.164
export MASTER_PORT=1234

NODE_HOST_LIST=( )
NODE_HOST_LIST[0]=172.31.2.164
NODE_HOST_LIST[1]=172.31.2.212
NODE_HOST_LIST[2]=172.31.2.316

export NODE_HOSTS_STRING=${NODE_HOST_LIST[@]}

export NUM_NODE=${#NODE_HOST_LIST[@]}

export NUM_WORKER_PER_NODE=4

export DEBUG_PRINT=0
export DEBUG_C_PRINT=0

# set up network
export DOCKER_IMAGE="public.ecr.aws/bdsldocker/ooo_backprop_data_parallel:latest"
export SSH_KEY_PATH="~/.ssh/hanyang_bdsl_oregon.pem"
export SSH_ID="ubuntu"

if [[ $TASK == "pretrain" ]]
    then
    ../../expr/pipeline-parallel/scripts/run_pretrining.sh preset
    else
    ../../expr/pipeline-parallel/scripts/run_finetuning.sh preset
fi