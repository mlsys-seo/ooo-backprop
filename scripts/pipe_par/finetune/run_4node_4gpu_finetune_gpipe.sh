#!/bin/bash

# set up Model
export MODEL="bert_24" # bert_12, bert_24, bert_36, bert_48
export GLOBAL_BATCH_SIZE=96
export MICRO_BATCH_SIZE=4
export MODULO_BATCH_SIZE=1
export NUM_TRAINING_STEP=30
export PIPELINE_STYLE="gpipe" # modulo, fastforward,, fastforward-push, gpipe
export TASK="finetune" # pretrain, finetune

# set up cluster
export MASTER_HOST=localhost
export MASTER_PORT=2232

NODE_HOST_LIST=( )
NODE_HOST_LIST[0]=127.0.0.1
NODE_HOST_LIST[1]=
NODE_HOST_LIST[2]=
NODE_HOST_LIST[3]=

export NODE_HOSTS_STRING=${NODE_HOST_LIST[@]}

export NUM_NODE=${#NODE_HOST_LIST[@]}

export NUM_WORKER_PER_NODE=1

# set up network
export DOCKER_IMAGE="mlsys.duckdns.org:9999/ooo_backprop_pipeline_parallel:latest"
export SSH_KEY_PATH="~/.ssh/id_rsa"
export SSH_ID="woals"

../../../expr/pipe_par/scripts/run.sh preset