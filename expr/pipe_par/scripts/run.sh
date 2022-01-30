#!/bin/bash

export REMOTE_ARG=$1
CODE_ROOT_PATH=$(dirname $(dirname $(realpath $0)))
echo $CODE_ROOT_PATH

if [[ $REMOTE_ARG == "preset" ]]
then
    declare -a NODE_HOST_LIST=($NODE_HOSTS_STRING)
else
    source $SCRIPT_ROOT_PATH/setup.sh
fi

DATA_DIR="$CODE_ROOT_PATH/code/data"
CONFIG_DIR="$CODE_ROOT_PATH/code/bert_config"

echo ""
echo "==================================== setup ===================================="
echo "MODEL CONFIGURE    :: MODEL: "$MODEL
echo "MODEL CONFIGURE    :: GLOBAL_BATCH_SIZE: "$GLOBAL_BATCH_SIZE
echo "MODEL CONFIGURE    :: MICRO_BATCH_SIZE: "$MICRO_BATCH_SIZE
echo "MODEL CONFIGURE    :: MODULO_BATCH_SIZE: "$MODULO_BATCH_SIZE
echo "MODEL CONFIGURE    :: NUM_TRAINING_STEP: "$NUM_TRAINING_STEP
echo ""
echo "CLUSTER CONFIGURE  :: MASTER_HOST: "$MASTER_HOST
echo "CLUSTER CONFIGURE  :: MASTER_PORT: "$MASTER_PORT
for node_idx in "${!NODE_HOST_LIST[@]}"
do
    NODE_HOST="${NODE_HOST_LIST[$node_idx]}"
    echo "CLUSTER CONFIGURE  :: NODE_HOST_"$node_idx": "$NODE_HOST
done
echo "CLUSTER CONFIGURE  :: NUM_NODE: "$NUM_NODE
echo "CLUSTER CONFIGURE  :: NUM_WORKER_PER_NODE: "$NUM_WORKER_PER_NODE
echo ""
echo "NETWORK CONFIGURE  :: DOCKER_IMAGE: "$DOCKER_IMAGE
echo "NETWORK CONFIGURE  :: SSH_KEY_PATH: "$SSH_KEY_PATH
echo "NETWORK CONFIGURE  :: SSH_ID: "$SSH_ID
echo "==============================================================================="
echo ""

# set variables

WORKER_HOST_STRING=""
for node_idx in "${!NODE_HOST_LIST[@]}"
do
    NODE_HOST="${NODE_HOST_LIST[$node_idx]}"
    WORKER_HOST_STRING="$WORKER_HOST_STRING$NODE_HOST:$MASTER_PORT,"
done
WORKER_HOST_STRING="${WORKER_HOST_STRING:(0):(-1)}"
MODEL_CONFIG_PATH=$CONFIG_DIR/$MODEL.json

# pull docker images
$CODE_ROOT_PATH/scripts/pull_image.sh

# kill existing containers before running containers
$CODE_ROOT_PATH/scripts/kill_all.sh &&

# run scripts
echo ""
echo ""
echo "============================= RUNNING CONTAINERS =============================="
echo ""
for NODE_IDX in "${!NODE_HOST_LIST[@]}"
do
    NODE_HOST="${NODE_HOST_LIST[$NODE_IDX]}"
    if [ $NODE_IDX == 0 ]
    then
      echo "::: RUN MASTER NODE "$NODE_IDX": "$NODE_HOST" :::"
        docker run $DETACH \
                --rm --privileged --ipc=host --net=host --gpus=all \
                -e DMLC_INTERFACE=$DMLC_INTERFACE \
                --name ooo-pipe-$NODE_IDX \
                $DOCKER_IMAGE \
                ./code/run_node.sh \
                $TASK $MODEL $PIPELINE_STYLE $NUM_TRAINING_STEP $GLOBAL_BATCH_SIZE $MICRO_BATCH_SIZE $MODULO_BATCH_SIZE $NUM_WORKER_PER_NODE $NUM_NODE $MASTER_HOST $WORKER_HOST_STRING $NODE_IDX &
        echo ""
    else
      echo "::: RUN REMOTE WORKER NODE "$NODE_IDX": "$NODE_HOST" :::"
        ssh -i $SSH_KEY_PATH -f $SSH_ID@$NODE_HOST \
            "docker run $DETACH \
                --rm --privileged --ipc=host --net=host --gpus=all \
                -e DMLC_INTERFACE=$DMLC_INTERFACE \
                --name ooo-pipe-$NODE_IDX \
                $DOCKER_IMAGE \
                ./code/run_subnode.sh \
                $NODE_IDX $WORKER_HOST_STRING" &
        echo ""
    fi
done
echo ""
echo "=================================== RUN DONE =================================="