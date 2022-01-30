#!/bin/bash

export REMOTE_ARG=$1
SCRIPT_ROOT_PATH=$(dirname $0)

if [[ $REMOTE_ARG == "preset" ]]
then
    declare -a NODE_HOST_LIST=($NODE_HOSTS_STRING)
else
    source $SCRIPT_ROOT_PATH/setup.sh
fi

echo ""
echo "==================================== setup ===================================="
echo "MODEL CONFIGURE    :: MODEL_SIZE: "$MODEL_SIZE
echo "MODEL CONFIGURE    :: BATCH_SIZE: "$BATCH_SIZE
echo "MODEL CONFIGURE    :: REVERSE_FIRST_K: "$REVERSE_FIRST_K
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
echo "CLUSTER CONFIGURE  :: NUM_WORKER: "$NUM_WORKER
echo "CLUSTER CONFIGURE  :: NUM_SERVER: "$NUM_SERVER
echo "CLUSTER CONFIGURE  :: NUM_WORKER_PER_NODE: "$NUM_WORKER_PER_NODE
echo "CLUSTER CONFIGURE  :: NUM_SERVER_PER_NODE: "$NUM_SERVER_PER_NODE
echo "CLUSTER CONFIGURE  :: DEBUG_PRINT: "$DEBUG_PRINT
echo "CLUSTER CONFIGURE  :: DEBUG_C_PRINT: "$DEBUG_C_PRINT
echo ""
echo "NETWORK CONFIGURE  :: DOCKER_IMAGE: "$DOCKER_IMAGE
echo "NETWORK CONFIGURE  :: SSH_KEY_PATH: "$SSH_KEY_PATH
echo "NETWORK CONFIGURE  :: SSH_ID: "$SSH_ID
echo "NETWORK CONFIGURE  :: DMLC_INTERFACE: "$DMLC_INTERFACE
echo "==============================================================================="
echo ""

$SCRIPT_ROOT_PATH/pull_image.sh

$SCRIPT_ROOT_PATH/kill_all.sh &&

echo ""
echo ""
echo "==============================================================================="
echo "::: RUN REMOTE SCHEDULER "$MASTER_HOST":"$MASTER_PORT" :::"

ssh -i $SSH_KEY_PATH -f $SSH_ID@$MASTER_HOST \
    sudo docker run -d \
        --rm --privileged --ipc=host --net=host --gpus=all \
        --name ooo-scheduler \
        $DOCKER_IMAGE \
        ./code/run_scheduler.sh $MASTER_HOST $MASTER_PORT $NUM_WORKER $NUM_SERVER &&

DETACH=""
for node_idx in "${!NODE_HOST_LIST[@]}"
do
    NODE_HOST="${NODE_HOST_LIST[$node_idx]}"

    for ((local_idx = 0 ; local_idx < $NUM_WORKER_PER_NODE ; local_idx++))
    do
        PRE_INDEX=`expr $node_idx \* $NUM_WORKER_PER_NODE`
        INDEX=`expr $PRE_INDEX + $local_idx`
        GPU_IDX=$local_idx

        echo ""
        echo "::: RUN REMOTE NODE "$node_idx": "$NODE_HOST" | IDX "$INDEX" | GPU_IDX: "$GPU_IDX" :::"
        ssh -i $SSH_KEY_PATH -f $SSH_ID@$NODE_HOST \
            "sudo docker run $DETACH \
                --rm --privileged --ipc=host --net=host --gpus=all \
                -e DMLC_INTERFACE=$DMLC_INTERFACE \
                --name ooo-worker-$INDEX \
                $DOCKER_IMAGE \
                ./code/run_node_resnet.sh \
                $MODEL_SIZE $BATCH_SIZE $NUM_TRAINING_STEP $REVERSE_FIRST_K $MASTER_HOST $MASTER_PORT $NODE_HOST $NUM_WORKER $NUM_SERVER $NUM_SERVER_PER_NODE $INDEX $GPU_IDX $DEBUG_PRINT $DEBUG_C_PRINT" &&
        echo ""
        
        if [[ $DETACH == "" ]]
        then
            DETACH="-d"
        fi
    done
done
echo "=================================== RUN DONE =================================="