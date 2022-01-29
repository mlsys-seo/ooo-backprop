#!/bin/bash

if [$1 != "pre-preset"]
then
    source $(dirname $0)/setup.sh
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
echo ""
echo "NETWORK CONFIGURE  :: DOCKER_IMAGE: "$DOCKER_IMAGE
echo "NETWORK CONFIGURE  :: SSH_KEY_PATH: "$SSH_KEY_PATH
echo "NETWORK CONFIGURE  :: SSH_ID: "$SSH_ID
echo "NETWORK CONFIGURE  :: DMLC_INTERFACE: "$DMLC_INTERFACE
echo "NETWORK CONFIGURE  :: ROOT_DIR: "$ROOT_DIR
echo ""
echo "RESULT CONFIGURE :: OUTPUT_DIR: "$OUTPUT_DIR
echo "==============================================================================="
echo ""

export DOCKER_ROOT_DIR="/workspace"

$(dirname $(realpath $0))/pull_image.sh

$(dirname $(realpath $0))/kill_all.sh

echo ""
echo "::: RUN REMOTE SCHEDULER "$MASTER_HOST":"$MASTER_PORT" :::"
ssh -i $SSH_KEY_PATH -f $SSH_ID@$MASTER_HOST \
    docker run -d \
        --rm --privileged --ipc=host --net=host --gpus=all \
        --name ooo-scheduler \
        $DOCKER_IMAGE \
        $DOCKER_ROOT_DIR/code/run_scheduler.sh $MASTER_HOST $MASTER_PORT $NUM_WORKER $NUM_SERVER &&

DETACH=""
for node_idx in "${!NODE_HOST_LIST[@]}"
do
    NODE_HOST="${NODE_HOST_LIST[$node_idx]}ssh "

    for ((local_idx = 0 ; local_idx < $NUM_WORKER_PER_NODE ; local_idx++))
    do
        PRE_INDEX=`expr $node_idx \* $NUM_WORKER_PER_NODE`
        INDEX=`expr $PRE_INDEX + $local_idx`
        GPU_IDX=$local_idx

        if [ $DETACH = "" ]
        then
            DETACH="-d"
        fi

        echo ""
        echo "::: RUN REMOTE NODE "$node_idx": "$NODE_HOST" | IDX "$INDEX" | GPU_IDX: "$GPU_IDX" :::"
        ssh -i $SSH_KEY_PATH -f $SSH_ID@$NODE_HOST \
            docker run $DETACH \
                --rm --privileged --ipc=host --net=host --gpus=all \
                -v $OUTPUT_DIR:$DOCKER_ROOT_DIR/outputs \
                -e DMLC_INTERFACE=$DMLC_INTERFACE \
                --name ooo-worker-$INDEX \
                $DOCKER_IMAGE \
                $DOCKER_ROOT_DIR/code/run_node_resnet.sh \
                $MODEL_SIZE $BATCH_SIZE $NUM_TRAINING_STEP $REVERSE_FIRST_K $MASTER_HOST $MASTER_PORT $NODE_HOST $NUM_WORKER $NUM_SERVER $NUM_SERVER_PER_NODE $INDEX $GPU_IDX $DEBUG_PRINT $DEBUG_C_PRINT &&
        echo ""
    done
done

echo "RUN DONE"

ssh -i $SSH_KEY_PATH -f $SSH_ID@$NODE_HOST \
    docker attach ooo-worker-0

# docker attach $FIRST_CONTAINER_NAME