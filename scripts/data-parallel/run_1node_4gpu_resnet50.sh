#!/bin/bash

# set up Model
export MODEL_SIZE=50
export BATCH_SIZE=128
export REVERSE_FIRST_K=30
export NUM_TRAINING_STEP=50

# set up cluster
export MASTER_HOST=127.0.0.1
export MASTER_PORT=1234

export NODE_HOST_LIST=( )
NODE_HOST_LIST[0]=127.0.0.1

export NUM_NODE=${#NODE_HOST_LIST[@]}

# set up cluster-setting
export NUM_WORKER=4
export NUM_SERVER=1

export NUM_WORKER_PER_NODE=`expr $NUM_WORKER / $NUM_NODE`
export NUM_SERVER_PER_NODE=`expr $NUM_SERVER / $NUM_NODE`

export DEBUG_PRINT=1
export DEBUG_C_PRINT=1

# set up network
export DOCKER_IMAGE="mlsys.duckdns.org:9999/ooo-backprop-byteps:latest"
export SSH_KEY_PATH="~/.ssh/bdsl_rsa"
export SSH_ID="cheezestick"
export DMLC_INTERFACE=enp2s0f1

export ROOT_DIR="/workspace/"

export OUTPUT_DIR=$(dirname $(dirname $(dirname $(realpath $0))))

echo ""
echo ""
echo "======================== setup.sh ========================"
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
echo "=========================================================="
echo ""
echo ""

echo "Killing Containers"
ssh -i $SSH_KEY_PATH $SSH_ID@$MASTER_HOST \
    docker kill ooo-scheduler

for node_idx in "${!NODE_HOST_LIST[@]}"
do
    NODE_HOST="${NODE_HOST_LIST[$node_idx]}"
    for ((local_idx = 0 ; local_idx < $NUM_WORKER_PER_NODE ; local_idx++))
    do
        PRE_INDEX=`expr $node_idx \* $NUM_WORKER_PER_NODE`
        INDEX=`expr $PRE_INDEX + $local_idx`

        ssh -i $SSH_KEY_PATH $SSH_ID@$NODE_HOST \
            docker kill ooo-worker-$INDEX
    done
done
echo "KILL FINISHED"

echo ""
echo "::: RUN REMOTE SCHEDULER "$MASTER_HOST":"$MASTER_PORT" :::"
ssh -i $SSH_KEY_PATH -f $SSH_ID@$MASTER_HOST \
    docker run -d \
        --rm --privileged --ipc=host --net=host --gpus=all \
        --name ooo-scheduler \
        $DOCKER_IMAGE \
        $ROOT_DIR/code/run_scheduler.sh $MASTER_HOST $MASTER_PORT $NUM_WORKER $NUM_SERVER &&


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
            docker run -d \
                --rm --privileged --ipc=host --net=host --gpus=all \
                -v $OUTPUT_DIR/outputs:$ROOT_DIR/outputs \
                -e DMLC_INTERFACE=$DMLC_INTERFACE \
                --name ooo-worker-$INDEX \
                $DOCKER_IMAGE \
                $ROOT_DIR/code/run_node_resnet.sh \
                $MODEL_SIZE $BATCH_SIZE $NUM_TRAINING_STEP $REVERSE_FIRST_K $MASTER_HOST $MASTER_PORT $NODE_HOST $NUM_WORKER $NUM_SERVER $NUM_SERVER_PER_NODE $INDEX $GPU_IDX $DEBUG_PRINT $DEBUG_C_PRINT &&
        echo ""
    done
done

echo "RUN DONE"

ssh -i $SSH_KEY_PATH -f $SSH_ID@$NODE_HOST \
    docker attach ooo-worker-0

# docker attach $FIRST_CONTAINER_NAME