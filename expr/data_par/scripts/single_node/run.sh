#!/bin/bash

export REMOTE_ARG=$1
SCRIPT_ROOT_PATH=$(dirname $0)

if [[ $REMOTE_ARG != "preset" ]]
then
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
echo "CLUSTER CONFIGURE  :: NODE_HOST: "$NODE_HOST
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
echo ""
echo ""
echo "============================== PULL DOCKER IMAGE =============================="
sudo docker pull $DOCKER_IMAGE &&
echo "==============================================================================="
echo ""
echo ""
echo ""
echo "=============================== Kill Containers ==============================="
sudo docker kill ooo-scheduler

for ((INDEX = 0 ; INDEX < $NUM_WORKER_PER_NODE ; INDEX++))
do
    docker kill ooo-worker-$INDEX
done
echo "==============================================================================="
echo ""
echo ""
echo ""
echo "==============================================================================="
echo "::: RUN SCHEDULER "$MASTER_HOST":"$MASTER_PORT" :::"

sudo docker run -d \
    --rm --privileged --ipc=host --net=host --gpus=all \
    --name ooo-scheduler \
    $DOCKER_IMAGE \
    ./run_scheduler.sh $MASTER_HOST $MASTER_PORT $NUM_WORKER $NUM_SERVER &&

DETACH=""
for ((INDEX = 0 ; INDEX < $NUM_WORKER_PER_NODE ; INDEX++))
do
    GPU_IDX=$INDEX
    echo ""
    echo "::: RUN IDX "$INDEX" :::"
    sudo docker run -d \
        --rm --privileged --ipc=host --net=host --gpus=all \
        --name ooo-worker-$INDEX \
        $DOCKER_IMAGE \
        ./run_node_resnet.sh \
        $MODEL_SIZE $BATCH_SIZE $NUM_TRAINING_STEP $REVERSE_FIRST_K $MASTER_HOST $MASTER_PORT $NODE_HOST $NUM_WORKER $NUM_SERVER $NUM_SERVER_PER_NODE $INDEX $GPU_IDX $DEBUG_PRINT $DEBUG_C_PRINT &&
    echo ""
    
done
echo "=================================== RUN DONE =================================="

docker attach ooo-worker-$GPU_IDX