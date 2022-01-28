#!/bin/bash

source $(dirname $0)/setup.sh

./pull_image.sh

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