#!/bin/bash

export MODEL_SIZE=50
export MODEL_SIZE=50
export BATCH_SIZE=64
export NUM_TRAINING_STEP=30

export REVERSE_FIRST_K=20

export MASTER_HOST=10.0.0.3
export MASTER_PORT=1234

export NODE_HOST=10.0.0.4
export NUM_NODE=2

export NUM_WORKER=4
export NUM_SERVER=2

export NUM_WORKER_PER_NODE=`expr $NUM_WORKER / $NUM_NODE`
export NUM_SERVER_PER_NODE=`expr $NUM_SERVER / $NUM_NODE`

export INDEX=2
export GPU_IDX=0
export DEBUG_PRINT=0

export DMLC_INTERFACE=ens10f4

DOCKER_IMAGE="mlsys.duckdns.org:9999/ooo-backprop-byteps:latest"
ROOT_DIR="/root/OOO_BackProp_BytePS"

docker run \
	--rm --privileged --ipc=host --net=host --gpus=all \
	--name ooo-worker-$INDEX \
	-e DMLC_INTERFACE=$DMLC_INTERFACE \
	$DOCKER_IMAGE \
	$ROOT_DIR/code/run_node_resnet.sh \
	$MODEL_SIZE $BATCH_SIZE $NUM_TRAINING_STEP $REVERSE_FIRST_K $MASTER_HOST $MASTER_PORT $NODE_HOST $NUM_WORKER $NUM_SERVER $NUM_SERVER_PER_NODE $INDEX $GPU_IDX $DEBUG_PRINT