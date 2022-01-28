#!/bin/bash

export MODEL_SIZE=50
export BATCH_SIZE=128

export REVERSE_FIRST_K=20

export MASTER_HOST=10.0.0.3
export MASTER_PORT=1234

export NODE_HOST=10.0.0.3
export NUM_NODE=1

export NUM_WORKER=2
export NUM_SERVER=1

export NUM_WORKER_PER_NODE=`expr $NUM_WORKER / $NUM_NODE`
export NUM_SERVER_PER_NODE=`expr $NUM_SERVER / $NUM_NODE`

export DMLC_INTERFACE=ens10f4

DOCKER_IMAGE="mlsys.duckdns.org:9999/ooo-backprop-byteps:latest"
ROOT_DIR="/root/OOO_BackProp_BytePS"

docker run \
    --rm --privileged --ipc=host --net=host --gpus=all \
    --name ooo-scheduler \
	-e DMLC_INTERFACE=$DMLC_INTERFACE \
    $DOCKER_IMAGE \
	$ROOT_DIR/code/run_scheduler.sh $MASTER_HOST $MASTER_PORT $NUM_WORKER $NUM_SERVER