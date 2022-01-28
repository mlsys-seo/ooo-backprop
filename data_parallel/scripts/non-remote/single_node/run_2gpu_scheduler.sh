#!/bin/bash

export MODEL_SIZE=50
export BATCH_SIZE=5

export MASTER_HOST=127.0.0.1
export MASTER_PORT=1234
export NODE_HOST=127.0.0.1
export NUM_WORKER=2
export NUM_SERVER=2

ROOT_DIR=$(dirname $(dirname $(dirname $(realpath $0))))

$ROOT_DIR/examples/run_scheduler.sh $MASTER_HOST $MASTER_PORT $NUM_WORKER $NUM_SERVER 