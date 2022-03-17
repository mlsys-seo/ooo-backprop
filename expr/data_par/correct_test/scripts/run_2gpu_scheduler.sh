#!/bin/bash

export MODEL_SIZE=50

export MASTER_HOST=127.0.0.1
export MASTER_PORT=1234
export NODE_HOST=127.0.0.1
export NUM_WORKER=2
export NUM_SERVER=2


../code/run_scheduler_env.sh $MASTER_HOST $MASTER_PORT $NUM_SERVER $NUM_WORKER
