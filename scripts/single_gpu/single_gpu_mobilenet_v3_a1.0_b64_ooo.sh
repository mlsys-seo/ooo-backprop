#!/bin/bash

DOCKER_IMAGE="hanyangbdsl/ooo_backprop_single_gpu:latest"

echo "Running the single-GPU expr container..."
docker run --rm --gpus all \
    ${DOCKER_IMAGE} \
    ./workspace/expr/scripts/run_mobilenet_v3_expr_ooo.sh 64 1.0
