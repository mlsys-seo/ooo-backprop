#!/bin/bash

DOCKER_IMAGE="mlsys.duckdns.org:9999/ooo_backprop_single_gpu:latest"

echo "Running the single-GPU expr container..."
docker run --rm --gpus all \
    ${DOCKER_IMAGE} \
    ./workspace/expr/scripts/run_mobilenet_v3_expr_base.sh 32 0.5
