#!/bin/bash

DOCKER_IMAGE="public.ecr.aws/bdsldocker/ooo_backprop_single_gpu:latest"

echo "Running the single-GPU expr container..."
docker run --rm --gpus all \
    ${DOCKER_IMAGE} \
    ./workspace/expr/scripts/run_densenet_expr_ooo.sh 32 24
