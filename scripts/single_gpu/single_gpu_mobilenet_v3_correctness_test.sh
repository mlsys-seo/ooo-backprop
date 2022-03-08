#!/bin/bash

DOCKER_IMAGE="public.ecr.aws/bdsldocker/ooo_backprop_single_gpu:latest"

echo "Running the single-GPU expr container..."
docker run --rm --gpus all \
    ${DOCKER_IMAGE} \
    ./workspace/correctness_test/scripts/run_mobilenet_correct_test.sh
