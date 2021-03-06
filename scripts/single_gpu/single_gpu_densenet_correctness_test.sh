#!/bin/bash

DOCKER_IMAGE="hanyangbdsl/ooo_backprop_single_gpu:latest"

sudo docker pull $DOCKER_IMAGE

echo "Running the single-GPU expr container..."
docker run --rm --gpus all \
    ${DOCKER_IMAGE} \
    ./workspace/correctness_test/scripts/run_densenet_correct_test.sh
