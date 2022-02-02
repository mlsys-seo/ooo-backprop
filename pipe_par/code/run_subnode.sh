#!/bin/bash

CODE_DIR=$(dirname $(realpath $0))
ROOT_DIR=$(dirname $CODE_DIR)

NODE_IDX="${1}"
WORKER_HOST_STRING="${2}"

echo ""
echo ""
echo "################# SubNode(Worker) Start ##################"
echo "S_W NODE_IDX "$NODE_IDX
echo "S_W WORKER_HOST_STRING "$WORKER_HOST_STRING
echo "##########################################################"
echo ""

python3 $CODE_DIR/OOO_backprop/sub_node.py \
    --task_index=${NODE_IDX} \
    --worker_hosts=$WORKER_HOST_STRING