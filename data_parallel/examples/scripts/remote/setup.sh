#!/bin/bash

# set up Model
export MODEL_SIZE=50
export BATCH_SIZE=128
export REVERSE_FIRST_K=30
export NUM_TRAINING_STEP=50

# set up cluster
export MASTER_HOST=127.0.0.1
export MASTER_PORT=1234

export NODE_HOST_LIST=( )
NODE_HOST_LIST[0]=127.0.0.1
# NODE_HOST_LIST[1]=172.31.2.252
# NODE_HOST_LIST[2]=172.31.2.34
# NODE_HOST_LIST[3]=172.31.2.48

export NUM_NODE=${#NODE_HOST_LIST[@]}

# set up cluster-setting
export NUM_WORKER=4
export NUM_SERVER=1

export NUM_WORKER_PER_NODE=`expr $NUM_WORKER / $NUM_NODE`
export NUM_SERVER_PER_NODE=`expr $NUM_SERVER / $NUM_NODE`

export DEBUG_PRINT=1
export DEBUG_C_PRINT=1

# set up network
# export DOCKER_IMAGE="195164261969.dkr.ecr.us-west-2.amazonaws.com/ooo_byteps:latest"
# export SSH_KEY_PATH="~/.ssh/hanyang_bdsl_oregon.pem"
# export SSH_ID="ubuntu"
# export DMLC_INTERFACE=ens3

export DOCKER_IMAGE="mlsys.duckdns.org:9999/ooo-backprop-byteps:latest"
export SSH_KEY_PATH="~/.ssh/bdsl_rsa"
export SSH_ID="cheezestick"
export DMLC_INTERFACE=enp2s0f1


export ROOT_DIR="/root/OOO_BackProp_BytePS"

export OUTPUT_DIR=$(dirname $(dirname $(dirname $(realpath $0))))


echo ""
echo ""
echo "======================== setup.sh ========================"
echo "MODEL CONFIGURE    :: MODEL_SIZE: "$MODEL_SIZE
echo "MODEL CONFIGURE    :: BATCH_SIZE: "$BATCH_SIZE
echo "MODEL CONFIGURE    :: REVERSE_FIRST_K: "$REVERSE_FIRST_K
echo "MODEL CONFIGURE    :: NUM_TRAINING_STEP: "$NUM_TRAINING_STEP
echo ""
echo "CLUSTER CONFIGURE  :: MASTER_HOST: "$MASTER_HOST
echo "CLUSTER CONFIGURE  :: MASTER_PORT: "$MASTER_PORT
for node_idx in "${!NODE_HOST_LIST[@]}"
do
    NODE_HOST="${NODE_HOST_LIST[$node_idx]}"
    echo "CLUSTER CONFIGURE  :: NODE_HOST_"$node_idx": "$NODE_HOST
done
echo "CLUSTER CONFIGURE  :: NUM_NODE: "$NUM_NODE
echo "CLUSTER CONFIGURE  :: NUM_WORKER: "$NUM_WORKER
echo "CLUSTER CONFIGURE  :: NUM_SERVER: "$NUM_SERVER
echo "CLUSTER CONFIGURE  :: NUM_WORKER_PER_NODE: "$NUM_WORKER_PER_NODE
echo "CLUSTER CONFIGURE  :: NUM_SERVER_PER_NODE: "$NUM_SERVER_PER_NODE
echo "CLUSTER CONFIGURE  :: DEBUG_PRINT: "$DEBUG_PRINT
echo ""
echo "NETWORK CONFIGURE  :: DOCKER_IMAGE: "$DOCKER_IMAGE
echo "NETWORK CONFIGURE  :: SSH_KEY_PATH: "$SSH_KEY_PATH
echo "NETWORK CONFIGURE  :: SSH_ID: "$SSH_ID
echo "NETWORK CONFIGURE  :: DMLC_INTERFACE: "$DMLC_INTERFACE
echo "NETWORK CONFIGURE  :: ROOT_DIR: "$ROOT_DIR
echo ""
echo "RESULT CONFIGURE :: OUTPUT_DIR: "$OUTPUT_DIR
echo "=========================================================="
echo ""
echo ""