#!/bin/bash


NODE_HOST=$1
MASTER_HOST=$2
MASTER_PORT=$3
NUM_WORKER=$4
NUM_SERVER=$5

echo ""
echo ""
echo "###################### Server Start ######################"
echo "Parameter Server - NODE_HOST: "$NODE_HOST
echo "Parameter Server - MASTER_HOST: "$MASTER_HOST
echo "Parameter Server - MASTER_PORT: "$MASTER_PORT
echo "Parameter Server - NUM_WORKER: "$NUM_WORKER
echo "Parameter Server - NUM_SERVER: "$NUM_SERVER
echo "##########################################################"
echo ""

export DMLC_ROLE=server

export DMLC_PS_ROOT_URI=$MASTER_HOST # the scheduler IP #TODO SET MASTER IP
export DMLC_PS_ROOT_PORT=$MASTER_PORT  # the scheduler port

export DMLC_NUM_WORKER=$NUM_WORKER
export DMLC_NUM_SERVER=$NUM_SERVER

export DMLC_NODE_HOST=$NODE_HOST

bpslaunch