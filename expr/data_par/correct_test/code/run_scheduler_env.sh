#!/bin/bash

echo ""
echo ""
echo "################### Scheduler Started ####################"
echo "MASTER_HOST: "$MASTER_HOST
echo "MASTER_PORT: "$MASTER_PORT
echo "NUM_WORKER: "$NUM_WORKER
echo "NUM_SERVER: "$NUM_SERVER
echo "##########################################################"
echo ""

export DMLC_ROLE=scheduler
export DMLC_NUM_WORKER=$NUM_WORKER
export DMLC_NUM_SERVER=$NUM_SERVER

export DMLC_PS_ROOT_URI=$MASTER_HOST # the scheduler IP
export DMLC_PS_ROOT_PORT=$MASTER_PORT  # the scheduler port

export DMLC_NODE_HOST=$MASTER_HOST
#export BYTEPS_SERVER_ENABLE_SCHEDULE=1

bpslaunch