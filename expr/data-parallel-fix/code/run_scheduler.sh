#!/bin/bash

MASTER_HOST=$1
MASTER_PORT=$2
NUM_WORKER=$3
NUM_SERVER=$4

export DMLC_ROLE=scheduler
export DMLC_NUM_WORKER=$NUM_WORKER
export DMLC_NUM_SERVER=$NUM_SERVER

export DMLC_PS_ROOT_URI=$MASTER_HOST # the scheduler IP
export DMLC_PS_ROOT_PORT=$MASTER_PORT  # the scheduler port

export DMLC_NODE_HOST=$MASTER_HOST

#export BYTEPS_SERVER_ENABLE_SCHEDULE=1

echo ""
echo ""
echo "################### Scheduler Started ####################"
echo "SCHEDULER MASTER_HOST: "$MASTER_HOST
echo "SCHEDULER MASTER_PORT: "$MASTER_PORT
echo "SCHEDULER NUM_SERVER: "$NUM_SERVER
echo "SCHEDULER NUM_WORKER: "$NUM_WORKER
echo ""
echo "DMLC DMLC_ROLE: "$DMLC_ROLE
echo "DMLC DMLC_NUM_SERVER: "$DMLC_NUM_SERVER
echo "DMLC DMLC_NUM_WORKER "$DMLC_NUM_WORKER
echo "DMLC DMLC_PS_ROOT_URI: "$DMLC_PS_ROOT_URI # the scheduler IP
echo "DMLC DMLC_PS_ROOT_PORT: "$DMLC_PS_ROOT_PORT  # the scheduler port
echo "DMLC DMLC_NODE_HOST: "$DMLC_NODE_HOST
echo "DMLC DMLC_INTERFACE: "$DMLC_INTERFACE
echo "##########################################################"
echo ""


bpslaunch