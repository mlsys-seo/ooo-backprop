#!/bin/bash
CODE_DIR=$(dirname $(realpath $0))
ROOT_DIR=$(dirname $CODE_DIR)

export LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/local/cuda/lib64

MODEL_SIZE=$1  # e.g. 50, 101 
BATCH_SIZE=$2
NUM_TRAINING_STEP=$3
REVERSE_FIRST_K=$4
MASTER_HOST=$5  # Master node IP
MASTER_PORT=$6
NODE_HOST=$7    # IP of the current worker
NUM_WORKER=$8
NUM_SERVER=$9   # Number of parameter servers. e.g. NUM_WORKER/4
NUM_SERVER_PER_NODE="${10}"  # Number of parameter servers per node (machine).
INDEX="${11}"                # Worker ID (starting from 0 and incremented by 1).
GPU_IDX="${12}"              # The GPU index for the worker to use.
DEBUG_PRINT="${13}"          # 1 or 0
export DEBUG_C_PRINT="${14}"

export DMLC_ROLE=worker

export DMLC_PS_ROOT_PORT=$MASTER_PORT # the scheduler port
export DMLC_PS_ROOT_URI=$MASTER_HOST # the scheduler IP

export DMLC_NUM_WORKER=$NUM_WORKER
export DMLC_NUM_SERVER=$NUM_SERVER

export DMLC_NODE_HOST=$NODE_HOST
export DMLC_WORKER_ID=$INDEX

export BYTEPS_SERVER_ENABLE_SCHEDULE=1

echo ""
echo ""
echo "################### Node(Worker) Start ###################"
echo "S_W MODEL_SIZE "$MODEL_SIZE
echo "S_W BATCH_SIZE "$BATCH_SIZE
echo "S_W NUM_TRAINING_STEP "$NUM_TRAINING_STEP
echo "S_W REVERSE_FIRST_K "$REVERSE_FIRST_K
echo "S_W MASTER_HOST "$MASTER_HOST
echo "S_W MASTER_PORT "$MASTER_PORT
echo "S_W NODE_HOST "$NODE_HOST
echo "S_W NUM_WORKER "$NUM_WORKER
echo "S_W NUM_SERVER "$NUM_SERVER
echo "S_W NUM_SERVER_PER_NODE "$NUM_SERVER_PER_NODE
echo "S_W INDEX "$INDEX
echo "S_W GPU_IDX "$GPU_IDX
echo "S_W DEBUG_PRINT "$DEBUG_PRINT
echo "S_W DEBUG_C_PRINT "$DEBUG_C_PRINT
echo ""
echo "DMLC DMLC_ROLE: "$DMLC_ROLE
echo "DMLC DMLC_PS_ROOT_URI: "$DMLC_PS_ROOT_URI
echo "DMLC DMLC_PS_ROOT_PORT: "$DMLC_PS_ROOT_PORT
echo "DMLC DMLC_NUM_WORKER: "$DMLC_NUM_WORKER
echo "DMLC DMLC_NUM_SERVER: "$DMLC_NUM_SERVER
echo "DMLC DMLC_NODE_HOST: "$DMLC_NODE_HOST
echo "DMLC DMLC_WORKER_ID: "$DMLC_WORKER_ID
echo "DMLC BYTEPS_SERVER_ENABLE_SCHEDULE: "$BYTEPS_SERVER_ENABLE_SCHEDULE
echo "DMLC DMLC_INTERFACE: "$DMLC_INTERFACE
echo "##########################################################"
echo ""

#export BYTEPS_SERVER_ENGINE_THREAD=8
export CUDA_VISIBLE_DEVICES=$GPU_IDX

if [ $GPU_IDX -lt $NUM_SERVER_PER_NODE ] # GPU_IDX is Local Index
then
    $CODE_DIR/run_server.sh $NODE_HOST $MASTER_HOST $MASTER_PORT $NUM_WORKER $NUM_SERVER &
fi

if [[ $NVPROF == "1" ]] # GPU_IDX is Local Index
then
    bpslaunch nvprof -fo $ROOT_DIR/outputs/profile-$INDEX.nvpp python3 $CODE_DIR/run.py \
        --model_size $MODEL_SIZE \
        --batch_size $BATCH_SIZE \
        --reverse_first_k $REVERSE_FIRST_K \
        --debug_print $DEBUG_PRINT \
        --num_training_step $NUM_TRAINING_STEP

else
    bpslaunch python3 $CODE_DIR/run.py \
        --model_size $MODEL_SIZE \
        --batch_size $BATCH_SIZE \
        --reverse_first_k $REVERSE_FIRST_K \
        --debug_print $DEBUG_PRINT \
        --num_training_step $NUM_TRAINING_STEP
fi

echo "################### Node(Worker) Done ###################"
