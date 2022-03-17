#!/bin/bash

export LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/local/cuda/lib64

MODEL_SIZE=$1
BATCH_SIZE=$2
REVERSE_FIRST_K=$3
MASTER_HOST=$4
MASTER_PORT=$5
NODE_HOST=$6
NUM_SERVER=$7
NUM_WORKER=$8
#NUM_WORKER=2
INDEX=$9
GPU_IDX="${10}"
DEBUG_PRINT="${11}"
NUM_TRAINING_STEP="${12}"

echo ""
echo ""
echo "################### Node(Worker) Start ###################"
echo "S_W MODEL_SIZE "$MODEL_SIZE
echo "S_W BATCH_SIZE "$BATCH_SIZE
echo "S_W REVERSE_FIRST_K "$REVERSE_FIRST_K
echo "S_W MASTER_HOST "$MASTER_HOST
echo "S_W MASTER_PORT "$MASTER_PORT
echo "S_W NODE_HOST "$NODE_HOST
echo "S_W NUM_WORKER "$NUM_WORKER
echo "S_W NUM_SERVER "$NUM_SERVER
echo "S_W INDEX "$INDEX
echo "S_W GPU_IDX "$GPU_IDX
echo "S_W DEBUG_PRINT "$DEBUG_PRINT
echo "S_W NUM_TRAINING_STEP" $NUM_TRAINING_STEP
echo "##########################################################"
echo ""

#export BYTEPS_SERVER_ENGINE_THREAD=8
export CUDA_VISIBLE_DEVICES=$GPU_IDX

if [ $INDEX -lt $NUM_SERVER ]
then
#     echo "###################### Launch Server #####################"
    $(dirname $0)/run_server.sh $NODE_HOST $MASTER_HOST $MASTER_PORT $NUM_WORKER $NUM_SERVER &
fi

export DMLC_ROLE=worker

export DMLC_PS_ROOT_URI=$MASTER_HOST # the scheduler IP
export DMLC_PS_ROOT_PORT=$MASTER_PORT # the scheduler port

export DMLC_NUM_WORKER=$NUM_WORKER
export DMLC_NUM_SERVER=$NUM_SERVER

export DMLC_NODE_HOST=$NODE_HOST

export DMLC_WORKER_ID=$INDEX

export BYTEPS_SERVER_ENABLE_SCHEDULE=1

# bpslaunch \
# nsys profile \
# -t cuda,osrt,nvtx,cudnn,cublas \
# -o ./outputs/profile-$INDEX.qdstrm \
# python3 run.py \
#     --model_size $MODEL_SIZE \
#     --batch_size $BATCH_SIZE \
#     --reverse_first_k $REVERSE_FIRST_K \
#     --debug_print $DEBUG_PRINT

#bpslaunch nvprof -fo $(dirname $0)/outputs/profile-$INDEX.nvpp python3 $(dirname $0)/conv_test.py \
bpslaunch python3 $(dirname $0)/run.py \
    --model_size $MODEL_SIZE \
    --batch_size $BATCH_SIZE \
    --reverse_first_k $REVERSE_FIRST_K \
    --debug_print $DEBUG_PRINT \
    --num_training_step $NUM_TRAINING_STEP
