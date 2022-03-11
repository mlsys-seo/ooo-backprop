#!/bin/bash

CODE_DIR=$(dirname $(realpath $0))
ROOT_DIR=$(dirname $CODE_DIR)

TASK=${1}   # "finetune" or "pretrain"
MODEL=${2}  # e.g. "bert_24"  
PIPELINE_STYLE=${3}  # "gpipe", "fastforward_push", or "modulo" (modulo means that both fastforwarding and modulo allocation are applied)
NUM_TRAINING_STEP=${4}
GLOBAL_BATCH_SIZE=${5} # mini-batch size
MICRO_BATCH_SIZE=${6}  # micro-batch size
MODULO_BATCH_SIZE=${7} # The number of transformers to group when applying modulo allocation.
                       # It is 1 for NVLink/PCIe interconnect and 2 or higher for 10Gb Ethernet interconnect. 
NUM_WORKER_PER_NODE=${8} # The number of GPUs per node
NUM_NODE=${9}
MASTER_HOST="${10}"  # Master IP
WORKER_HOST_STRING="${11}"  # IP of the current worker
NODE_IDX="${12}"     # Worker ID starting from zero and incremented by one.


INPUT_DATA=$ROOT_DIR/data/tf_examples.tfrecord
DATA_DIR=$ROOT_DIR/data
MODEL_CONFIG_PATH=$CODE_DIR/bert_config/$MODEL.json



echo ""
echo ""
echo "################### Node(Worker) Start ###################"
echo "S_W TASK "$TASK
echo "S_W MODEL "$MODEL
echo "S_W PIPELINE_STYLE "$PIPELINE_STYLE
echo "S_W GLOBAL_BATCH_SIZE "$GLOBAL_BATCH_SIZE
echo "S_W MICRO_BATCH_SIZE "$MICRO_BATCH_SIZE
echo "S_W MODULO_BATCH_SIZE "$MODULO_BATCH_SIZE
echo "S_W NUM_WORKER_PER_NODE "$NUM_WORKER_PER_NODE
echo "S_W NUM_NODE "$NUM_NODE
echo "S_W MASTER_HOST "$MASTER_HOST
echo "S_W WORKER_HOST_STRING "$WORKER_HOST_STRING
echo "S_W NODE_IDX "$NODE_IDX
echo "S_W INPUT_DATA "$INPUT_DATA
echo "S_W DATA_DIR "$DATA_DIR
echo "S_W MODEL_CONFIG_PATH "$MODEL_CONFIG_PATH
echo "##########################################################"
echo ""

export grpc_master=$MASTER_HOST

if [[ "$TASK" == "pretrain" ]]
then
    # pretrining
    python3 $CODE_DIR/OOO_backprop/run_pretraining.py \
            --input_file=$INPUT_DATA \
            --do_train=true \
            --max_seq_length=128 \
            --learning_rate=2e-5 \
            --output_dir=/outputs/ \
            --num_train_steps=${NUM_TRAINING_STEP} \
            --train_batch_size=${GLOBAL_BATCH_SIZE} \
            --micro_batch_size=${MICRO_BATCH_SIZE} \
            --modulo_batch=${MODULO_BATCH_SIZE} \
            --bert_config_file=${MODEL_CONFIG_PATH} \
            --gpu_size=${NUM_WORKER_PER_NODE} \
            --cluster_size=${NUM_NODE} \
            --task_index=${NODE_IDX} \
            --worker_hosts=${WORKER_HOST_STRING} \
            --pipeline_style=${PIPELINE_STYLE}
else
    # finetuning
    python3 $CODE_DIR/OOO_backprop/run_classifier.py \
        --task_name=MRPC \
        --data_dir=${DATA_DIR} \
        --vocab_file=${DATA_DIR}/vocab.txt \
        --do_train=true \
        --max_seq_length=128 \
        --learning_rate=2e-5 \
        --output_dir=/outputs/ \
        --num_train_steps=${NUM_TRAINING_STEP} \
        --train_batch_size=${GLOBAL_BATCH_SIZE} \
        --micro_batch_size=${MICRO_BATCH_SIZE} \
        --modulo_batch=${MODULO_BATCH_SIZE} \
        --bert_config_file=${MODEL_CONFIG_PATH} \
        --gpu_size=${NUM_WORKER_PER_NODE} \
        --cluster_size=${NUM_NODE} \
        --task_index=${NODE_IDX} \
        --worker_hosts=${WORKER_HOST_STRING} \
        --pipeline_style=${PIPELINE_STYLE}
fi






