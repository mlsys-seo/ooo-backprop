export REMOTE_ARG=$1
SCRIPT_ROOT_PATH=$(dirname $0)

if [[ $REMOTE_ARG == "preset" ]]
then
    declare -a NODE_HOST_LIST=($NODE_HOSTS_STRING)
else
    source $SCRIPT_ROOT_PATH/setup.sh
fi

DATA_DIR="../code/data"
INPUT_DATA=${DATA_DIR}/tf_examples.tfrecord

echo ""
echo "==================================== setup ===================================="
echo "MODEL CONFIGURE    :: MODEL: "$MODEL
echo "MODEL CONFIGURE    :: GLOBAL_BATCH_SIZE: "$GLOBAL_BATCH_SIZE
echo "MODEL CONFIGURE    :: MICRO_BATCH_SIZE: "$MICRO_BATCH_SIZE
echo "MODEL CONFIGURE    :: MODULO_BATCH_SIZE: "$MODULO_BATCH_SIZE
echo "MODEL CONFIGURE    :: NUM_TRAINING_STEP: "$NUM_TRAINING_STEP
echo "MODEL CONFIGURE    :: TASK: "$TASK
echo ""
echo "CLUSTER CONFIGURE  :: MASTER_HOST: "$MASTER_HOST
echo "CLUSTER CONFIGURE  :: MASTER_PORT: "$MASTER_PORT
for node_idx in "${!NODE_HOST_LIST[@]}"
do
    NODE_HOST="${NODE_HOST_LIST[$node_idx]}"
    echo "CLUSTER CONFIGURE  :: NODE_HOST_"$node_idx": "$NODE_HOST
done
echo "CLUSTER CONFIGURE  :: NUM_NODE: "$NUM_NODE
echo "CLUSTER CONFIGURE  :: NUM_WORKER_PER_NODE: "$NUM_WORKER_PER_NODE
echo ""
echo "NETWORK CONFIGURE  :: DOCKER_IMAGE: "$DOCKER_IMAGE
echo "NETWORK CONFIGURE  :: SSH_KEY_PATH: "$SSH_KEY_PATH
echo "NETWORK CONFIGURE  :: SSH_ID: "$SSH_ID
echo "NETWORK CONFIGURE  :: DMLC_INTERFACE: "$DMLC_INTERFACE
echo "==============================================================================="
echo ""

# set variables

WORKER_HOST_STRING=""
for node_idx in "${!NODE_HOST_LIST[@]}"
do
    NODE_HOST="${NODE_HOST_LIST[$node_idx]}"
    WORKER_HOST_STRING="$WORKER_HOST_STRING$NODE_HOST:$MASTER_PORT,"
done
WORKER_HOST_STRING="${WORKER_HOST_STRING:(0):(-1)}"
MODEL_CONFIG_PATH=$CONFIG_DIR/$MODEL.json


# run scripts

for NODE_IDX in "${!NODE_HOST_LIST[@]}"
do
    NODE_HOST="${NODE_HOST_LIST[$NODE_IDX]}"
    if [ $NODE_IDX == 0 ]
    then
      echo "::: RUN SERVER NODE "$NODE_IDX": "$NODE_HOST" :::"
        python /workspace/OutOfOrder_Backprop/src/run_classifier.py \
            --input_data=$INPUT_DATA \
            --do_train=true \
            --max_seq_length=128 \
            --learning_rate=2e-5 \
            --output_dir=/outputs/ \
            --num_train_steps=${NUM_TRAINING_STEP} \
            --train_batch_size=${GLOBAL_BATCH_SIZE} \
            --micro_batch_size=${MICRO_BATCH_SIZE} \
            --modulo_batch=${MODULO_BATCH_SIZE}
            --bert_config_file=${MODEL_CONFIG_PATH} \
            --gpu_size=${NUM_WORKER_PER_NODE} \
            --cluster_size=${NUM_NODE} \
            --task_index=${NODE_IDX} \
            --worker_hosts=${WORKER_HOST_STRING} \
            --pipeline_style=${PIPELINE_STYLE}
        echo ""
    else
      echo "::: RUN REMOTE WORKER NODE "$NODE_IDX": "$NODE_HOST" :::"
        ssh -i $SSH_KEY_PATH -f $SSH_ID@$NODE_HOST \
            "sudo docker run $DETACH \
                --rm --privileged --ipc=host --net=host --gpus=all \
                -e DMLC_INTERFACE=$DMLC_INTERFACE \
                --name ooo-pipe-$NODE_IDX \
                $DOCKER_IMAGE \
                ./code/OOO_backprop/sub_node.py \
                --task_index=${NODE_IDX} \
                --worker_hosts=$WORKER_HOST_STRING" &
        echo ""
    fi
done