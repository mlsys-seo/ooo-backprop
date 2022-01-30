export grpc_master=localhost
grpc_worker0=${SUB_NODE_IP0}
grpc_worker1=${SUB_NODE_IP1}
grpc_worker2=${SUB_NODE_IP2}

CONFIG_DIR=/workspace/OutOfOrder_Backprop/bert_config
BERT_24=${CONFIG_DIR}/bert_24_config.json
INPUT_DATA=${CONFIG_DIR}/tf_examples.tfrecord
DATA_DIR=/workspace/OutOfOrder_Backprop/MRPC

rm -rf ./mrpc_profile

task_index=0
worker_hosts=$grpc_master:2232,$grpc_worker0:2232,$grpc_worker1:2232,$grpc_worker2:2232

MODEL=${BERT_24}
GPU_SIZE=1            # GPUs per cluster
CLUSTER_SIZE=4        # clusters
GLOBAL_BATCH=96       # global batch size
MICRO_BATCH=8         # micro batch size

PIPELINE_STYLE='gpipe'

python /workspace/OutOfOrder_Backprop/src/run_classifier.py \
  --task_name=MRPC \
  --do_train=true \
  --data_dir=${DATA_DIR} \
  --vocab_file=${CONFIG_DIR}/vocab.txt \
  --bert_config_file=${MODEL} \
  --max_seq_length=128 \
  --train_batch_size=${GLOBAL_BATCH} \
  --learning_rate=2e-5 \
  --num_train_epochs=1 \
  --num_train_steps=10 \
  --output_dir=./mrpc_test/ \
  --gpu_size=${GPU_SIZE} \
  --cluster_size=${CLUSTER_SIZE} \
  --micro_batch_size=${MICRO_BATCH} \
  --task_index=${task_index} \
  --worker_hosts=${worker_hosts} \
  --pipeline_style=${PIPELINE_STYLE}
  
rm -r ./mrpc_test