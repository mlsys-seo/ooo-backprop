export grpc_master=localhost

CONFIG_DIR=/workspace/OutOfOrder_Backeprop/bert_config/bert_config
BERT_12=${CONFIG_DIR}/bert_12_config.json
INPUT_DATA=${CONFIG_DIR}/tf_examples.tfrecord

rm -r ./mrpc_test

task_index=0
worker_hosts=$grpc_master:2232

MODEL=${BERT_12}   # BERT_LARGE or BERT_BASE
GPU_SIZE=8            # GPUs per cluster
CLUSTER_SIZE=1        # clusters
GLOBAL_BATCH=480       # global batch size
MICRO_BATCH=40         # micro batch size
MODULO_BATCH=1

PIPELINE_STYLE='gpipe'      # gpipe / fastforward / fastforward_push / modulo

python run_pretraining.py \
  --do_train=true \
  --input_file=${INPUT_DATA} \
  --bert_config_file=${MODEL} \
  --max_seq_length=128 \
  --train_batch_size=${GLOBAL_BATCH} \
  --learning_rate=2e-5 \
  --num_train_steps=10 \
  --output_dir=./mrpc_test/ \
  --gpu_size=${GPU_SIZE} \
  --cluster_size=${CLUSTER_SIZE} \
  --micro_batch_size=${MICRO_BATCH} \
  --task_index=${task_index} \
  --worker_hosts=${worker_hosts} \
  --pipeline_style=${PIPELINE_STYLE} \
  --modulo_batch=${MODULO_BATCH} \

rm -r ./mrpc_test