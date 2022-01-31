#!/bin/bash

export TF_XLA_FLAGS=--tf_xla_auto_jit=2

BATCH=$1
ALPHA=$2

echo "##########################################################"
echo "  Single GPU Experiment - MobileNet v3 with Tensorflow XLA"
echo "  Batch size : ${BATCH}"
echo "  Growth K : ${ALPHA}"
echo "##########################################################"

python ../code/mobilenet_v3_expr_base.py ${BATCH} ${ALPHA}

unset TF_XLA_FLAGS
