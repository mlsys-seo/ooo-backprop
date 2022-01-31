#!/bin/bash

export TF_XLA_FLAGS=--tf_xla_auto_jit=2

BATCH=$1
GROWTH_K=$2

echo "##########################################################"
echo "  Single GPU Experiment - DenseNet121 with Tensorflow XLA"
echo "  Batch size : ${BATCH}"
echo "  Growth K : ${GROWTH_K}"
echo "##########################################################"

python /workspace/expr/code/densenet_expr_base.py ${BATCH} ${GROWTH_K}

unset TF_XLA_FLAGS
