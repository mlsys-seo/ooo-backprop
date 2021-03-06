#!/bin/bash

export TF_XLA_FLAGS=--tf_xla_auto_jit=2
export TF_CPP_MIN_LOG_LEVEL=1
export DO_OOO_BACKPROP="true"
export OOO_CAPTURE_OP="cluster_1_1/xla_run"
export OOO_CAPTURE_ITER=3
export OOO_NUM_BLOCK_OVERLAP_FORWARD=88
export OOO_OVERLAP_START="B4"
export OOO_OVERLAP_END="B3"
export OOO_USE_SUB_STREAM="true"

BATCH=$1
GROWTH_K=$2

echo "##########################################################"
echo "  Single GPU Experiment - DenseNet121 with OOO-Backprop"
echo "  Batch size : ${BATCH}"
echo "  Growth K : ${GROWTH_K}"
echo "##########################################################"

python /workspace/expr/code/densenet_expr_ooo.py ${BATCH} ${GROWTH_K}

unset TF_XLA_FLAGS
unset TF_CPP_MIN_LOG_LEVEL
unset DO_OOO_BACKPROP
unset OOO_CAPTURE_OP
unset OOO_CAPTURE_ITER
unset OOO_NUM_BLOCK_OVERLAP_FORWARD
unset OOO_OVERLAP_START
unset OOO_OVERLAP_END
unset OOO_USE_SUB_STREAM
