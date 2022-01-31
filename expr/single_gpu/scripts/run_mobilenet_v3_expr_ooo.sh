#!/bin/bash

export TF_XLA_FLAGS=--tf_xla_auto_jit=2
export DO_OOO_BACKPROP="true"
export OOO_CAPTURE_OP="cluster_1_1/xla_run"
export OOO_CAPTURE_ITER=3
export OOO_USE_SUB_STREAM="true"

BATCH=$1
ALPHA=$2

echo "##########################################################"
echo "  Single GPU Experiment - MobileNet v3 with OOO-Backprop"
echo "  Batch size : ${BATCH}"
echo "  Growth K : ${ALPHA}"
echo "##########################################################"

python /workspace/expr/code/mobilenet_v3_expr_ooo.py ${BATCH} ${ALPHA}

unset TF_XLA_FLAGS
unset DO_OOO_BACKPROP
unset OOO_CAPTURE_OP
unset OOO_CAPTURE_ITER
unset OOO_USE_SUB_STREAM
