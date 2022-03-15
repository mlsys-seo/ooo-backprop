#!/bin/bash

export TF_XLA_FLAGS=--tf_xla_auto_jit=2
export TF_CPP_MIN_LOG_LEVEL=1

for BATCH in 8 16
do
    for GROWTH_K in 12
    do
        for ITER in 40 80
        do
            echo "##########################################################"
            echo "  Batch size : ${BATCH}"
            echo "  Growth K : ${GROWTH_K}"
	          echo "  Num iters : ${ITER}"
            echo "##########################################################"
            python /workspace/correctness_test/code/dense_base.py ${BATCH} ${GROWTH_K} ${ITER}

            export DO_OOO_BACKPROP="true"
            export OOO_CAPTURE_OP="cluster_1_1/xla_run"
            export OOO_CAPTURE_ITER=2
            export OOO_NUM_BLOCK_OVERLAP_FORWARD=88
            export OOO_OVERLAP_START="B4"
            export OOO_OVERLAP_END="B3"
            export OOO_USE_SUB_STREAM="true"
            python /workspace/correctness_test/code/dense_ooo.py ${BATCH} ${GROWTH_K} ${ITER}
            unset DO_OOO_BACKPROP
            unset OOO_CAPTURE_OP
            unset OOO_CAPTURE_ITER
            unset OOO_NUM_BLOCK_OVERLAP_FORWARD
	          unset OOO_OVERLAP_START
	          unset OOO_OVERLAP_END
            unset OOO_USE_SUB_STREAM

	    python /workspace/correctness_test/code/logit_diff.py
        done
    done
done
