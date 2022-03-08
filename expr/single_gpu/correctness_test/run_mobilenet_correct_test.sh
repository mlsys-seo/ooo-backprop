#!/bin/bash

export TF_XLA_FLAGS=--tf_xla_auto_jit=2
export TF_CPP_MIN_LOG_LEVEL=1

for BATCH in 32
do
    for ALPHA in 1 0.5
    do
        for ITER in 40 80
        do
            echo "##########################################################"
            echo "  Batch size : ${BATCH}"
            echo "  Alpha : ${ALPHA}"
	    echo "  Num iters : ${ITER}"
            echo "##########################################################"
            python mobile_base.py ${BATCH} ${ALPHA} ${ITER}

            export DO_OOO_BACKPROP="true"
            export OOO_CAPTURE_OP="cluster_1_1/xla_run"
            export OOO_CAPTURE_ITER=2
            export OOO_USE_SUB_STREAM="true"
            python mobile_ooo.py ${BATCH} ${ALPHA} ${ITER}
            unset DO_OOO_BACKPROP
            unset OOO_CAPTURE_OP
            unset OOO_CAPTURE_ITER
            unset OOO_USE_SUB_STREAM

	    python logit_diff.py
        done
    done
done
