#!/bin/bash

source $(dirname $0)/setup.sh

echo "Killing Containers"

ssh -i $SSH_KEY_PATH $SSH_ID@$MASTER_HOST \
    docker kill ooo-scheduler

for node_idx in "${!NODE_HOST_LIST[@]}"
do
    NODE_HOST="${NODE_HOST_LIST[$node_idx]}"
    for ((local_idx = 0 ; local_idx < $NUM_WORKER_PER_NODE ; local_idx++))
    do
        PRE_INDEX=`expr $node_idx \* $NUM_WORKER_PER_NODE`
        INDEX=`expr $PRE_INDEX + $local_idx`

        ssh -i $SSH_KEY_PATH $SSH_ID@$NODE_HOST \
            docker kill ooo-worker-$INDEX
    done
done

echo "KILL FINISHED"