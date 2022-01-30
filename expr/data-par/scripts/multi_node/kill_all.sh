#!/bin/bash

if [[ $REMOTE_ARG == "preset" ]]
then
    declare -a NODE_HOST_LIST=($NODE_HOSTS_STRING)
else
    source $(dirname $0)/setup.sh
fi


echo ""
echo ""
echo "=============================== Kill Containers ==============================="

echo "::: KILL SCHEDULER CONTAINER :::"
ssh -i $SSH_KEY_PATH $SSH_ID@$MASTER_HOST \
    sudo docker kill ooo-scheduler

for node_idx in "${!NODE_HOST_LIST[@]}"
do
    NODE_HOST="${NODE_HOST_LIST[$node_idx]}"
    echo "::: KILL CONTAINER AT "$NODE_HOST" :::"

    for ((local_idx = 0 ; local_idx < $NUM_WORKER_PER_NODE ; local_idx++))
    do
        PRE_INDEX=`expr $node_idx \* $NUM_WORKER_PER_NODE`
        INDEX=`expr $PRE_INDEX + $local_idx`

        ssh -i $SSH_KEY_PATH $SSH_ID@$NODE_HOST \
            sudo docker kill ooo-worker-$INDEX
    done
done

echo "==============================================================================="
echo ""
echo ""